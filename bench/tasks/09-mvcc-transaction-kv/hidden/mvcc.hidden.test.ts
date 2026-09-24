import { describe, it, expect } from "vitest";
import { MvccStore } from "../src/mvcc";
import { SerializationConflictError } from "../src/types";

describe("MVCC Store (hidden)", () => {
  it("detects write-write conflict (First-Committer-Wins)", () => {
    const store = new MvccStore();
    const initTx = store.beginTransaction();
    initTx.set("counter", "0");
    initTx.commit();

    const txA = store.beginTransaction();
    const txB = store.beginTransaction();

    txA.set("counter", "1");
    txB.set("counter", "2");

    txA.commit();

    expect(() => txB.commit()).toThrow(SerializationConflictError);
  });

  it("detects conflict when overlapping transaction deletes the key", () => {
    const store = new MvccStore();
    const initTx = store.beginTransaction();
    initTx.set("key1", "val1");
    initTx.commit();

    const txA = store.beginTransaction();
    const txB = store.beginTransaction();

    txA.delete("key1");
    txB.set("key1", "val2");

    txA.commit();

    expect(() => txB.commit()).toThrow(SerializationConflictError);
  });

  it("isolates transactions from concurrent commits", () => {
    const store = new MvccStore();
    const txA = store.beginTransaction();

    const txB = store.beginTransaction();
    txB.set("secret", "42");
    txB.commit();

    // txA started before txB committed, so txA must NOT see "secret"
    expect(txA.get("secret")).toBeUndefined();
    txA.rollback();

    // New transaction after commit sees it
    const txC = store.beginTransaction();
    expect(txC.get("secret")).toBe("42");
    txC.rollback();
  });

  it("throws error when operating on a closed transaction", () => {
    const store = new MvccStore();
    const tx = store.beginTransaction();
    tx.set("a", "1");
    tx.commit();

    expect(() => tx.get("a")).toThrow(/Transaction is closed/);
    expect(() => tx.set("a", "2")).toThrow(/Transaction is closed/);
    expect(() => tx.delete("a")).toThrow(/Transaction is closed/);
    expect(() => tx.commit()).toThrow(/Transaction is closed/);
    expect(() => tx.rollback()).toThrow(/Transaction is closed/);
  });

  it("vacuums versions older than the oldest active snapshot", () => {
    const store = new MvccStore();

    // Create 3 committed versions of "score"
    for (let i = 1; i <= 3; i++) {
      const tx = store.beginTransaction();
      tx.set("score", String(i * 10));
      tx.commit();
    }

    // Open transaction holding snapshot at version 3
    const longRunningTx = store.beginTransaction();
    expect(longRunningTx.readSnapshotVersion).toBe(3);

    // Commit 2 more versions
    for (let i = 4; i <= 5; i++) {
      const tx = store.beginTransaction();
      tx.set("score", String(i * 10));
      tx.commit();
    }

    // Vacuum should prune versions 1 and 2 (retaining version 3 for longRunningTx, and 4, 5)
    const pruned = store.vacuum();
    expect(pruned).toBe(2);

    expect(longRunningTx.get("score")).toBe("30");
    longRunningTx.rollback();

    // Now no active transactions, vacuuming should prune version 3 and 4, keeping only latest (5)
    const pruned2 = store.vacuum();
    expect(pruned2).toBe(2);
    expect(store.getCommitted("score")).toBe("50");
  });

  it("completely removes tombstoned keys with no newer versions upon vacuum", () => {
    const store = new MvccStore();
    const t1 = store.beginTransaction();
    t1.set("temp", "foo");
    t1.commit();

    const t2 = store.beginTransaction();
    t2.delete("temp");
    t2.commit();

    // No active tx, vacuum should prune the set and delete tombstone completely
    const pruned = store.vacuum();
    expect(pruned).toBeGreaterThanOrEqual(1);
    expect(store.getCommitted("temp")).toBeUndefined();
  });
});
