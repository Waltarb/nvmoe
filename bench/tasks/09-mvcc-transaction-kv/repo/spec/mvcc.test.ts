import { describe, it, expect } from "vitest";
import { MvccStore } from "../src/mvcc";

describe("MVCC Store (visible)", () => {
  it("commits writes and reads snapshot", () => {
    const store = new MvccStore();
    const tx1 = store.beginTransaction();
    tx1.set("name", "Alice");
    tx1.commit();

    expect(store.getCommitted("name")).toBe("Alice");

    const tx2 = store.beginTransaction();
    expect(tx2.get("name")).toBe("Alice");
    tx2.set("name", "Bob");
    expect(tx2.get("name")).toBe("Bob");
    tx2.commit();

    expect(store.getCommitted("name")).toBe("Bob");
  });

  it("reads from uncommitted writes locally", () => {
    const store = new MvccStore();
    const tx = store.beginTransaction();
    tx.set("k", "v1");
    expect(tx.get("k")).toBe("v1");
    tx.delete("k");
    expect(tx.get("k")).toBeUndefined();
    tx.rollback();
  });
});
