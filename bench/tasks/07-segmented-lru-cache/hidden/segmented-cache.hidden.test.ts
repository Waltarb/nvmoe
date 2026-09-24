import { describe, it, expect } from "vitest";
import { SegmentedCache } from "../src/segmented-cache";

describe("SegmentedCache (hidden)", () => {
  it("promotes ghost key directly to protected on set", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 2,
      protectedCapacity: 2,
      ghostCapacity: 2,
    });

    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3); // 'a' moves to ghost

    expect(cache.size().ghost).toBe(1);

    // Now re-insert 'a'
    cache.set("a", 100);
    const sz = cache.size();
    expect(sz.protected).toBe(1);
    expect(sz.ghost).toBe(0);
    expect(cache.get("a")).toBe(100);
  });

  it("handles demotion from protected to probation when protected overflows", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 1,
      protectedCapacity: 2,
      ghostCapacity: 2,
    });

    // Fill protected with 'a' and 'b'
    cache.set("a", 1);
    cache.get("a"); // protected: ['a']
    cache.set("b", 2);
    cache.get("b"); // protected: ['a', 'b'] (b is MRU)

    expect(cache.size().protected).toBe(2);

    // Promote 'c' from probation to protected
    cache.set("c", 3);
    cache.get("c"); // 'c' promoted, protected exceeds 2! LRU ('a') demoted to probation

    const sz = cache.size();
    expect(sz.protected).toBe(2);
    expect(sz.probation).toBe(1);
    expect(cache.get("a")).toBe(1); // accessing 'a' promotes it back
  });

  it("drops oldest ghost entry when ghostCapacity is exceeded", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 1,
      protectedCapacity: 1,
      ghostCapacity: 2,
    });

    cache.set("a", 1);
    cache.set("b", 2); // 'a' enters ghost
    cache.set("c", 3); // 'b' enters ghost; ghost has ['a', 'b']
    cache.set("d", 4); // 'c' enters ghost; 'a' evicted from ghost

    expect(cache.size().ghost).toBe(2);

    // 'a' is no longer in ghost, so setting 'a' goes into probation, NOT protected
    cache.set("a", 10);
    expect(cache.size().protected).toBe(0);
    expect(cache.size().probation).toBe(1);
  });

  it("handles TTL expiration and does not move expired item to ghost", async () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 2,
      protectedCapacity: 2,
      ghostCapacity: 2,
      defaultTtlMs: 25,
    });

    cache.set("quick", 99);
    expect(cache.has("quick")).toBe(true);

    await new Promise(r => setTimeout(r, 40));

    expect(cache.has("quick")).toBe(false);
    expect(cache.get("quick")).toBeUndefined();
    expect(cache.size().ghost).toBe(0);
    expect(cache.size().probation).toBe(0);
  });

  it("deletes items properly from all segments", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 1,
      protectedCapacity: 1,
      ghostCapacity: 1,
    });

    cache.set("a", 1);
    cache.get("a"); // in protected
    cache.set("b", 2); // in probation
    cache.set("c", 3); // 'b' into ghost

    expect(cache.delete("a")).toBe(true);
    expect(cache.delete("b")).toBe(true);
    expect(cache.delete("nonexistent")).toBe(false);

    expect(cache.size().protected).toBe(0);
    expect(cache.size().ghost).toBe(0);
  });

  it("clear() resets all structures", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 2,
      protectedCapacity: 2,
      ghostCapacity: 2,
    });

    cache.set("x", 1);
    cache.get("x");
    cache.set("y", 2);
    cache.clear();

    expect(cache.size()).toEqual({ probation: 0, protected: 0, ghost: 0 });
    expect(cache.get("x")).toBeUndefined();
  });
});
