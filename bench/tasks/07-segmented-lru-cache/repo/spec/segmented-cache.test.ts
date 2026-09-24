import { describe, it, expect } from "vitest";
import { SegmentedCache } from "../src/segmented-cache";

describe("SegmentedCache (visible)", () => {
  it("inserts into probation and retrieves value", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 2,
      protectedCapacity: 2,
      ghostCapacity: 2,
    });

    cache.set("a", 1);
    expect(cache.get("a")).toBe(1);
    const sz = cache.size();
    expect(sz.protected).toBe(1);
    expect(sz.probation).toBe(0);
  });

  it("evicts from probation to ghost when exceeding probation capacity", () => {
    const cache = new SegmentedCache<string, number>({
      probationCapacity: 2,
      protectedCapacity: 2,
      ghostCapacity: 2,
    });

    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3); // 'a' evicted to ghost

    expect(cache.get("b")).toBe(2);
    expect(cache.get("c")).toBe(3);
    expect(cache.get("a")).toBeUndefined();

    const sz = cache.size();
    expect(sz.ghost).toBe(1);
  });
});
