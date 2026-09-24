import { describe, it, expect, vi } from "vitest";
import { BatchScheduler } from "../src/scheduler";

describe("BatchScheduler (visible)", () => {
  it("batches requests and resolves values", async () => {
    const batchFn = vi.fn(async (keys: string[]) => {
      const map = new Map<string, number>();
      for (const k of keys) map.set(k, k.length);
      return map;
    });

    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 10, maxWaitMs: 20 });
    const p1 = scheduler.schedule("apple");
    const p2 = scheduler.schedule("banana");

    const [v1, v2] = await Promise.all([p1, p2]);
    expect(v1).toBe(5);
    expect(v2).toBe(6);
    expect(batchFn).toHaveBeenCalledTimes(1);
    expect(batchFn).toHaveBeenCalledWith(["apple", "banana"]);
  });

  it("dispatches immediately when maxBatchSize is reached", async () => {
    const batchFn = vi.fn(async (keys: number[]) => new Map(keys.map(k => [k, k * 2])));
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 2, maxWaitMs: 1000 });

    const p1 = scheduler.schedule(10);
    const p2 = scheduler.schedule(20);

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toBe(20);
    expect(r2).toBe(40);
    expect(batchFn).toHaveBeenCalledTimes(1);
  });

  it("deduplicates identical keys within the same batch", async () => {
    const batchFn = vi.fn(async (keys: string[]) => new Map(keys.map(k => [k, `res-${k}`])));
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 20 });

    const p1 = scheduler.schedule("item1");
    const p2 = scheduler.schedule("item1");

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toBe("res-item1");
    expect(r2).toBe("res-item1");
    expect(batchFn).toHaveBeenCalledTimes(1);
    expect(batchFn).toHaveBeenCalledWith(["item1"]);
  });
});
