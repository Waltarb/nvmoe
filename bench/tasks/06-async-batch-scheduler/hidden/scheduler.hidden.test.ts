import { describe, it, expect, vi } from "vitest";
import { BatchScheduler } from "../src/scheduler";

describe("BatchScheduler (hidden)", () => {
  it("rejects immediately if signal is already aborted", async () => {
    const batchFn = vi.fn(async () => new Map<string, number>());
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 50 });

    const controller = new AbortController();
    controller.abort(new Error("pre-aborted"));

    await expect(scheduler.schedule("a", controller.signal)).rejects.toThrow("pre-aborted");
    expect(scheduler.pendingCount).toBe(0);
    expect(batchFn).not.toHaveBeenCalled();
  });

  it("handles abort before batch dispatch", async () => {
    const batchFn = vi.fn(async () => new Map<string, number>());
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 100 });

    const controller = new AbortController();
    const p = scheduler.schedule("k1", controller.signal);
    expect(scheduler.pendingCount).toBe(1);

    controller.abort(new Error("user cancelled"));
    await expect(p).rejects.toThrow("user cancelled");
    expect(scheduler.pendingCount).toBe(0);

    await new Promise(r => setTimeout(r, 150));
    expect(batchFn).not.toHaveBeenCalled();
  });

  it("partially aborts when duplicate callers exist for the same key", async () => {
    const batchFn = vi.fn(async (keys: string[]) => new Map(keys.map(k => [k, 42])));
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 50 });

    const c1 = new AbortController();
    const p1 = scheduler.schedule("sharedKey", c1.signal);
    const p2 = scheduler.schedule("sharedKey"); // not aborted

    c1.abort(new Error("c1 aborted"));
    await expect(p1).rejects.toThrow("c1 aborted");

    const v2 = await p2;
    expect(v2).toBe(42);
    expect(batchFn).toHaveBeenCalledWith(["sharedKey"]);
  });

  it("removes abort listener after completion", async () => {
    const batchFn = vi.fn(async (keys: string[]) => new Map(keys.map(k => [k, 100])));
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 20 });

    const controller = new AbortController();
    let listenerCount = 0;
    const origAdd = controller.signal.addEventListener.bind(controller.signal);
    const origRemove = controller.signal.removeEventListener.bind(controller.signal);

    controller.signal.addEventListener = ((type: string, listener: any, opts: any) => {
      listenerCount++;
      return origAdd(type, listener, opts);
    }) as any;

    controller.signal.removeEventListener = ((type: string, listener: any, opts: any) => {
      listenerCount--;
      return origRemove(type, listener, opts);
    }) as any;

    const res = await scheduler.schedule("key1", controller.signal);
    expect(res).toBe(100);
    expect(listenerCount).toBe(0);
  });

  it("rejects with specific error when key is missing in returned map", async () => {
    const batchFn = vi.fn(async () => new Map<string, number>([["a", 1]])); // missing 'b'
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 20 });

    const pa = scheduler.schedule("a");
    const pb = scheduler.schedule("b");

    expect(await pa).toBe(1);
    await expect(pb).rejects.toThrow(/Missing key in batch result: b/);
  });

  it("propagates batchFn failure to all callers in the batch", async () => {
    const batchErr = new Error("Database down");
    const batchFn = vi.fn(async () => { throw batchErr; });
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 5, maxWaitMs: 20 });

    const p1 = scheduler.schedule("k1");
    const p2 = scheduler.schedule("k2");

    await expect(p1).rejects.toThrow("Database down");
    await expect(p2).rejects.toThrow("Database down");
  });

  it("flush() dispatches pending batch immediately and waits for resolution", async () => {
    let resolved = false;
    const batchFn = vi.fn(async (keys: string[]) => {
      await new Promise(r => setTimeout(r, 50));
      resolved = true;
      return new Map(keys.map(k => [k, 99]));
    });

    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 10, maxWaitMs: 5000 });
    const p1 = scheduler.schedule("fastKey");

    expect(scheduler.pendingCount).toBe(1);
    await scheduler.flush();

    expect(resolved).toBe(true);
    expect(await p1).toBe(99);
    expect(scheduler.pendingCount).toBe(0);
  });

  it("handles consecutive batches cleanly after first completes", async () => {
    const batchFn = vi.fn(async (keys: number[]) => new Map(keys.map(k => [k, k * 10])));
    const scheduler = new BatchScheduler({ batchFn, maxBatchSize: 2, maxWaitMs: 20 });

    const r1 = await Promise.all([scheduler.schedule(1), scheduler.schedule(2)]);
    expect(r1).toEqual([10, 20]);

    const r2 = await Promise.all([scheduler.schedule(3), scheduler.schedule(4)]);
    expect(r2).toEqual([30, 40]);

    expect(batchFn).toHaveBeenCalledTimes(2);
  });
});
