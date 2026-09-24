import type { BatchSchedulerOptions } from "../src/types";

interface PendingCall<V> {
  resolve: (value: V) => void;
  reject: (err: any) => void;
  signal?: AbortSignal;
  abortListener?: () => void;
  aborted?: boolean;
}

export class BatchScheduler<K, V> {
  private options: BatchSchedulerOptions<K, V>;
  private pending = new Map<K, PendingCall<V>[]>();
  private timer: any = null;
  private currentBatchPromise: Promise<void> | null = null;

  constructor(options: BatchSchedulerOptions<K, V>) {
    this.options = options;
  }

  get pendingCount(): number {
    let count = 0;
    for (const calls of this.pending.values()) {
      if (calls.some(c => !c.aborted)) {
        count++;
      }
    }
    return count;
  }

  async schedule(key: K, signal?: AbortSignal): Promise<V> {
    if (signal?.aborted) {
      throw signal.reason || new Error("Aborted");
    }

    return new Promise<V>((resolve, reject) => {
      const call: PendingCall<V> = { resolve, reject, signal };

      if (signal) {
        const onAbort = () => {
          call.aborted = true;
          signal.removeEventListener("abort", onAbort);
          reject(signal.reason || new Error("Aborted"));
          this.cleanupKeyIfEmpty(key);
        };
        call.abortListener = onAbort;
        signal.addEventListener("abort", onAbort);
      }

      if (!this.pending.has(key)) {
        this.pending.set(key, []);
      }
      this.pending.get(key)!.push(call);

      if (this.pendingCount >= this.options.maxBatchSize) {
        this.dispatch();
      } else if (!this.timer) {
        this.timer = setTimeout(() => {
          this.timer = null;
          this.dispatch();
        }, this.options.maxWaitMs);
      }
    });
  }

  private cleanupKeyIfEmpty(key: K): void {
    const list = this.pending.get(key);
    if (!list) return;
    const remaining = list.filter(c => !c.aborted);
    if (remaining.length === 0) {
      this.pending.delete(key);
      if (this.pendingCount === 0 && this.timer) {
        clearTimeout(this.timer);
        this.timer = null;
      }
    } else {
      this.pending.set(key, remaining);
    }
  }

  private dispatch(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    const currentMap = this.pending;
    this.pending = new Map();

    const keysToFetch: K[] = [];
    for (const [k, calls] of currentMap.entries()) {
      if (calls.some(c => !c.aborted)) {
        keysToFetch.push(k);
      }
    }

    if (keysToFetch.length === 0) {
      return Promise.resolve();
    }

    const batchPromise = (async () => {
      try {
        const resultMap = await this.options.batchFn(keysToFetch);
        for (const k of keysToFetch) {
          const calls = currentMap.get(k) || [];
          if (!resultMap.has(k)) {
            const err = new Error("Missing key in batch result: " + String(k));
            for (const call of calls) {
              if (call.signal && call.abortListener) {
                call.signal.removeEventListener("abort", call.abortListener);
              }
              if (!call.aborted) {
                call.reject(err);
              }
            }
          } else {
            const val = resultMap.get(k)!;
            for (const call of calls) {
              if (call.signal && call.abortListener) {
                call.signal.removeEventListener("abort", call.abortListener);
              }
              if (!call.aborted) {
                call.resolve(val);
              }
            }
          }
        }
      } catch (err) {
        for (const calls of currentMap.values()) {
          for (const call of calls) {
            if (call.signal && call.abortListener) {
              call.signal.removeEventListener("abort", call.abortListener);
            }
            if (!call.aborted) {
              call.reject(err);
            }
          }
        }
      }
    })();

    this.currentBatchPromise = batchPromise;
    return batchPromise;
  }

  async flush(): Promise<void> {
    if (this.pendingCount > 0) {
      await this.dispatch();
    }
    if (this.currentBatchPromise) {
      await this.currentBatchPromise;
    }
  }
}
