import { evaluateCandidate } from "./run-eval.ts";

const candidateResponse = `
src/scheduler.ts
<<<<<<< SEARCH
  get pendingCount(): number {
    return 0;
  }

  async schedule(key: K, signal?: AbortSignal): Promise<V> {
    throw new Error("Not implemented");
  }

  async flush(): Promise<void> {
    // Not implemented
  }
=======
  private pending = new Map<K, { resolve: (v: V) => void; reject: (err: any) => void; signal?: AbortSignal; onAbort?: () => void; aborted?: boolean }[]>();
  private timer: any = null;
  private inFlight: Promise<void> | null = null;

  get pendingCount(): number {
    let count = 0;
    for (const callers of this.pending.values()) {
      if (callers.some(c => !c.aborted)) count++;
    }
    return count;
  }

  async schedule(key: K, signal?: AbortSignal): Promise<V> {
    if (signal?.aborted) {
      throw signal.reason || new Error("Aborted");
    }

    return new Promise<V>((resolve, reject) => {
      const entry = { resolve, reject, signal, onAbort: undefined as (() => void) | undefined, aborted: false };

      if (signal) {
        entry.onAbort = () => {
          entry.aborted = true;
          signal.removeEventListener("abort", entry.onAbort!);
          reject(signal.reason || new Error("Aborted"));
          this.pruneKey(key);
        };
        signal.addEventListener("abort", entry.onAbort);
      }

      if (!this.pending.has(key)) {
        this.pending.set(key, []);
      }
      this.pending.get(key)!.push(entry);

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

  private pruneKey(key: K) {
    const list = this.pending.get(key);
    if (!list) return;
    const active = list.filter(c => !c.aborted);
    if (active.length === 0) {
      this.pending.delete(key);
      if (this.pendingCount === 0 && this.timer) {
        clearTimeout(this.timer);
        this.timer = null;
      }
    } else {
      this.pending.set(key, active);
    }
  }

  private dispatch(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const current = this.pending;
    this.pending = new Map();

    const keys: K[] = [];
    for (const [k, callers] of current.entries()) {
      if (callers.some(c => !c.aborted)) keys.push(k);
    }

    if (keys.length === 0) return Promise.resolve();

    const p = (async () => {
      try {
        const res = await this.options.batchFn(keys);
        for (const k of keys) {
          const callers = current.get(k) || [];
          if (!res.has(k)) {
            const err = new Error("Missing key in batch result: " + String(k));
            for (const c of callers) {
              if (c.signal && c.onAbort) c.signal.removeEventListener("abort", c.onAbort);
              if (!c.aborted) c.reject(err);
            }
          } else {
            const val = res.get(k)!;
            for (const c of callers) {
              if (c.signal && c.onAbort) c.signal.removeEventListener("abort", c.onAbort);
              if (!c.aborted) c.resolve(val);
            }
          }
        }
      } catch (err) {
        for (const callers of current.values()) {
          for (const c of callers) {
            if (c.signal && c.onAbort) c.signal.removeEventListener("abort", c.onAbort);
            if (!c.aborted) c.reject(err);
          }
        }
      }
    })();

    this.inFlight = p;
    return p;
  }

  async flush(): Promise<void> {
    if (this.pendingCount > 0) {
      await this.dispatch();
    }
    if (this.inFlight) {
      await this.inFlight;
    }
  }
>>>>>>> REPLACE
`;

await evaluateCandidate("06-async-batch-scheduler", candidateResponse, "Gemini-Flash-3.8(high)");
