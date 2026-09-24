import type { BatchSchedulerOptions } from "./types";

export class BatchScheduler<K, V> {
  private options: BatchSchedulerOptions<K, V>;

  constructor(options: BatchSchedulerOptions<K, V>) {
    this.options = options;
  }

  get pendingCount(): number {
    return 0;
  }

  async schedule(key: K, signal?: AbortSignal): Promise<V> {
    throw new Error("Not implemented");
  }

  async flush(): Promise<void> {
    // Not implemented
  }
}
