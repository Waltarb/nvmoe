export type BatchFn<K, V> = (keys: K[]) => Promise<Map<K, V>>;

export interface BatchSchedulerOptions<K, V> {
  batchFn: BatchFn<K, V>;
  maxBatchSize: number;
  maxWaitMs: number;
}
