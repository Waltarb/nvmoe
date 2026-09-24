# Feature: Async Batch Scheduler

Implement `BatchScheduler<K, V>` in `src/scheduler.ts` to batch individual async requests into periodic bulk loader calls.

## Requirements

### Constructor & Options
`new BatchScheduler<K, V>({ batchFn, maxBatchSize, maxWaitMs })`
- `batchFn: (keys: K[]) => Promise<Map<K, V>>`: Bulk fetch function returning a map from key to value.
- `maxBatchSize`: Maximum number of unique keys per batch.
- `maxWaitMs`: Maximum time in ms before dispatching pending keys.

### Methods
- `schedule(key: K, signal?: AbortSignal): Promise<V>`
  - Groups requests into batches.
  - If identical keys are requested within the same batch:
    - `batchFn` is called with that key only once (deduplication).
    - All callers for that key resolve with the same fetched value.
  - When unique keys count reaches `maxBatchSize`, dispatches immediately without waiting for `maxWaitMs`.
  - Otherwise dispatches after `maxWaitMs` from when the first key of the current batch was scheduled.
  - If `signal` is aborted (or already aborted upon call):
    - Rejects the caller's promise immediately with `signal.reason` (or `new Error("Aborted")` if reason is empty).
    - Cleans up abort event listeners to prevent memory leaks.
    - If all callers for a key abort before dispatch, that key is removed from the batch.
    - If all keys in a pending batch abort, cancel any pending timer; do not call `batchFn`.
  - If `batchFn` rejects, all non-aborted callers in that batch reject with the same error.
  - If `batchFn` succeeds but a key is missing from the returned Map, that key's callers reject with `new Error("Missing key in batch result: " + String(key))`.
- `flush(): Promise<void>`
  - Dispatches any pending batch immediately and returns a promise that resolves when the dispatched batch finishes processing. If no batch is pending, resolves immediately.
- `get pendingCount(): number`
  - Returns the number of unique keys currently waiting in the pending batch (excluding keys where all callers have aborted).
