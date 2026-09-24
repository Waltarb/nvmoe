# Feature: Segmented LRU (2Q / SLRU) Cache with Ghost Buffer and TTL

Implement `SegmentedCache<K, V>` in `src/segmented-cache.ts` providing a two-segment LRU cache with a non-resident ghost buffer (2Q variant) and time-to-live expiration.

## Cache Segments
The cache manages entries across three pools:
1. `probation` (resident, capacity `probationCapacity`): Holds newly inserted items on their first observation.
2. `protected` (resident, capacity `protectedCapacity`): Holds frequently accessed items promoted from probation or ghost.
3. `ghost` (non-resident, capacity `ghostCapacity`): Holds recently evicted keys from probation (without values) to recognize re-accessed items.

## Eviction & Promotion Rules

### 1. `get(key: K): V | undefined`
- If the item is expired (current timestamp >= expiry), delete it immediately and return `undefined`. (Expired items do NOT enter ghost).
- If the item is in `probation`:
  - Promote it to `protected` (as MRU).
  - If `protected` exceeds `protectedCapacity`, demote the LRU item of `protected` to `probation` (as MRU in probation).
  - If `probation` exceeds `probationCapacity`, evict its LRU item and insert its key into `ghost` (as MRU in ghost).
  - If `ghost` exceeds `ghostCapacity`, evict its oldest key.
  - Return the item's value.
- If the item is in `protected`:
  - Mark it as MRU in `protected`.
  - Return the value.
- If not present in probation or protected, return `undefined`.

### 2. `set(key: K, value: V, ttlMs?: number): void`
- Calculates expiry = `ttlMs ?? defaultTtlMs ? Date.now() + (ttlMs ?? defaultTtlMs) : undefined`.
- If the key is already in `protected`:
  - Update value and expiry, mark as MRU in `protected`.
- If the key is already in `probation`:
  - Update value and expiry, promote to `protected` following the promotion cascade above.
- If the key is in `ghost`:
  - Remove from `ghost`.
  - Insert directly into `protected` (as MRU in protected), following the demotion cascade if protected overflows.
- If the key is in none of the above:
  - Insert into `probation` (as MRU).
  - If `probation` exceeds `probationCapacity`, evict its LRU item and add its key to `ghost` (as MRU).
  - If `ghost` exceeds `ghostCapacity`, evict its oldest key.

### 3. Additional Methods
- `has(key: K): boolean`: Returns true if key is currently resident and unexpired in `probation` or `protected`. If expired, cleans it up and returns false.
- `delete(key: K): boolean`: Removes key from `probation`, `protected`, or `ghost`. Returns true if it was found.
- `size(): { probation: number; protected: number; ghost: number }`: Returns the count of active items in each segment.
- `clear(): void`: Clears all three segments.
