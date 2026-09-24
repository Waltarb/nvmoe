# Feature: In-Memory MVCC Key-Value Store with Snapshot Isolation

Implement `MvccStore` and `Transaction` in `src/mvcc.ts` supporting Snapshot Isolation and First-Committer-Wins conflict resolution.

## Architectural Model
Every committed write generates a new versioned entry `{ version: number, value: string | null }` where `null` represents a deleted tombstone.

### 1. `MvccStore`
- `beginTransaction(): Transaction`: Creates an active transaction with `readSnapshotVersion = currentVersion`.
- `getCommitted(key: string): string | undefined`: Returns the latest globally committed value for `key`, or `undefined` if deleted or unset.
- `vacuum(): number`: Prunes obsolete versions.
  - Determine `minSnapshot = min(activeTx.readSnapshotVersion)`. If there are no active transactions, `minSnapshot = currentVersion`.
  - For each key's version chain, retain only the newest version with `version <= minSnapshot` and all newer versions (`version > minSnapshot`). Any older versions are discarded. If the retained version is a tombstone and has no newer versions, it can be removed completely.
  - Returns the total count of version entries purged.
- `get currentVersion(): number`: monotonically increasing version number (starts at 0).
- `get activeTxCount(): number`: count of currently open transactions.

### 2. `Transaction`
- `readonly id: number`
- `readonly readSnapshotVersion: number`
- `get(key: string): string | undefined`:
  - Returns value from local uncommitted write set if modified by this transaction (if marked deleted, returns `undefined`).
  - Otherwise, searches committed versions and returns the value of the newest version with `version <= readSnapshotVersion`.
  - Returns `undefined` if no such version exists or if it is a tombstone (`null`).
- `set(key: string, value: string): void`: Buffers write locally.
- `delete(key: string): void`: Buffers deletion tombstone locally.
- `commit(): void`:
  - Conflict Check (First-Committer-Wins): For every key in the transaction's write set (both sets and deletes), inspect if the store has any committed version with `version > readSnapshotVersion`. If any exists, abort and throw `new SerializationConflictError(key)`.
  - If no conflicts: increment `currentVersion` by 1. For each key in write set, append a new version `{ version: currentVersion, value }`. Mark transaction as committed and remove from active transactions.
- `rollback(): void`: Discards uncommitted write set and marks transaction closed.
- Any operation (`get`, `set`, `delete`, `commit`, `rollback`) on a transaction that is already committed or rolled back must throw `new Error("Transaction is closed")`.
