import { SerializationConflictError, type VersionRecord } from "../src/types";

export class Transaction {
  readonly id: number;
  readonly readSnapshotVersion: number;
  private store: MvccStore;
  private writes = new Map<string, string | null>();
  private closed = false;

  constructor(id: number, readSnapshotVersion: number, store: MvccStore) {
    this.id = id;
    this.readSnapshotVersion = readSnapshotVersion;
    this.store = store;
  }

  private checkOpen(): void {
    if (this.closed) {
      throw new Error("Transaction is closed");
    }
  }

  get(key: string): string | undefined {
    this.checkOpen();
    if (this.writes.has(key)) {
      const val = this.writes.get(key)!;
      return val === null ? undefined : val;
    }
    return this.store.readVersion(key, this.readSnapshotVersion);
  }

  set(key: string, value: string): void {
    this.checkOpen();
    this.writes.set(key, value);
  }

  delete(key: string): void {
    this.checkOpen();
    this.writes.set(key, null);
  }

  commit(): void {
    this.checkOpen();
    try {
      this.store.commitTransaction(this, this.writes);
    } finally {
      this.closed = true;
    }
  }

  rollback(): void {
    this.checkOpen();
    this.closed = true;
    this.store.rollbackTransaction(this);
  }
}

export class MvccStore {
  private _currentVersion = 0;
  private nextTxId = 1;
  private activeTx = new Map<number, Transaction>();
  private data = new Map<string, VersionRecord[]>();

  get currentVersion(): number {
    return this._currentVersion;
  }

  get activeTxCount(): number {
    return this.activeTx.size;
  }

  beginTransaction(): Transaction {
    const txId = this.nextTxId++;
    const tx = new Transaction(txId, this._currentVersion, this);
    this.activeTx.set(txId, tx);
    return tx;
  }

  readVersion(key: string, maxVersion: number): string | undefined {
    const versions = this.data.get(key);
    if (!versions || versions.length === 0) return undefined;

    for (let i = versions.length - 1; i >= 0; i--) {
      const rec = versions[i];
      if (rec.version <= maxVersion) {
        return rec.value === null ? undefined : rec.value;
      }
    }
    return undefined;
  }

  getCommitted(key: string): string | undefined {
    return this.readVersion(key, this._currentVersion);
  }

  commitTransaction(tx: Transaction, writes: Map<string, string | null>): void {
    this.activeTx.delete(tx.id);

    // Conflict detection: First-Committer-Wins
    for (const key of writes.keys()) {
      const versions = this.data.get(key);
      if (versions && versions.length > 0) {
        const latest = versions[versions.length - 1];
        if (latest.version > tx.readSnapshotVersion) {
          throw new SerializationConflictError(key);
        }
      }
    }

    if (writes.size === 0) return;

    this._currentVersion++;
    const commitVer = this._currentVersion;

    for (const [key, val] of writes.entries()) {
      let versions = this.data.get(key);
      if (!versions) {
        versions = [];
        this.data.set(key, versions);
      }
      versions.push({ version: commitVer, value: val });
    }
  }

  rollbackTransaction(tx: Transaction): void {
    this.activeTx.delete(tx.id);
  }

  vacuum(): number {
    let minSnapshot = this._currentVersion;
    if (this.activeTx.size > 0) {
      for (const tx of this.activeTx.values()) {
        if (tx.readSnapshotVersion < minSnapshot) {
          minSnapshot = tx.readSnapshotVersion;
        }
      }
    }

    let prunedCount = 0;

    for (const [key, versions] of Array.from(this.data.entries())) {
      let lastLeIdx = -1;
      for (let i = 0; i < versions.length; i++) {
        if (versions[i].version <= minSnapshot) {
          lastLeIdx = i;
        } else {
          break;
        }
      }

      if (lastLeIdx > 0) {
        prunedCount += lastLeIdx;
        versions.splice(0, lastLeIdx);
      }

      // If the remaining newest record is a tombstone and there are no other versions, remove completely
      if (versions.length === 1 && versions[0].value === null) {
        this.data.delete(key);
        prunedCount += 1;
      }
    }

    return prunedCount;
  }
}
