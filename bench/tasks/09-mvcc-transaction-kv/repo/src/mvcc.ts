import { SerializationConflictError, type VersionRecord } from "./types";

export class Transaction {
  readonly id: number;
  readonly readSnapshotVersion: number;

  constructor(id: number, readSnapshotVersion: number) {
    this.id = id;
    this.readSnapshotVersion = readSnapshotVersion;
  }

  get(key: string): string | undefined {
    throw new Error("Not implemented");
  }

  set(key: string, value: string): void {
    throw new Error("Not implemented");
  }

  delete(key: string): void {
    throw new Error("Not implemented");
  }

  commit(): void {
    throw new Error("Not implemented");
  }

  rollback(): void {
    throw new Error("Not implemented");
  }
}

export class MvccStore {
  get currentVersion(): number {
    return 0;
  }

  get activeTxCount(): number {
    return 0;
  }

  beginTransaction(): Transaction {
    throw new Error("Not implemented");
  }

  getCommitted(key: string): string | undefined {
    return undefined;
  }

  vacuum(): number {
    return 0;
  }
}
