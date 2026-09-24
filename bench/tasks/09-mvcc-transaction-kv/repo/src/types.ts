export class SerializationConflictError extends Error {
  readonly conflictKey: string;
  constructor(key: string) {
    super(`Serialization conflict on key: ${key}`);
    this.name = "SerializationConflictError";
    this.conflictKey = key;
  }
}

export interface VersionRecord {
  version: number;
  value: string | null;
}
