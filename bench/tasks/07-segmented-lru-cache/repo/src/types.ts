export interface SegmentedCacheOptions {
  probationCapacity: number;
  protectedCapacity: number;
  ghostCapacity: number;
  defaultTtlMs?: number;
}

export interface CacheSizes {
  probation: number;
  protected: number;
  ghost: number;
}
