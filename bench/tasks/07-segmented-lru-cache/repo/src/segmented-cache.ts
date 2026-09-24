import type { SegmentedCacheOptions, CacheSizes } from "./types";

export class SegmentedCache<K, V> {
  private options: SegmentedCacheOptions;

  constructor(options: SegmentedCacheOptions) {
    this.options = options;
  }

  get(key: K): V | undefined {
    return undefined;
  }

  set(key: K, value: V, ttlMs?: number): void {
    // Not implemented
  }

  has(key: K): boolean {
    return false;
  }

  delete(key: K): boolean {
    return false;
  }

  size(): CacheSizes {
    return { probation: 0, protected: 0, ghost: 0 };
  }

  clear(): void {
    // Not implemented
  }
}
