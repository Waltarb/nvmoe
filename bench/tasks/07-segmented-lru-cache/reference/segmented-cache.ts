import type { SegmentedCacheOptions, CacheSizes } from "../src/types";

interface CacheEntry<V> {
  value: V;
  expiresAt?: number;
}

export class SegmentedCache<K, V> {
  private options: SegmentedCacheOptions;
  private probation = new Map<K, CacheEntry<V>>();
  private protected = new Map<K, CacheEntry<V>>();
  private ghost = new Map<K, boolean>();

  constructor(options: SegmentedCacheOptions) {
    this.options = options;
  }

  private isExpired(entry: CacheEntry<V>): boolean {
    return entry.expiresAt !== undefined && Date.now() >= entry.expiresAt;
  }

  private makeMru<T>(map: Map<K, T>, key: K, val: T): void {
    map.delete(key);
    map.set(key, val);
  }

  private insertGhost(key: K): void {
    this.makeMru(this.ghost, key, true);
    if (this.ghost.size > this.options.ghostCapacity) {
      const oldestGhost = this.ghost.keys().next().value;
      if (oldestGhost !== undefined) {
        this.ghost.delete(oldestGhost);
      }
    }
  }

  private insertProbation(key: K, entry: CacheEntry<V>): void {
    this.makeMru(this.probation, key, entry);
    if (this.probation.size > this.options.probationCapacity) {
      const oldestProbationKey = this.probation.keys().next().value;
      if (oldestProbationKey !== undefined) {
        this.probation.delete(oldestProbationKey);
        this.insertGhost(oldestProbationKey);
      }
    }
  }

  private insertProtected(key: K, entry: CacheEntry<V>): void {
    this.makeMru(this.protected, key, entry);
    if (this.protected.size > this.options.protectedCapacity) {
      const oldestProtectedKey = this.protected.keys().next().value;
      if (oldestProtectedKey !== undefined) {
        const demotedEntry = this.protected.get(oldestProtectedKey)!;
        this.protected.delete(oldestProtectedKey);
        this.insertProbation(oldestProtectedKey, demotedEntry);
      }
    }
  }

  get(key: K): V | undefined {
    const pEntry = this.probation.get(key);
    if (pEntry) {
      if (this.isExpired(pEntry)) {
        this.probation.delete(key);
        return undefined;
      }
      this.probation.delete(key);
      this.insertProtected(key, pEntry);
      return pEntry.value;
    }

    const protEntry = this.protected.get(key);
    if (protEntry) {
      if (this.isExpired(protEntry)) {
        this.protected.delete(key);
        return undefined;
      }
      this.makeMru(this.protected, key, protEntry);
      return protEntry.value;
    }

    return undefined;
  }

  set(key: K, value: V, ttlMs?: number): void {
    const effectiveTtl = ttlMs ?? this.options.defaultTtlMs;
    const expiresAt = effectiveTtl !== undefined ? Date.now() + effectiveTtl : undefined;
    const entry: CacheEntry<V> = { value, expiresAt };

    if (this.protected.has(key)) {
      this.makeMru(this.protected, key, entry);
      return;
    }

    if (this.probation.has(key)) {
      this.probation.delete(key);
      this.insertProtected(key, entry);
      return;
    }

    if (this.ghost.has(key)) {
      this.ghost.delete(key);
      this.insertProtected(key, entry);
      return;
    }

    this.insertProbation(key, entry);
  }

  has(key: K): boolean {
    const pEntry = this.probation.get(key);
    if (pEntry) {
      if (this.isExpired(pEntry)) {
        this.probation.delete(key);
        return false;
      }
      return true;
    }

    const protEntry = this.protected.get(key);
    if (protEntry) {
      if (this.isExpired(protEntry)) {
        this.protected.delete(key);
        return false;
      }
      return true;
    }

    return false;
  }

  delete(key: K): boolean {
    let found = false;
    if (this.probation.delete(key)) found = true;
    if (this.protected.delete(key)) found = true;
    if (this.ghost.delete(key)) found = true;
    return found;
  }

  size(): CacheSizes {
    return {
      probation: this.probation.size,
      protected: this.protected.size,
      ghost: this.ghost.size,
    };
  }

  clear(): void {
    this.probation.clear();
    this.protected.clear();
    this.ghost.clear();
  }
}
