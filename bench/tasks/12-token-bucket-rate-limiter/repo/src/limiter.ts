import type { RateLimiterOptions, AcquireResult } from "./types";

export class RateLimiter {
  private options: RateLimiterOptions;

  constructor(options: RateLimiterOptions) {
    this.options = options;
  }

  get availableTokens(): number {
    return 0;
  }

  acquire(tokens = 1): AcquireResult {
    throw new Error("Not implemented");
  }

  tryAcquire(tokens = 1): boolean {
    return false;
  }

  reset(): void {
    // Not implemented
  }
}
