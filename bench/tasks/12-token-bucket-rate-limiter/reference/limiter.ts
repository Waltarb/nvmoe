import type { RateLimiterOptions, AcquireResult } from "../src/types";

export class RateLimiter {
  private options: RateLimiterOptions;
  private tokens: number;
  private lastRefill: number;

  constructor(options: RateLimiterOptions) {
    this.options = options;
    const now = options.nowFn ? options.nowFn() : Date.now();
    this.tokens = options.initialTokens !== undefined ? options.initialTokens : options.capacity;
    this.lastRefill = now;
  }

  private getNow(): number {
    return this.options.nowFn ? this.options.nowFn() : Date.now();
  }

  private refill(): void {
    const now = this.getNow();
    if (now > this.lastRefill) {
      const elapsedMs = now - this.lastRefill;
      const tokensToAdd = elapsedMs * (this.options.refillRatePerSec / 1000);
      this.tokens = Math.min(this.options.capacity, this.tokens + tokensToAdd);
      this.lastRefill = now;
    } else {
      // Clock skew backwards jump: update lastRefill to now without penalizing tokens
      this.lastRefill = now;
    }
  }

  get availableTokens(): number {
    this.refill();
    return this.tokens;
  }

  acquire(tokens = 1): AcquireResult {
    if (tokens <= 0) {
      throw new Error("Tokens requested must be positive");
    }

    this.refill();

    if (this.options.maxBurstSpikeRatio !== undefined) {
      const maxAllowed = this.options.capacity * this.options.maxBurstSpikeRatio;
      if (tokens > maxAllowed) {
        return {
          allowed: false,
          remainingTokens: this.tokens,
          retryAfterMs: Infinity,
        };
      }
    }

    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return {
        allowed: true,
        remainingTokens: this.tokens,
        retryAfterMs: 0,
      };
    }

    const deficit = tokens - this.tokens;
    const ratePerMs = this.options.refillRatePerSec / 1000;
    const retryAfterMs = Math.ceil(deficit / ratePerMs);

    return {
      allowed: false,
      remainingTokens: this.tokens,
      retryAfterMs,
    };
  }

  tryAcquire(tokens = 1): boolean {
    return this.acquire(tokens).allowed;
  }

  reset(): void {
    this.tokens = this.options.capacity;
    this.lastRefill = this.getNow();
  }
}
