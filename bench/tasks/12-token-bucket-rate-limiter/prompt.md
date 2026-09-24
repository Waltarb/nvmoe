# Feature: Fractional Token Bucket Rate Limiter

Implement `RateLimiter` in `src/limiter.ts` providing precision fractional token replenishment, burst spike mitigation, and clock-skew safety.

## Configuration & Types
```ts
export interface RateLimiterOptions {
  capacity: number;             // Maximum tokens the bucket can hold
  refillRatePerSec: number;     // Tokens added per second
  initialTokens?: number;       // Defaults to capacity
  maxBurstSpikeRatio?: number;  // Max single-request ratio (0 < ratio <= 1.0)
  nowFn?: () => number;         // Millisecond clock (defaults to Date.now)
}

export interface AcquireResult {
  allowed: boolean;
  remainingTokens: number;
  retryAfterMs: number;
}
```

## Methods
- `acquire(tokens = 1): AcquireResult`
  - Refills bucket based on elapsed time: `(now - lastRefill) * (refillRatePerSec / 1000)`.
  - Tokens cannot exceed `capacity`.
  - If `tokens <= 0`, throws `new Error("Tokens requested must be positive")`.
  - **Burst Spike Cap**: If `maxBurstSpikeRatio` is configured and `tokens > capacity * maxBurstSpikeRatio`, reject immediately with `{ allowed: false, remainingTokens: current, retryAfterMs: Infinity }`.
  - **Clock Skew Protection**: If current clock `now < lastRefill` (e.g. backward NTP step), clamp elapsed time to 0; do NOT subtract tokens or crash.
  - If sufficient tokens are available: deduct `tokens`, update `lastRefill = now`, return `{ allowed: true, remainingTokens, retryAfterMs: 0 }`.
  - If insufficient: do not deduct tokens, update `lastRefill = now`, return `{ allowed: false, remainingTokens, retryAfterMs: Math.ceil((tokens - remainingTokens) / (refillRatePerSec / 1000)) }`.
- `tryAcquire(tokens = 1): boolean`
  - Returns `true` if `acquire(tokens).allowed` is true, otherwise `false`.
- `get availableTokens(): number`
  - Performs refill calculation up to `now` and returns current available tokens.
- `reset(): void`
  - Resets tokens to `capacity` and sets `lastRefill = now`.
