export interface RateLimiterOptions {
  capacity: number;
  refillRatePerSec: number;
  initialTokens?: number;
  maxBurstSpikeRatio?: number;
  nowFn?: () => number;
}

export interface AcquireResult {
  allowed: boolean;
  remainingTokens: number;
  retryAfterMs: number;
}
