import { describe, it, expect } from "vitest";
import { RateLimiter } from "../src/limiter";

describe("RateLimiter (hidden)", () => {
  it("preserves fractional token balances with high precision", () => {
    let now = 1000;
    const limiter = new RateLimiter({
      capacity: 10,
      refillRatePerSec: 10, // 0.01 tokens / ms
      initialTokens: 0,
      nowFn: () => now,
    });

    now += 25; // 0.25 tokens
    expect(limiter.availableTokens).toBeCloseTo(0.25, 4);

    now += 50; // +0.50 -> 0.75 tokens
    expect(limiter.availableTokens).toBeCloseTo(0.75, 4);

    now += 25; // +0.25 -> 1.00 token
    expect(limiter.availableTokens).toBeCloseTo(1.0, 4);
    expect(limiter.tryAcquire(1)).toBe(true);
    expect(limiter.availableTokens).toBeCloseTo(0, 4);
  });

  it("handles clock skew backwards jumps safely", () => {
    let now = 10000;
    const limiter = new RateLimiter({
      capacity: 10,
      refillRatePerSec: 10,
      initialTokens: 5,
      nowFn: () => now,
    });

    // Clock steps backward by 2000ms (NTP readjustment)
    now = 8000;
    const res = limiter.acquire(3);
    expect(res.allowed).toBe(true);
    expect(res.remainingTokens).toBeCloseTo(2, 4);

    // Clock steps forward from backward time
    now = 8100; // +100ms -> +1 token -> 3 tokens
    expect(limiter.availableTokens).toBeCloseTo(3, 4);
  });

  it("enforces burst spike ratio limits with retryAfterMs: Infinity", () => {
    let now = 1000;
    const limiter = new RateLimiter({
      capacity: 100,
      refillRatePerSec: 50,
      maxBurstSpikeRatio: 0.5, // max 50 tokens in single request
      nowFn: () => now,
    });

    // Requesting 60 tokens even when full must be rejected
    const res = limiter.acquire(60);
    expect(res.allowed).toBe(false);
    expect(res.remainingTokens).toBe(100);
    expect(res.retryAfterMs).toBe(Infinity);

    // Requesting 50 tokens is allowed
    const resOk = limiter.acquire(50);
    expect(resOk.allowed).toBe(true);
    expect(resOk.remainingTokens).toBe(50);
  });

  it("throws on non-positive token acquisition", () => {
    const limiter = new RateLimiter({
      capacity: 10,
      refillRatePerSec: 5,
    });

    expect(() => limiter.acquire(0)).toThrow(/Tokens requested must be positive/);
    expect(() => limiter.acquire(-5)).toThrow(/Tokens requested must be positive/);
  });

  it("resets tokens and timestamp on reset()", () => {
    let now = 1000;
    const limiter = new RateLimiter({
      capacity: 20,
      refillRatePerSec: 10,
      nowFn: () => now,
    });

    limiter.acquire(20);
    expect(limiter.availableTokens).toBe(0);

    now += 10000;
    limiter.reset();
    expect(limiter.availableTokens).toBe(20);
  });
});
