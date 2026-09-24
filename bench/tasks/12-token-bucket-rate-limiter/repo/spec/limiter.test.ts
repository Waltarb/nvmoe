import { describe, it, expect } from "vitest";
import { RateLimiter } from "../src/limiter";

describe("RateLimiter (visible)", () => {
  it("acquires tokens up to capacity", () => {
    let now = 1000;
    const limiter = new RateLimiter({
      capacity: 10,
      refillRatePerSec: 10,
      nowFn: () => now,
    });

    const res1 = limiter.acquire(6);
    expect(res1.allowed).toBe(true);
    expect(res1.remainingTokens).toBe(4);
    expect(res1.retryAfterMs).toBe(0);

    const res2 = limiter.acquire(5); // deficit of 1
    expect(res2.allowed).toBe(false);
    expect(res2.remainingTokens).toBe(4);
    expect(res2.retryAfterMs).toBe(100); // 1 token / (10/1000) = 100ms
  });

  it("refills tokens over time", () => {
    let now = 1000;
    const limiter = new RateLimiter({
      capacity: 10,
      refillRatePerSec: 10,
      nowFn: () => now,
    });

    limiter.acquire(10);
    expect(limiter.availableTokens).toBe(0);

    // Advance 500ms -> should refill 5 tokens
    now += 500;
    expect(limiter.availableTokens).toBe(5);
    expect(limiter.tryAcquire(5)).toBe(true);
  });
});
