import { describe, it, expect } from "vitest";
import { paginate } from "../src/paginate";

const ten = Array.from({ length: 10 }, (_, i) => i + 1);

describe("paginate", () => {
  it("returns the first page for page 1", () => {
    const p = paginate(ten, 1, 3);
    expect(p.items).toEqual([1, 2, 3]);
    expect(p.hasPrev).toBe(false);
    expect(p.hasNext).toBe(true);
  });
});
