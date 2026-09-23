import { describe, it, expect } from "vitest";
import { paginate } from "../src/paginate";

const ten = Array.from({ length: 10 }, (_, i) => i + 1);

describe("paginate (hidden)", () => {
  it("first page", () => expect(paginate(ten, 1, 3)).toEqual({ items: [1, 2, 3], page: 1, totalPages: 4, hasNext: true, hasPrev: false }));
  it("last partial page", () => expect(paginate(ten, 4, 3)).toEqual({ items: [10], page: 4, totalPages: 4, hasNext: false, hasPrev: true }));
  it("middle page", () => expect(paginate(ten, 2, 3).items).toEqual([4, 5, 6]));
  it("clamps below 1", () => expect(paginate(ten, 0, 3).page).toBe(1));
  it("clamps above total", () => {
    const p = paginate(ten, 99, 3);
    expect(p.page).toBe(4);
    expect(p.items).toEqual([10]);
  });
  it("empty list", () => expect(paginate([], 1, 5)).toEqual({ items: [], page: 1, totalPages: 1, hasNext: false, hasPrev: false }));
  it("exact multiple", () => expect(paginate([1, 2, 3, 4], 2, 2)).toEqual({ items: [3, 4], page: 2, totalPages: 2, hasNext: false, hasPrev: true }));
});
