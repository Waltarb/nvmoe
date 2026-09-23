import { it, expect } from "vitest";
import { slugify } from "../src/slugify";

it("basic", () => expect(slugify("Hello World")).toBe("hello-world"));
it("diacritics", () => expect(slugify("Crème Brûlée!")).toBe("creme-brulee"));
it("ampersand", () => expect(slugify("Tom & Jerry")).toBe("tom-and-jerry"));
it("collapses and trims", () => expect(slugify("  --a__b--  ")).toBe("a-b"));
it("numbers kept", () => expect(slugify("Top 10 Tips (2026)")).toBe("top-10-tips-2026"));
it("cuts at word boundary", () => expect(slugify("the quick brown fox jumps", { maxLength: 12 })).toBe("the-quick"));
it("hard cut without dash", () => expect(slugify("abcdefghijkl", { maxLength: 5 })).toBe("abcde"));
it("default max 60", () => expect(slugify("word ".repeat(30)).length).toBeLessThanOrEqual(60));
it("no trailing dash after cut", () => expect(slugify("aaaa bbbb cccc", { maxLength: 10 })).toBe("aaaa-bbbb"));
it("empty", () => expect(slugify("")).toBe(""));
it("symbols only", () => expect(slugify("!!!???")).toBe(""));
