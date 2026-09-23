import { it, expect } from "vitest";
import { slugify } from "../src/slugify";

it("slugifies a simple title", () => {
  expect(slugify("Hello World")).toBe("hello-world");
});
