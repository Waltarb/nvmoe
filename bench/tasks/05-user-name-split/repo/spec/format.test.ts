import { it, expect } from "vitest";
import { formatUser } from "../src/format";

it("formats last, first", () => {
  expect(formatUser({ id: "1", firstName: "Ada", lastName: "Lovelace" })).toBe("Lovelace, Ada");
});
