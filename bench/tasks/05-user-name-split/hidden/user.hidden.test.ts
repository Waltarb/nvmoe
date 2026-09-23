import { it, expect } from "vitest";
import { toUser } from "../src/api";
import { formatUser, initials } from "../src/format";

it("splits two names", () => expect(toUser({ id: "1", name: "Ada Lovelace" })).toEqual({ id: "1", firstName: "Ada", lastName: "Lovelace" }));
it("splits on first whitespace only", () => expect(toUser({ id: "2", name: "Jean Luc  Picard" })).toEqual({ id: "2", firstName: "Jean", lastName: "Luc  Picard" }));
it("single name", () => expect(toUser({ id: "3", name: "Cher" })).toEqual({ id: "3", firstName: "Cher", lastName: "" }));
it("trims", () => expect(toUser({ id: "4", name: "  Grace   Hopper " })).toEqual({ id: "4", firstName: "Grace", lastName: "Hopper" }));
it("formats single name", () => expect(formatUser({ id: "3", firstName: "Cher", lastName: "" })).toBe("Cher"));
it("formats full", () => expect(formatUser({ id: "1", firstName: "Ada", lastName: "Lovelace" })).toBe("Lovelace, Ada"));
it("initials", () => {
  expect(initials({ id: "1", firstName: "ada", lastName: "lovelace" })).toBe("AL");
  expect(initials({ id: "3", firstName: "Cher", lastName: "" })).toBe("C");
});
it("no fullName left", () => expect("fullName" in toUser({ id: "1", name: "Ada Lovelace" })).toBe(false));
