import { it, expect } from "vitest";
import { validateSignup } from "../src/validate";

it("accepts a valid body and normalises the email", () => {
  expect(validateSignup({ email: "  Ada@Example.COM ", password: "hunter22" })).toEqual({
    ok: true,
    value: { email: "ada@example.com", password: "hunter22" },
  });
});
