import { it, expect } from "vitest";
import { validateSignup } from "../src/validate";

it("valid with age", () =>
  expect(validateSignup({ email: "a@b.io", password: "abcdefg1", age: 30 })).toEqual({ ok: true, value: { email: "a@b.io", password: "abcdefg1", age: 30 } }));
it("non-object bodies", () => {
  for (const b of [null, [], "x", 42, undefined]) expect(validateSignup(b)).toEqual({ ok: false, errors: { _: "invalid_body" } });
});
it("missing fields", () => expect(validateSignup({})).toEqual({ ok: false, errors: { email: "required", password: "required" } }));
it("collects all errors", () =>
  expect(validateSignup({ email: "nope", password: "short", age: 12 })).toEqual({ ok: false, errors: { email: "invalid", password: "too_short", age: "invalid" } }));
it("needs digit", () => expect(validateSignup({ email: "a@b.io", password: "abcdefgh" })).toEqual({ ok: false, errors: { password: "needs_digit" } }));
it("email with space invalid", () => expect(validateSignup({ email: "a b@c.io", password: "abcdefg1" })).toEqual({ ok: false, errors: { email: "invalid" } }));
it("float age invalid", () => expect(validateSignup({ email: "a@b.io", password: "abcdefg1", age: 20.5 })).toEqual({ ok: false, errors: { age: "invalid" } }));
it("string age invalid", () => expect(validateSignup({ email: "a@b.io", password: "abcdefg1", age: "20" })).toEqual({ ok: false, errors: { age: "invalid" } }));
it("age bounds", () => {
  expect(validateSignup({ email: "a@b.io", password: "abcdefg1", age: 13 }).ok).toBe(true);
  expect(validateSignup({ email: "a@b.io", password: "abcdefg1", age: 121 }).ok).toBe(false);
});
it("non-string email", () => expect(validateSignup({ email: 5, password: "abcdefg1" })).toEqual({ ok: false, errors: { email: "required" } }));
