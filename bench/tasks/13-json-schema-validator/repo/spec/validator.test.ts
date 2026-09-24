import { describe, it, expect } from "vitest";
import { createValidator } from "../src/validator";
import type { JSONSchema } from "../src/types";

describe("JSON Schema Validator (visible)", () => {
  it("validates basic object schema", () => {
    const schema: JSONSchema = {
      type: "object",
      required: ["name", "age"],
      properties: {
        name: { type: "string", minLength: 2 },
        age: { type: "number", minimum: 0 },
      },
    };

    const validator = createValidator(schema);

    expect(validator({ name: "Alice", age: 30 })).toEqual({ valid: true, errors: [] });
    const invalid = validator({ name: "A", age: -1 });
    expect(invalid.valid).toBe(false);
    expect(invalid.errors.length).toBeGreaterThan(0);
  });
});
