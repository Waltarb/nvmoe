import { describe, it, expect } from "vitest";
import { createValidator } from "../src/validator";
import type { JSONSchema } from "../src/types";

describe("JSON Schema Validator (hidden)", () => {
  it("distinguishes null from object", () => {
    const objSchema: JSONSchema = { type: "object" };
    const nullSchema: JSONSchema = { type: "null" };

    const validateObj = createValidator(objSchema);
    const validateNull = createValidator(nullSchema);

    expect(validateObj(null).valid).toBe(false);
    expect(validateNull(null).valid).toBe(true);
    expect(validateNull({}).valid).toBe(false);
  });

  it("handles additionalProperties: false", () => {
    const schema: JSONSchema = {
      type: "object",
      properties: { a: { type: "number" } },
      additionalProperties: false,
    };
    const validate = createValidator(schema);

    expect(validate({ a: 1 }).valid).toBe(true);
    const res = validate({ a: 1, extra: "disallowed" });
    expect(res.valid).toBe(false);
    expect(res.errors[0].path).toContain("extra");
  });

  it("validates recursive cyclic $ref schemas (binary tree)", () => {
    const schema: JSONSchema = {
      $ref: "#/$defs/treeNode",
      $defs: {
        treeNode: {
          type: "object",
          required: ["value"],
          properties: {
            value: { type: "number" },
            left: { $ref: "#/$defs/treeNode" },
            right: { $ref: "#/$defs/treeNode" },
          },
        },
      },
    };

    const validate = createValidator(schema);

    const validTree = {
      value: 10,
      left: {
        value: 5,
        left: { value: 2 },
      },
      right: { value: 15 },
    };
    expect(validate(validTree).valid).toBe(true);

    const invalidTree = {
      value: 10,
      left: {
        value: "not-a-number",
      },
    };
    const res = validate(invalidTree);
    expect(res.valid).toBe(false);
    expect(res.errors[0].path).toBe("#/left/value");
  });

  it("evaluates oneOf correctly (exact 1 match rule)", () => {
    const schema: JSONSchema = {
      oneOf: [
        { type: "number", minimum: 0, maximum: 10 },
        { type: "number", minimum: 5, maximum: 15 },
      ],
    };
    const validate = createValidator(schema);

    // 2 is in [0, 10] only -> matches 1 -> valid
    expect(validate(2).valid).toBe(true);
    // 12 is in [5, 15] only -> matches 1 -> valid
    expect(validate(12).valid).toBe(true);
    // 7 is in BOTH [0, 10] and [5, 15] -> matches 2 -> INVALID
    expect(validate(7).valid).toBe(false);
    // 20 matches NEITHER -> matches 0 -> INVALID
    expect(validate(20).valid).toBe(false);
  });

  it("validates regex pattern and minItems array constraints", () => {
    const schema: JSONSchema = {
      type: "array",
      minItems: 2,
      items: {
        type: "string",
        pattern: "^[A-Z]{3}-\\d{3}$",
      },
    };
    const validate = createValidator(schema);

    expect(validate(["ABC-123", "XYZ-999"]).valid).toBe(true);
    expect(validate(["ABC-123"]).valid).toBe(false); // minItems fail
    expect(validate(["ABC-123", "invalid-code"]).valid).toBe(false); // pattern fail
  });
});
