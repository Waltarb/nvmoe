import type { JSONSchema, ValidationResult } from "./types";

export function createValidator(
  schema: JSONSchema,
  defs?: Record<string, JSONSchema>
): (data: unknown) => ValidationResult {
  return (data: unknown) => {
    return { valid: true, errors: [] };
  };
}
