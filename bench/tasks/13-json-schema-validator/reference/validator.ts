import type { JSONSchema, ValidationError, ValidationResult, SchemaType } from "../src/types";

export function createValidator(
  rootSchema: JSONSchema,
  externalDefs?: Record<string, JSONSchema>
): (data: unknown) => ValidationResult {
  const allDefs: Record<string, JSONSchema> = {
    ...externalDefs,
    ...rootSchema.$defs,
    ...rootSchema.definitions,
  };

  function resolveRef(ref: string): JSONSchema {
    if (ref.startsWith("#/$defs/")) {
      const name = ref.slice("#/$defs/".length);
      const target = allDefs[name];
      if (!target) throw new Error(`Cannot resolve ref: ${ref}`);
      return target;
    }
    if (ref.startsWith("#/definitions/")) {
      const name = ref.slice("#/definitions/".length);
      const target = allDefs[name];
      if (!target) throw new Error(`Cannot resolve ref: ${ref}`);
      return target;
    }
    throw new Error(`Unsupported ref format: ${ref}`);
  }

  function validateNode(data: unknown, schema: JSONSchema, path: string, errors: ValidationError[]): void {
    if (schema.$ref) {
      const target = resolveRef(schema.$ref);
      validateNode(data, target, path, errors);
      return;
    }

    if (schema.oneOf) {
      let matchCount = 0;
      for (const sub of schema.oneOf) {
        const subErrors: ValidationError[] = [];
        validateNode(data, sub, path, subErrors);
        if (subErrors.length === 0) {
          matchCount++;
        }
      }
      if (matchCount !== 1) {
        errors.push({ path, message: `Expected exactly 1 match in oneOf, matched ${matchCount}` });
      }
      return;
    }

    if (schema.type) {
      const types = Array.isArray(schema.type) ? schema.type : [schema.type];
      let matchesType = false;

      for (const t of types) {
        if (t === "null" && data === null) matchesType = true;
        else if (t === "array" && Array.isArray(data)) matchesType = true;
        else if (t === "object" && typeof data === "object" && data !== null && !Array.isArray(data)) matchesType = true;
        else if (t === "string" && typeof data === "string") matchesType = true;
        else if (t === "number" && typeof data === "number" && !isNaN(data)) matchesType = true;
        else if (t === "boolean" && typeof data === "boolean") matchesType = true;
      }

      if (!matchesType) {
        errors.push({ path, message: `Expected type ${JSON.stringify(schema.type)}, got ${typeof data === "object" ? (data === null ? "null" : Array.isArray(data) ? "array" : "object") : typeof data}` });
        return; // No point in further property checks if type mismatch
      }
    }

    if (typeof data === "string") {
      if (schema.minLength !== undefined && data.length < schema.minLength) {
        errors.push({ path, message: `String length ${data.length} < minLength ${schema.minLength}` });
      }
      if (schema.maxLength !== undefined && data.length > schema.maxLength) {
        errors.push({ path, message: `String length ${data.length} > maxLength ${schema.maxLength}` });
      }
      if (schema.pattern !== undefined) {
        const re = new RegExp(schema.pattern);
        if (!re.test(data)) {
          errors.push({ path, message: `String does not match pattern ${schema.pattern}` });
        }
      }
    }

    if (typeof data === "number") {
      if (schema.minimum !== undefined && data < schema.minimum) {
        errors.push({ path, message: `Value ${data} < minimum ${schema.minimum}` });
      }
      if (schema.maximum !== undefined && data > schema.maximum) {
        errors.push({ path, message: `Value ${data} > maximum ${schema.maximum}` });
      }
    }

    if (Array.isArray(data)) {
      if (schema.minItems !== undefined && data.length < schema.minItems) {
        errors.push({ path, message: `Array length ${data.length} < minItems ${schema.minItems}` });
      }
      if (schema.items) {
        for (let i = 0; i < data.length; i++) {
          validateNode(data[i], schema.items, `${path}/${i}`, errors);
        }
      }
    }

    if (typeof data === "object" && data !== null && !Array.isArray(data)) {
      const obj = data as Record<string, unknown>;

      if (schema.required) {
        for (const req of schema.required) {
          if (!(req in obj)) {
            errors.push({ path: `${path}/${req}`, message: `Required property missing: ${req}` });
          }
        }
      }

      if (schema.properties) {
        for (const [prop, propSchema] of Object.entries(schema.properties)) {
          if (prop in obj) {
            validateNode(obj[prop], propSchema, `${path}/${prop}`, errors);
          }
        }
      }

      if (schema.additionalProperties !== undefined) {
        for (const key of Object.keys(obj)) {
          if (!schema.properties || !(key in schema.properties)) {
            if (schema.additionalProperties === false) {
              errors.push({ path: `${path}/${key}`, message: `Additional property not allowed: ${key}` });
            } else if (typeof schema.additionalProperties === "object") {
              validateNode(obj[key], schema.additionalProperties, `${path}/${key}`, errors);
            }
          }
        }
      }
    }
  }

  return (data: unknown): ValidationResult => {
    const errors: ValidationError[] = [];
    validateNode(data, rootSchema, "#", errors);
    return {
      valid: errors.length === 0,
      errors,
    };
  };
}
