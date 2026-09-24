# Feature: JSON Schema Validator with Cyclic $ref and oneOf

Implement `createValidator(schema, defs?)` in `src/validator.ts` to validate arbitrary JSON data structures against a JSON schema specification.

## Schema Types
```ts
export type SchemaType = "string" | "number" | "boolean" | "null" | "array" | "object";

export interface JSONSchema {
  type?: SchemaType | SchemaType[];
  properties?: Record<string, JSONSchema>;
  required?: string[];
  additionalProperties?: boolean | JSONSchema;
  items?: JSONSchema;
  minItems?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  minimum?: number;
  maximum?: number;
  $ref?: string;
  oneOf?: JSONSchema[];
}

export interface ValidationError {
  path: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}
```

## Requirements & Invariants

1. **Primitive Type Checking**:
   - `"null"` matches `null`. (Note: in JS `typeof null === "object"`, so ensure `data === null` is correctly distinguished).
   - `"array"` matches `Array.isArray(data)`.
   - `"object"` matches non-null objects that are not arrays.
   - If `type` is an array of types, value must match at least one type in the array.
2. **String Constraints**:
   - `minLength` / `maxLength` apply to string character count.
   - `pattern` is a regex string tested with `new RegExp(pattern).test(str)`.
3. **Number Constraints**:
   - `minimum` / `maximum` apply to numbers.
4. **Array Constraints**:
   - `minItems` applies to array length.
   - `items`: each element in array must validate against the sub-schema at path `${path}/${index}`.
5. **Object Constraints**:
   - `required`: every string in `required` must be present on `data`.
   - `properties`: each defined property must validate against its schema if present at `${path}/${prop}`.
   - `additionalProperties`:
     - If `false`, any property on `data` not in `properties` causes an error.
     - If a `JSONSchema` object, any extra property must validate against that schema.
6. **Cyclic / Recursive `$ref`**:
   - Ref paths of the form `"#/$defs/<name>"` or `"#/definitions/<name>"` look up definitions in `defs[name]` or `schema.$defs[name]`.
   - Supports self-referential / recursive structures (e.g. linked lists or trees) without stack overflow.
7. **`oneOf`**:
   - Must validate against EXACTLY one sub-schema. If it matches 0 or >1 schemas, emit a validation error.
