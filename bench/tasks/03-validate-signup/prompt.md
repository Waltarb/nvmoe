# Feature: validate the signup request body

Implement `validateSignup(body)` in `src/validate.ts`. It is called by the `POST /signup` handler with the parsed JSON body. Collect **all** errors, don't stop at the first one.

- If `body` is not a plain object (null, array, string...), return `{ ok: false, errors: { _: "invalid_body" } }`.
- `email`: required string. Trim and lowercase it. Must match `something@something.tld` (no spaces). Errors: `"required"`, `"invalid"`.
- `password`: required string, at least 8 characters, must contain a digit. Errors: `"required"`, `"too_short"`, `"needs_digit"` (check length first; report only one error per field).
- `age`: optional. If present, must be an integer from 13 to 120. Error: `"invalid"`.
- On success return `{ ok: true, value: { email, password, age? } }` with the normalised email. Omit `age` from `value` when it wasn't given.
