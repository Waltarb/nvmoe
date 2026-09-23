export interface SignupInput {
  email: string;
  password: string;
  age?: number;
}

export type ValidationResult =
  | { ok: true; value: SignupInput }
  | { ok: false; errors: Record<string, string> };

export function validateSignup(body: unknown): ValidationResult {
  // TODO
  return { ok: false, errors: { _: "not_implemented" } };
}
