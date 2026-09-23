# Refactor: split `fullName` into `firstName` / `lastName`

Replace `User.fullName` with `firstName` and `lastName` across the codebase. The project must type-check.

- `toUser(raw)` in `src/api.ts` splits `raw.name` on the **first** run of whitespace after trimming: `"Ada Lovelace"` -> `Ada` / `Lovelace`, `"Jean Luc  Picard"` -> `Jean` / `Luc  Picard`, `"Cher"` -> `Cher` / `""`.
- `formatUser(user)` in `src/format.ts` returns `"Lastname, Firstname"`, or just the first name when `lastName` is empty.
- `initials(user)` in `src/format.ts` returns the uppercase first letter of each non-empty name part: `"AL"`, or `"C"` for Cher.
