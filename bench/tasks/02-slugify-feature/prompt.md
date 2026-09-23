# Feature: implement slugify for blog URLs

Implement `slugify(input, options?)` in `src/slugify.ts`.

Rules:
- Lowercase the result.
- Strip diacritics (`é` -> `e`, `ü` -> `u`).
- Replace `&` with the word `and`.
- Every run of characters that are not `a-z` or `0-9` becomes a single `-`.
- No leading or trailing `-`.
- `options.maxLength` (default 60): if the slug is longer, cut it at the last `-` that keeps it within the limit. If there is no `-` within the limit, hard-cut at `maxLength`. Never end with `-`.
- Empty or symbol-only input returns `""`.
