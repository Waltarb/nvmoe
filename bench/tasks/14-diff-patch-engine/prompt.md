# Feature: Unified Diff Patch Engine & 3-Way Merge

Implement `parseUnifiedDiff`, `applyPatch`, and `threeWayMerge` in `src/patch.ts`.

## Data Structures
```ts
export interface HunkLine {
  type: "context" | "add" | "delete";
  content: string;
}

export interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: HunkLine[];
}

export interface PatchResult {
  success: boolean;
  content: string;
  failedHunks: number[];
}

export interface MergeResult {
  merged: string;
  conflicts: boolean;
}
```

## Functions

### 1. `parseUnifiedDiff(diff: string): DiffHunk[]`
- Parses hunks matching header `@@ -oldStart[,oldLines] +newStart[,newLines] @@`.
- Lines starting with `' '` are context (`type: "context"`).
- Lines starting with `'+'` are additions (`type: "add"`).
- Lines starting with `'-'` are deletions (`type: "delete"`).
- `content` is the line text excluding the leading indicator character.

### 2. `applyPatch(original: string, diff: string, options?: { fuzzFactor?: number }): PatchResult`
- Applies parsed hunks sequentially to `original`.
- Tracks cumulative line delta from earlier applied hunks (additions minus deletions) to shift subsequent hunk start lines.
- Fuzzy matching: If the context/delete lines do not match at `targetLine = oldStart - 1 + delta`, searches within `[targetLine - fuzzFactor, targetLine + fuzzFactor]` (clamped to file bounds, where `fuzzFactor` defaults to 0).
- If a hunk cannot match, its 0-based index is added to `failedHunks`, the hunk is skipped, and `success: false`.
- Returns `{ success: failedHunks.length === 0, content, failedHunks }`.

### 3. `threeWayMerge(base: string, ours: string, theirs: string): MergeResult`
- Normalizes CRLF `\r\n` to `\n`.
- Performs 3-way line merge:
  - If a line is identical across all three, keep it.
  - If `ours` modified the line and `theirs` matches `base`, take `ours`.
  - If `theirs` modified the line and `ours` matches `base`, take `theirs`.
  - If both `ours` and `theirs` modified the line identically, take the modification cleanly (`conflicts: false`).
  - If `ours` and `theirs` made conflicting changes, output standard conflict markers and set `conflicts: true`:
    ```
    <<<<<<< OURS
    ours line(s)
    =======
    theirs line(s)
    >>>>>>> THEIRS
    ```
