import type { DiffHunk, HunkLine, PatchResult, MergeResult } from "../src/types";

export function parseUnifiedDiff(diff: string): DiffHunk[] {
  const lines = diff.replace(/\r\n/g, "\n").split("\n");
  const hunks: DiffHunk[] = [];
  let currentHunk: DiffHunk | null = null;

  const HUNK_HEADER = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

  for (const line of lines) {
    const match = line.match(HUNK_HEADER);
    if (match) {
      if (currentHunk) {
        hunks.push(currentHunk);
      }
      currentHunk = {
        oldStart: parseInt(match[1], 10),
        oldLines: match[2] !== undefined ? parseInt(match[2], 10) : 1,
        newStart: parseInt(match[3], 10),
        newLines: match[4] !== undefined ? parseInt(match[4], 10) : 1,
        lines: [],
      };
      continue;
    }

    if (currentHunk) {
      if (line.startsWith(" ")) {
        currentHunk.lines.push({ type: "context", content: line.slice(1) });
      } else if (line.startsWith("+")) {
        currentHunk.lines.push({ type: "add", content: line.slice(1) });
      } else if (line.startsWith("-")) {
        currentHunk.lines.push({ type: "delete", content: line.slice(1) });
      }
    }
  }

  if (currentHunk) {
    hunks.push(currentHunk);
  }

  return hunks;
}

export function applyPatch(
  original: string,
  diff: string,
  options?: { fuzzFactor?: number }
): PatchResult {
  const hunks = parseUnifiedDiff(diff);
  const fileLines = original.replace(/\r\n/g, "\n").split("\n");
  const fuzz = options?.fuzzFactor ?? 0;
  const failedHunks: number[] = [];
  let lineOffset = 0;

  for (let hIdx = 0; hIdx < hunks.length; hIdx++) {
    const hunk = hunks[hIdx];
    const targetIdx = hunk.oldStart - 1 + lineOffset;

    const oldPattern = hunk.lines
      .filter((l) => l.type === "context" || l.type === "delete")
      .map((l) => l.content);

    // Try finding match within +/- fuzz
    let matchIdx = -1;
    for (let f = 0; f <= fuzz; f++) {
      const candidates = f === 0 ? [targetIdx] : [targetIdx - f, targetIdx + f];
      for (const c of candidates) {
        if (c >= 0 && c + oldPattern.length <= fileLines.length) {
          let matched = true;
          for (let k = 0; k < oldPattern.length; k++) {
            if (fileLines[c + k] !== oldPattern[k]) {
              matched = false;
              break;
            }
          }
          if (matched) {
            matchIdx = c;
            break;
          }
        }
      }
      if (matchIdx !== -1) break;
    }

    if (matchIdx === -1) {
      failedHunks.push(hIdx);
      continue;
    }

    // Apply replacement
    const replacement: string[] = [];
    for (const hl of hunk.lines) {
      if (hl.type === "context" || hl.type === "add") {
        replacement.push(hl.content);
      }
    }

    fileLines.splice(matchIdx, oldPattern.length, ...replacement);
    lineOffset += replacement.length - oldPattern.length;
  }

  return {
    success: failedHunks.length === 0,
    content: fileLines.join("\n"),
    failedHunks,
  };
}

export function threeWayMerge(baseStr: string, oursStr: string, theirsStr: string): MergeResult {
  const base = baseStr.replace(/\r\n/g, "\n").split("\n");
  const ours = oursStr.replace(/\r\n/g, "\n").split("\n");
  const theirs = theirsStr.replace(/\r\n/g, "\n").split("\n");

  const maxLen = Math.max(base.length, ours.length, theirs.length);
  const outLines: string[] = [];
  let hasConflicts = false;

  let i = 0;
  while (i < maxLen) {
    const b = base[i];
    const o = ours[i];
    const t = theirs[i];

    if (o === t) {
      if (o !== undefined) outLines.push(o);
    } else if (o === b) {
      if (t !== undefined) outLines.push(t);
    } else if (t === b) {
      if (o !== undefined) outLines.push(o);
    } else {
      // Conflict
      hasConflicts = true;
      outLines.push("<<<<<<< OURS");
      if (o !== undefined) outLines.push(o);
      outLines.push("=======");
      if (t !== undefined) outLines.push(t);
      outLines.push(">>>>>>> THEIRS");
    }
    i++;
  }

  return {
    merged: outLines.join("\n"),
    conflicts: hasConflicts,
  };
}
