import type { DiffHunk, PatchResult, MergeResult } from "./types";

export function parseUnifiedDiff(diff: string): DiffHunk[] {
  return [];
}

export function applyPatch(
  original: string,
  diff: string,
  options?: { fuzzFactor?: number }
): PatchResult {
  return { success: false, content: original, failedHunks: [] };
}

export function threeWayMerge(base: string, ours: string, theirs: string): MergeResult {
  return { merged: "", conflicts: false };
}
