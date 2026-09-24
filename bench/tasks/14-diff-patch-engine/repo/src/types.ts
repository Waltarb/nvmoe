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
