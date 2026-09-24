import { describe, it, expect } from "vitest";
import { parseUnifiedDiff, applyPatch, threeWayMerge } from "../src/patch";

describe("Patch Engine (visible)", () => {
  it("parses and applies a simple unified diff", () => {
    const orig = "alpha\nbeta\ngamma";
    const diff = `--- a/file\n+++ b/file\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma`;

    const hunks = parseUnifiedDiff(diff);
    expect(hunks.length).toBe(1);
    expect(hunks[0].oldStart).toBe(1);

    const res = applyPatch(orig, diff);
    expect(res.success).toBe(true);
    expect(res.content).toBe("alpha\nBETA\ngamma");
  });

  it("performs clean 3-way merge on non-conflicting edits", () => {
    const base = "line1\nline2\nline3";
    const ours = "LINE_ONE\nline2\nline3";
    const theirs = "line1\nline2\nLINE_THREE";

    const res = threeWayMerge(base, ours, theirs);
    expect(res.conflicts).toBe(false);
    expect(res.merged).toBe("LINE_ONE\nline2\nLINE_THREE");
  });
});
