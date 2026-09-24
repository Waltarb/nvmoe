import { describe, it, expect } from "vitest";
import { parseUnifiedDiff, applyPatch, threeWayMerge } from "../src/patch";

describe("Patch Engine (hidden)", () => {
  it("compensates for cumulative line offset across multiple hunks", () => {
    const orig = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"].join("\n");
    // Hunk 1 inserts 2 lines at line 2.
    // Hunk 2 modifies line 8 (which in orig is line 8, but after hunk 1 is now line 10).
    const diff = `
@@ -2,2 +2,4 @@
 2
+2a
+2b
 3
@@ -8,2 +10,2 @@
 8
-9
+NINE
`.trim();

    const res = applyPatch(orig, diff);
    expect(res.success).toBe(true);
    expect(res.content.split("\n")).toEqual([
      "1", "2", "2a", "2b", "3", "4", "5", "6", "7", "8", "NINE", "10"
    ]);
  });

  it("applies fuzzy matching when lines are slightly offset", () => {
    // In original, target is pushed down by 1 line
    const orig = "EXTRA_PREAMBLE\napple\nbanana\ncherry";
    const diff = `@@ -1,3 +1,3 @@\n apple\n-banana\n+BANANA\n cherry`;

    // Without fuzz factor, fails
    const failRes = applyPatch(orig, diff, { fuzzFactor: 0 });
    expect(failRes.success).toBe(false);
    expect(failRes.failedHunks).toEqual([0]);

    // With fuzz factor 2, succeeds
    const okRes = applyPatch(orig, diff, { fuzzFactor: 2 });
    expect(okRes.success).toBe(true);
    expect(okRes.content).toBe("EXTRA_PREAMBLE\napple\nBANANA\ncherry");
  });

  it("emits conflict markers on 3-way merge collision", () => {
    const base = "foo\nbar\nbaz";
    const ours = "foo\nBAR_OURS\nbaz";
    const theirs = "foo\nBAR_THEIRS\nbaz";

    const res = threeWayMerge(base, ours, theirs);
    expect(res.conflicts).toBe(true);
    expect(res.merged).toContain("<<<<<<< OURS");
    expect(res.merged).toContain("BAR_OURS");
    expect(res.merged).toContain("=======");
    expect(res.merged).toContain("BAR_THEIRS");
    expect(res.merged).toContain(">>>>>>> THEIRS");
  });

  it("handles identical concurrent changes cleanly in 3-way merge", () => {
    const base = "hello\nworld";
    const ours = "hello\nWORLD!";
    const theirs = "hello\nWORLD!";

    const res = threeWayMerge(base, ours, theirs);
    expect(res.conflicts).toBe(false);
    expect(res.merged).toBe("hello\nWORLD!");
  });

  it("normalizes CRLF line endings", () => {
    const base = "a\r\nb\r\nc";
    const ours = "a\r\nB_NEW\r\nc";
    const theirs = "a\r\nb\r\nC_NEW";

    const res = threeWayMerge(base, ours, theirs);
    expect(res.conflicts).toBe(false);
    expect(res.merged).toBe("a\nB_NEW\nC_NEW");
  });
});
