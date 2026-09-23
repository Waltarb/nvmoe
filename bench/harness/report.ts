import type { TaskResult } from "./task.ts";
import { median } from "./util.ts";

export function summarize(rs: TaskResult[]) {
  const solved = rs.filter((r) => r.solved);
  const out = rs.reduce((s, r) => s + r.outputTokens, 0);
  const hiddenPassed = rs.reduce((s, r) => s + r.hiddenPassed, 0);
  return {
    config: rs[0]?.config ?? "?",
    runs: rs.length,
    passRate: `${((solved.length / Math.max(1, rs.length)) * 100).toFixed(0)}%`,
    solved: `${solved.length}/${rs.length}`,
    medWallSolvedMin: +(median(solved.map((r) => r.wallMs)) / 60000).toFixed(1),
    testsPer1kOut: +((hiddenPassed / Math.max(1, out)) * 1000).toFixed(2),
    outTokens: out,
    decodeTokS: +median(rs.flatMap((r) => r.decodeTokS)).toFixed(2),
    ttftS: +(median(rs.flatMap((r) => r.ttftMs)) / 1000).toFixed(1),
    formatErr: rs.reduce((s, r) => s + r.formatErrors, 0),
    applyErr: rs.reduce((s, r) => s + r.applyErrors, 0),
    budgetOrTimeout: rs.filter((r) => r.stopReason === "budget" || r.stopReason === "timeout").length,
    errors: rs.filter((r) => r.stopReason === "error").length,
  };
}
