import { cp, rm, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { chat, type Msg } from "./llm.ts";
import { parseEdits, applyEdits } from "./edits.ts";
import { BENCH_ROOT, bin, run, listFiles, readText, approxTokens, trimLines, exists } from "./util.ts";

export interface TaskMeta {
  id: string;
  tier: "smoke" | "core" | "full";
  category: string;
  maxTurns: number;
  outputBudget: number;   // total completion tokens across all turns
  timeoutSec: number;     // wall-clock budget for the whole task
}

export interface TaskResult {
  config: string; task: string; category: string; run: number;
  solved: boolean; hiddenPassed: number; hiddenTotal: number; specGreen: boolean; tscOk: boolean;
  turns: number; formatErrors: number; applyErrors: number;
  stopReason: "spec_green" | "done" | "budget" | "timeout" | "turns" | "error";
  promptTokens: number; outputTokens: number; usageEstimated: boolean;
  ttftMs: number[]; decodeTokS: number[]; wallMs: number; error?: string;
}

const SYSTEM = `You are a senior TypeScript developer working in a small repo.
Keep any internal reasoning (<think>) concise (under 200 words) and proceed directly to outputting the complete SEARCH/REPLACE code edits.
Reply ONLY with edit blocks in exactly this format, one block per change:

Example edit:
src/math.ts
<<<<<<< SEARCH
export function add(a: number, b: number): number {
  return 0;
}
=======
export function add(a: number, b: number): number {
  return a + b;
}
>>>>>>> REPLACE

Rules:
- Do NOT output conversational chatter or explanations. Start immediately with the file path.
- SEARCH must match the current file exactly, including whitespace. Keep it short but unique.
- To create a new file or replace the entire file, leave SEARCH empty:
path/to/file.ts
<<<<<<< SEARCH
=======
// replacement content
>>>>>>> REPLACE
- Do not edit files under spec/.
- After your edits the spec tests run automatically and you will see the result.
- When the task is complete, write DONE on its own line.`;

interface TestRun { passed: number; total: number; failure: string }

async function runTests(workdir: string, filter: string): Promise<TestRun> {
  const outFile = join(workdir, `.vitest-${filter.replace("/", "")}.json`);
  await rm(outFile, { force: true });
  const r = await run(bin("vitest"), ["run", filter, "--root", workdir, "--reporter=json", `--outputFile=${outFile}`], BENCH_ROOT, 120_000);
  if (!(await exists(outFile))) return { passed: 0, total: 0, failure: trimLines(r.out.trim(), 20) };
  const j = JSON.parse(await readFile(outFile, "utf8"));
  let failure = "";
  for (const f of j.testResults ?? []) {
    const a = (f.assertionResults ?? []).find((x: any) => x.status === "failed");
    if (a) { failure = `${a.fullName}\n${a.failureMessages?.[0] ?? ""}`; break; }
    if (f.status === "failed" && f.message) { failure = f.message; break; }
  }
  return { passed: j.numPassedTests ?? 0, total: j.numTotalTests ?? 0, failure: trimLines(failure, 20) };
}

async function buildPrompt(taskDir: string, workdir: string): Promise<string> {
  const ticket = await readText(join(taskDir, "prompt.md"));
  const files = await listFiles(workdir);
  const shown = files.filter((f) => /^(src|spec)\//.test(f));
  const blocks = await Promise.all(shown.map(async (f) => `### ${f}\n\`\`\`ts\n${await readText(join(workdir, f))}\`\`\``));
  return `${ticket.trim()}\n\n## Files\n\n${blocks.join("\n\n")}`;
}

export async function runTask(taskDir: string, meta: TaskMeta, config: string, runNo: number): Promise<TaskResult> {
  const workdir = join(BENCH_ROOT, ".work", `${config}-${meta.id}-${runNo}`);
  await rm(workdir, { recursive: true, force: true });
  await mkdir(workdir, { recursive: true });
  await cp(join(taskDir, "repo"), workdir, { recursive: true });
  await cp(join(BENCH_ROOT, "harness", "tsconfig.task.json"), join(workdir, "tsconfig.json"));

  const res: TaskResult = {
    config, task: meta.id, category: meta.category, run: runNo,
    solved: false, hiddenPassed: 0, hiddenTotal: 0, specGreen: false, tscOk: false,
    turns: 0, formatErrors: 0, applyErrors: 0, stopReason: "turns",
    promptTokens: 0, outputTokens: 0, usageEstimated: false, ttftMs: [], decodeTokS: [], wallMs: 0,
  };

  const prompt = await buildPrompt(taskDir, workdir);
  if (approxTokens(SYSTEM + prompt) > 4000) console.warn(`  ! ${meta.id}: prompt ~${approxTokens(SYSTEM + prompt)} tokens, over the 4k budget`);
  const messages: Msg[] = [{ role: "system", content: SYSTEM }, { role: "user", content: prompt }];
  const t0 = performance.now();
  const noTimeout = process.env.BENCH_NO_TIMEOUT === "1" || process.env.BENCH_TIMEOUT_SEC === "0";
  const timeoutSec = noTimeout ? 0 : (process.env.BENCH_TIMEOUT_SEC ? parseInt(process.env.BENCH_TIMEOUT_SEC, 10) : (meta.timeoutSec || 1800));
  const deadline = timeoutSec > 0 ? t0 + timeoutSec * 1000 : Infinity;

  try {
    const totalBudget = process.env.BENCH_OUTPUT_BUDGET ? parseInt(process.env.BENCH_OUTPUT_BUDGET, 10) : (meta.outputBudget || 4000);
    for (let turn = 1; turn <= meta.maxTurns; turn++) {
      const remaining = totalBudget - res.outputTokens;
      const timeLeft = timeoutSec > 0 ? (deadline - performance.now()) : 0;
      if (remaining <= 0) { res.stopReason = "budget"; break; }
      if (timeoutSec > 0 && timeLeft <= 0) { res.stopReason = "timeout"; break; }

      let r;
      try { r = await chat(messages, remaining, timeLeft); }
      catch (e: any) {
        if (e?.name === "AbortError") { res.stopReason = "timeout"; break; }
        throw e;
      }
      res.turns = turn;
      res.promptTokens += r.promptTokens;
      res.outputTokens += r.completionTokens;
      res.usageEstimated ||= r.usageEstimated;
      res.ttftMs.push(Math.round(r.ttftMs));
      const decodeMs = r.totalMs - r.ttftMs;
      if (decodeMs > 0 && r.completionTokens > 1) res.decodeTokS.push(+(r.completionTokens / (decodeMs / 1000)).toFixed(2));
      messages.push({ role: "assistant", content: r.text });

      console.log(`\n=== MODEL RESPONSE (Turn ${turn}, ${r.completionTokens} tok) ===\n${r.text}\n=== END MODEL RESPONSE ===`);
      const edits = parseEdits(r.text);
      const saidDone = /^\s*DONE\s*$/m.test(r.text);
      console.log(`Edits parsed: ${edits.length}, saidDone: ${saidDone}`);
      const feedback: string[] = [];

      if (!edits.length && !saidDone) {
        res.formatErrors++;
        feedback.push("No valid edit blocks found. Use the exact SEARCH/REPLACE format from the instructions.");
      }
      const errs = await applyEdits(workdir, edits);
      res.applyErrors += errs.length;
      if (errs.length) feedback.push(`Edit errors:\n${errs.join("\n")}`);

      const spec = await runTests(workdir, "spec/");
      res.specGreen = spec.total > 0 && spec.passed === spec.total;
      if (res.specGreen && !errs.length) { res.stopReason = "spec_green"; break; }
      if (saidDone) { res.stopReason = "done"; break; }

      feedback.push(`Spec tests: ${spec.passed}/${spec.total} passed.${spec.failure ? `\nFirst failure:\n${spec.failure}` : ""}`);
      messages.push({ role: "user", content: feedback.join("\n\n") });
    }
  } catch (e: any) {
    res.stopReason = "error";
    res.error = String(e?.message ?? e);
  }
  res.wallMs = Math.round(performance.now() - t0);

  // Final scoring: hidden tests + typecheck, never shown to the model.
  await cp(join(taskDir, "hidden"), join(workdir, "hidden"), { recursive: true });
  const hidden = await runTests(workdir, "hidden/");
  res.hiddenPassed = hidden.passed;
  res.hiddenTotal = hidden.total;
  res.tscOk = (await run(bin("tsc"), ["-p", join(workdir, "tsconfig.json")], BENCH_ROOT, 120_000)).code === 0;
  res.solved = hidden.total > 0 && hidden.passed === hidden.total && res.tscOk;
  return res;
}
