import { cp, rm, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { parseEdits, applyEdits } from "./edits.ts";
import { BENCH_ROOT, bin, run, exists, trimLines } from "./util.ts";

async function testDir(workdir: string, filter: string) {
  const outFile = join(workdir, `.vitest-${filter.replace("/", "")}.json`);
  await rm(outFile, { force: true });
  const r = await run(bin("vitest"), ["run", filter, "--root", workdir, "--reporter=json", `--outputFile=${outFile}`], BENCH_ROOT, 30_000);
  if (!(await exists(outFile))) {
    return { passed: 0, total: 0, failure: trimLines(r.out.trim(), 20) };
  }
  const j = JSON.parse(await readFile(outFile, "utf8"));
  let failure = "";
  for (const f of j.testResults ?? []) {
    const a = (f.assertionResults ?? []).find((x: any) => x.status === "failed");
    if (a) { failure = `${a.fullName}\n${a.failureMessages?.[0] ?? ""}`; break; }
    if (f.status === "failed" && f.message) { failure = f.message; break; }
  }
  return { passed: j.numPassedTests ?? 0, total: j.numTotalTests ?? 0, failure: trimLines(failure, 20) };
}

export async function evaluateCandidate(taskId: string, candidateResponse: string, label: string) {
  const taskDir = join(BENCH_ROOT, "tasks", taskId);
  const workdir = join(BENCH_ROOT, ".work", `eval-${label}-${taskId}`);

  await rm(workdir, { recursive: true, force: true });
  await mkdir(workdir, { recursive: true });
  await cp(join(taskDir, "repo"), workdir, { recursive: true });
  await cp(join(BENCH_ROOT, "harness", "tsconfig.task.json"), join(workdir, "tsconfig.json"));

  console.log(`\n============================================================`);
  console.log(`Evaluating [${label}] on Task: ${taskId}`);
  console.log(`============================================================`);

  const edits = parseEdits(candidateResponse);
  console.log(`Edits parsed: ${edits.length}`);
  const errs = await applyEdits(workdir, edits);
  if (errs.length) {
    console.error(`Apply errors:\n${errs.join("\n")}`);
  }

  // 1. Run spec tests
  const spec = await testDir(workdir, "spec/");
  console.log(`Spec tests: ${spec.passed}/${spec.total} passed`);
  if (spec.failure) console.log(`Spec failure: ${spec.failure}`);

  // 2. Run hidden tests
  await cp(join(taskDir, "hidden"), join(workdir, "hidden"), { recursive: true });
  const hidden = await testDir(workdir, "hidden/");
  console.log(`Hidden tests: ${hidden.passed}/${hidden.total} passed`);
  if (hidden.failure) console.log(`Hidden failure: ${hidden.failure}`);

  // 3. Type check
  const tscRes = await run(bin("tsc"), ["-p", join(workdir, "tsconfig.json")], BENCH_ROOT, 30_000);
  const tscOk = tscRes.code === 0;
  console.log(`TypeScript check: ${tscOk ? "PASSED (0 errors)" : "FAILED:\n" + tscRes.out}`);

  const solved = hidden.total > 0 && hidden.passed === hidden.total && tscOk;
  console.log(`Task Solved: ${solved ? "✅ PASS (100/100)" : `❌ FAIL (${hidden.passed}/${hidden.total})`}`);
  return { spec, hidden, tscOk, solved };
}
