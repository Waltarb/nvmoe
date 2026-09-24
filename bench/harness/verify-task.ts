import { cp, rm, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { BENCH_ROOT, bin, run, exists } from "./util.ts";

const taskId = process.argv[2];
if (!taskId) {
  console.error("Usage: tsx harness/verify-task.ts <taskId>");
  process.exit(1);
}

const taskDir = join(BENCH_ROOT, "tasks", taskId);
const testWorkdir = join(BENCH_ROOT, ".work", `verify-${taskId}`);

async function testDir(label: string) {
  const outFile = join(testWorkdir, `.vitest-verify.json`);
  await rm(outFile, { force: true });
  const r = await run(bin("vitest"), ["run", "--root", testWorkdir, "--reporter=json", `--outputFile=${outFile}`], BENCH_ROOT, 30_000);
  if (!(await exists(outFile))) {
    console.log(`[${label}] Vitest failed to produce JSON output:`, r.out);
    return { passed: 0, total: 0 };
  }
  const j = JSON.parse(await readFile(outFile, "utf8"));
  return { passed: j.numPassedTests ?? 0, total: j.numTotalTests ?? 0 };
}

async function verify() {
  console.log(`Verifying ${taskId}...`);
  await rm(testWorkdir, { recursive: true, force: true });
  await mkdir(testWorkdir, { recursive: true });
  await cp(join(taskDir, "repo"), testWorkdir, { recursive: true });
  await cp(join(taskDir, "hidden"), join(testWorkdir, "hidden"), { recursive: true });
  await cp(join(BENCH_ROOT, "harness", "tsconfig.task.json"), join(testWorkdir, "tsconfig.json"));

  // 1. Starter code test
  const starterRes = await testDir("STARTER");
  console.log(`Starter result: ${starterRes.passed}/${starterRes.total} tests passed (Expected < total)`);

  // 2. Reference code test
  if (await exists(join(taskDir, "reference"))) {
    await cp(join(taskDir, "reference"), join(testWorkdir, "src"), { recursive: true });
    const refRes = await testDir("REFERENCE");
    console.log(`Reference result: ${refRes.passed}/${refRes.total} tests passed`);

    // 3. Typecheck
    const tscRes = await run(bin("tsc"), ["-p", join(testWorkdir, "tsconfig.json")], BENCH_ROOT, 30_000);
    console.log(`Reference tsc: code=${tscRes.code} (${tscRes.code === 0 ? "OK" : "ERR:\n" + tscRes.out})`);

    if (refRes.passed === refRes.total && refRes.total > 0 && tscRes.code === 0) {
      console.log(`✅ ${taskId} VERIFIED PERFECTLY!`);
    } else {
      console.error(`❌ ${taskId} FAILED VERIFICATION!`);
      process.exit(1);
    }
  }
}

verify().catch((e) => {
  console.error("Verification error:", e);
  process.exit(1);
});
