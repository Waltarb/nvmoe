import { readdir, mkdir, appendFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { runTask, type TaskMeta, type TaskResult } from "./task.ts";
import { summarize } from "./report.ts";
import { BENCH_ROOT, readText } from "./util.ts";

const { values } = parseArgs({
  options: {
    config: { type: "string" },
    tier: { type: "string", default: "smoke" },
    runs: { type: "string", default: "1" },
    task: { type: "string" },
    timeout: { type: "string" },
    noTimeout: { type: "boolean" },
    budget: { type: "string" },
  },
});
if (values.noTimeout) process.env.BENCH_NO_TIMEOUT = "1";
if (values.timeout) process.env.BENCH_TIMEOUT_SEC = values.timeout;
if (values.budget) process.env.BENCH_OUTPUT_BUDGET = values.budget;
if (!values.config) { console.error("usage: pnpm bench --config <name> [--tier smoke|core|full] [--runs N] [--task id] [--noTimeout] [--budget N]"); process.exit(1); }

const tiers = {
  smoke: ["smoke"],
  core: ["core"],
  "smoke+core": ["smoke", "core"],
  full: ["smoke", "core", "full"],
  all: ["smoke", "core", "full"],
} as const;
const allowed = tiers[values.tier as keyof typeof tiers] ?? tiers.smoke;
const tasksDir = join(BENCH_ROOT, "tasks");
const tasks: { dir: string; meta: TaskMeta }[] = [];
for (const id of (await readdir(tasksDir)).sort()) {
  const meta: TaskMeta = { id, ...JSON.parse(await readText(join(tasksDir, id, "task.json"))) };
  if (values.task ? meta.id === values.task : (allowed as readonly string[]).includes(meta.tier)) tasks.push({ dir: join(tasksDir, id), meta });
}

const outDir = join(BENCH_ROOT, "results", values.config);
await mkdir(outDir, { recursive: true });
const outFile = join(outDir, `${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`);
const results: TaskResult[] = [];

console.log(`${values.config}: ${tasks.length} task(s) x ${values.runs} run(s) -> ${outFile}`);
for (let runNo = 1; runNo <= Number(values.runs); runNo++) {
  for (const { dir, meta } of tasks) {
    process.stdout.write(`[run ${runNo}] ${meta.id} ... `);
    const r = await runTask(dir, meta, values.config, runNo);
    results.push(r);
    await appendFile(outFile, JSON.stringify(r) + "\n");
    console.log(`${r.solved ? "PASS" : "FAIL"} hidden ${r.hiddenPassed}/${r.hiddenTotal} tsc:${r.tscOk ? "ok" : "x"} ` +
      `${r.stopReason} ${r.turns}t ${r.outputTokens}tok ${(r.wallMs / 60000).toFixed(1)}min${r.error ? ` ERR ${r.error}` : ""}`);
  }
}
console.table([summarize(results)]);
