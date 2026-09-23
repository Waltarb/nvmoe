// Compares the latest result file of every config: pnpm compare [config ...]
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { summarize } from "./report.ts";
import { BENCH_ROOT, readText, exists } from "./util.ts";

const root = join(BENCH_ROOT, "results");
if (!(await exists(root))) { console.error("no results yet"); process.exit(1); }
const wanted = process.argv.slice(2);
const rows = [];
for (const cfg of (await readdir(root)).sort()) {
  if (wanted.length && !wanted.includes(cfg)) continue;
  const files = (await readdir(join(root, cfg))).filter((f) => f.endsWith(".jsonl")).sort();
  if (!files.length) continue;
  const lines = (await readText(join(root, cfg, files.at(-1)!))).trim().split("\n").filter(Boolean);
  rows.push(summarize(lines.map((l) => JSON.parse(l))));
}
console.table(rows);
