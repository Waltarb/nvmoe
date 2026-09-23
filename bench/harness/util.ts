import { spawn } from "node:child_process";
import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const BENCH_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const bin = (name: string) => join(BENCH_ROOT, "node_modules", ".bin", name);

export function run(cmd: string, args: string[], cwd: string, timeoutMs: number) {
  return new Promise<{ code: number | null; out: string; timedOut: boolean }>((resolve) => {
    const p = spawn(cmd, args, { cwd, env: { ...process.env, FORCE_COLOR: "0" } });
    let out = "";
    let timedOut = false;
    const t = setTimeout(() => { timedOut = true; p.kill("SIGKILL"); }, timeoutMs);
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (out += d));
    p.on("close", (code) => { clearTimeout(t); resolve({ code, out, timedOut }); });
  });
}

export async function listFiles(dir: string, base = dir): Promise<string[]> {
  const out: string[] = [];
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...(await listFiles(p, base)));
    else out.push(relative(base, p));
  }
  return out.sort();
}

export const exists = (p: string) => stat(p).then(() => true, () => false);
export const readText = (p: string) => readFile(p, "utf8");
export const approxTokens = (s: string) => Math.ceil(s.length / 4);
export const trimLines = (s: string, n: number) => s.split("\n").slice(0, n).join("\n");
export const median = (xs: number[]) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
