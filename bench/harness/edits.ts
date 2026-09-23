import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, resolve, dirname, sep } from "node:path";

export interface Edit { path: string; search: string; replace: string }

const PATH_LINE = /(?:^|\n)(?:```[a-z]*\s*\n)?(?:[#*\s`]*)([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)(?:[*:\s`]*)(?=\n)/g;

export function parseEdits(text: string): Edit[] {
  let cleaned = text;
  if (cleaned.includes("</think>")) {
    cleaned = cleaned.slice(cleaned.lastIndexOf("</think>") + "</think>".length);
  }
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>|<\/?think>/g, "").trim();
  const edits: Edit[] = [];
  const parts = cleaned.split("<<<<<<< SEARCH\n");
  let currentPath = "";

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (i === 0) {
      const matches = [...part.matchAll(PATH_LINE)];
      if (matches.length) currentPath = matches.at(-1)![1].trim();
      continue;
    }
    let sepIdx = -1;
    let search = "";
    let replaceStart = -1;
    if (part.startsWith("=======\n")) {
      sepIdx = 0;
      search = "";
      replaceStart = "=======\n".length;
    } else {
      sepIdx = part.indexOf("\n=======\n");
      if (sepIdx !== -1) {
        search = part.slice(0, sepIdx + 1);
        replaceStart = sepIdx + "\n=======\n".length;
      }
    }
    const endIdx = part.indexOf("\n>>>>>>> REPLACE");
    if (sepIdx !== -1 && endIdx !== -1 && endIdx >= sepIdx) {
      const replace = part.slice(replaceStart, endIdx + 1);
      if (currentPath) {
        edits.push({ path: currentPath, search, replace });
      }
      const after = part.slice(endIdx + "\n>>>>>>> REPLACE".length);
      const matches = [...after.matchAll(PATH_LINE)];
      if (matches.length) currentPath = matches.at(-1)![1].trim();
    }
  }

  return edits;
}

export async function applyEdits(workdir: string, edits: Edit[]): Promise<string[]> {
  const errors: string[] = [];
  for (const e of edits) {
    const abs = resolve(workdir, e.path);
    if (!abs.startsWith(resolve(workdir) + sep)) { errors.push(`${e.path}: path outside repo`); continue; }
    if (e.search === "") {
      await mkdir(dirname(abs), { recursive: true });
      await writeFile(abs, e.replace);
      continue;
    }
    let cur: string;
    try { cur = await readFile(abs, "utf8"); } catch { errors.push(`${e.path}: file not found`); continue; }
    const hits = cur.split(e.search).length - 1;
    if (hits === 0) errors.push(`${e.path}: SEARCH block not found (must match exactly)`);
    else if (hits > 1) errors.push(`${e.path}: SEARCH block matches ${hits} places, make it unique`);
    else await writeFile(abs, cur.replace(e.search, () => e.replace));
  }
  return errors;
}
