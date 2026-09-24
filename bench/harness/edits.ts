import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, resolve, dirname, sep } from "node:path";

export interface Edit { path: string; search: string; replace: string }

const PATH_LINE = /(?:^|\n)(?:```[a-z]*\s*\n)?(?:[#*\s`]*)([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)(?:[*:\s`]*)(?=\n)/g;

export function parseEdits(text: string): Edit[] {
  let cleaned = text.replace(/\r\n/g, "\n");
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

  if (!edits.length) {
    let codeText = cleaned;
    // If the model was cut off mid-code-block, close it
    const openBlock = codeText.match(/```[a-z]*\n(?![\s\S]*```)/);
    if (openBlock) {
      codeText += "\n```";
    }

    const codeBlockRegex = /(?:^|\n)(?:[#*\s`]*)([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)[*:\s`]*\n```[a-z]*\n([\s\S]*?)\n```/g;
    for (const m of codeText.matchAll(codeBlockRegex)) {
      const filePath = m[1].trim();
      const code = m[2];
      if (filePath.startsWith("src/") || filePath.startsWith("lib/")) {
        edits.push({ path: filePath, search: "", replace: code });
      }
    }

    // Fallback: If code block has no explicit file header, infer from exported class or function
    if (!edits.length) {
      const bareBlockRegex = /```(?:ts|typescript)\n([\s\S]*?)\n```/g;
      const symbolToFile: Record<string, string> = {
        BatchScheduler: "src/scheduler.ts",
        AsyncBatchScheduler: "src/scheduler.ts",
        SegmentedCache: "src/segmented-cache.ts",
        SegmentedLruCache: "src/segmented-cache.ts",
        encodeFrame: "src/codec.ts",
        FrameDecoder: "src/codec.ts",
        WebSocketFrameCodec: "src/codec.ts",
        Transaction: "src/mvcc.ts",
        MvccStore: "src/mvcc.ts",
        MvccTransactionKv: "src/mvcc.ts",
        RaftNode: "src/raft.ts",
        RaftStateMachine: "src/raft.ts",
        optimizeAST: "src/optimizer.ts",
        ExpressionAstOptimizer: "src/optimizer.ts",
        RateLimiter: "src/limiter.ts",
        TokenBucketRateLimiter: "src/limiter.ts",
        createValidator: "src/validator.ts",
        JsonSchemaValidator: "src/validator.ts",
        parseUnifiedDiff: "src/patch.ts",
        applyPatch: "src/patch.ts",
        threeWayMerge: "src/patch.ts",
        DiffPatchEngine: "src/patch.ts",
        TypedEmitter: "src/emitter.ts",
        paginate: "src/paginate.ts",
        slugify: "src/slugify.ts",
        validateSignup: "src/validate.ts",
        cartReducer: "src/cart.ts",
        splitName: "src/name.ts",
      };

      for (const m of codeText.matchAll(bareBlockRegex)) {
        const code = m[1];
        let target = "";
        for (const [sym, file] of Object.entries(symbolToFile)) {
          if (code.includes(sym)) {
            target = file;
            break;
          }
        }
        if (target) {
          edits.push({ path: target, search: "", replace: code });
        }
      }
    }
  }

  return edits;
}

export async function applyEdits(workdir: string, edits: Edit[]): Promise<string[]> {
  const errors: string[] = [];
  for (const e of edits) {
    const abs = resolve(workdir, e.path);
    if (!abs.startsWith(resolve(workdir) + sep)) { errors.push(`${e.path}: path outside repo`); continue; }
    if (e.path.startsWith("spec/") || e.path.startsWith("hidden/")) {
      errors.push(`${e.path}: editing test files is forbidden. Modify implementation in src/ only.`);
      continue;
    }
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
