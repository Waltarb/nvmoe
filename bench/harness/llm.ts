// Minimal streaming client for any OpenAI-compatible /chat/completions endpoint.
export interface Msg { role: "system" | "user" | "assistant"; content: string }
export interface ChatResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  usageEstimated: boolean;
  ttftMs: number;
  totalMs: number;
}

const ENDPOINT = process.env.BENCH_ENDPOINT ?? "http://localhost:8080/v1";
const MODEL = process.env.BENCH_MODEL ?? "local";
const API_KEY = process.env.BENCH_API_KEY ?? "none";
// Extra request fields, e.g. '{"chat_template_kwargs":{"enable_thinking":false}}'
const EXTRA = JSON.parse(process.env.BENCH_EXTRA_BODY ?? "{}");

import { Agent } from "undici";
import { writeFileSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { BENCH_ROOT } from "./util.ts";

const LIVE_STREAM_FILE = join(BENCH_ROOT, ".live_stream.txt");

const agent = new Agent({
  headersTimeout: 0,
  bodyTimeout: 0,
  connectTimeout: 60_000,
});

export async function chat(messages: Msg[], maxTokens: number, timeoutMs: number): Promise<ChatResult> {
  const t0 = performance.now();
  const ctrl = new AbortController();
  const timer = (timeoutMs > 0 && Number.isFinite(timeoutMs)) ? setTimeout(() => ctrl.abort(), timeoutMs) : undefined;
  try { writeFileSync(LIVE_STREAM_FILE, ""); } catch {}
  let ttft = -1;
  let text = "";
  let reasoningChars = 0;
  let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;

  try {
    const res = await fetch(`${ENDPOINT}/chat/completions`, {
      method: "POST",
      signal: ctrl.signal,
      dispatcher: agent,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
      body: JSON.stringify({
        model: MODEL, messages, max_tokens: maxTokens, temperature: 0, stream: true,
        stream_options: { include_usage: true }, ...EXTRA,
      }),
    } as any);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        const chunk = JSON.parse(data);
        if (chunk.usage) usage = chunk.usage;
        const d = chunk.choices?.[0]?.delta ?? {};
        const piece: string = d.content ?? "";
        const reasoning: string = d.reasoning_content ?? d.reasoning ?? "";
        if ((piece || reasoning) && ttft < 0) ttft = performance.now() - t0;
        text += piece;
        reasoningChars += reasoning.length;
        if (piece || reasoning) {
          try { appendFileSync(LIVE_STREAM_FILE, piece || reasoning); } catch {}
        }
      }
    }
  } catch (err: any) {
    if (err?.name === "AbortError" && text.length > 0) {
      console.warn("  [warn] Stream aborted by timeout, preserving partial response");
    } else {
      throw err;
    }
  } finally {
    if (timer) clearTimeout(timer);
  }

  const totalMs = performance.now() - t0;
  const estimated = !usage?.completion_tokens;
  return {
    text,
    promptTokens: usage?.prompt_tokens ?? Math.ceil(JSON.stringify(messages).length / 4),
    completionTokens: usage?.completion_tokens ?? Math.ceil((text.length + reasoningChars) / 4),
    usageEstimated: estimated,
    ttftMs: ttft < 0 ? totalMs : ttft,
    totalMs,
  };
}
