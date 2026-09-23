#!/usr/bin/env python3
"""Controlled single-prompt decode benchmark: temperature=0, fixed prompt, many
tokens, to average per-token NVMe-miss variance out of the steady-state tok/s
comparison between the eager and layer-graph runners."""
import time
import json
import sys
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-nvfp4-ftw"


def run(max_tokens: int, label: str):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 200, one number per line, no other text."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, stream=True, timeout=600)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)
    ttft = None
    token_times = []
    n = 0
    last = time.perf_counter()
    for line in resp.iter_lines():
        if not line:
            continue
        d = line.decode("utf-8")
        if not d.startswith("data: "):
            continue
        s = d[6:].strip()
        if s == "[DONE]":
            break
        now = time.perf_counter()
        try:
            chunk = json.loads(s)
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("reasoning_content") or delta.get("content")
            if text:
                n += 1
                if ttft is None:
                    ttft = now - t0
                    last = now
                else:
                    token_times.append(now - last)
                    last = now
        except Exception:
            pass
    total = time.perf_counter() - t0
    steady = token_times[5:] if len(token_times) > 5 else token_times  # drop warmup tokens
    avg_ms = sum(steady) / len(steady) * 1000 if steady else 0.0
    tok_s = 1000.0 / avg_ms if avg_ms else 0.0
    print(f"[{label}] tokens={n} ttft={ttft:.3f}s total={total:.3f}s "
          f"steady_avg={avg_ms:.1f}ms steady_tok_s={tok_s:.3f} "
          f"(n_steady={len(steady)}, min={min(steady)*1000:.1f}ms, max={max(steady)*1000:.1f}ms)"
          if steady else f"[{label}] tokens={n} no steady samples")
    return tok_s


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    run(max_tokens, label)
