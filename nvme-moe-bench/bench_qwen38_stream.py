#!/usr/bin/env python3
"""
Streaming Benchmark for Qwen3.8-Flash-Next-NVFP4 on FreeToken 3-Tier NVMe Offload Engine.
Measures TTFT, per-token latency, tokens/sec, and reads cache telemetry from server log.
"""

import time
import json
import requests
import sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-nvfp4-ftw"

def run_bench(prompt: str, max_tokens: int = 60, reasoning_effort: str = None, label: str = "Test"):
    print(f"\n{'='*60}")
    print(f"RUNNING: {label} (max_tokens={max_tokens}, prompt='{prompt[:50]}...')")
    print(f"{'='*60}")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, stream=True, timeout=300)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    ttft = None
    token_times = []
    reasoning_tokens = 0
    content_tokens = 0
    reasoning_text = []
    content_text = []

    last_time = time.perf_counter()
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8')
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:].strip()
        if data_str == "[DONE]":
            break

        now = time.perf_counter()
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            is_real_token = False
            if "reasoning_content" in delta and delta["reasoning_content"]:
                is_real_token = True
                reasoning_tokens += 1
                reasoning_text.append(delta["reasoning_content"])
                sys.stdout.write(f"\033[90m{delta['reasoning_content']}\033[0m")
                sys.stdout.flush()
            if "content" in delta and delta["content"]:
                is_real_token = True
                content_tokens += 1
                content_text.append(delta["content"])
                sys.stdout.write(delta["content"])
                sys.stdout.flush()

            if is_real_token:
                if ttft is None:
                    ttft = now - t0
                    last_time = now
                else:
                    token_times.append(now - last_time)
                    last_time = now
        except Exception:
            pass

    t_end = time.perf_counter()
    total_time = t_end - t0
    total_gen_tokens = reasoning_tokens + content_tokens
    decode_time = total_time - (ttft if ttft else 0)
    decode_tok_per_sec = (total_gen_tokens - 1) / decode_time if (decode_time > 0 and total_gen_tokens > 1) else 0.0

    print("\n" + "-"*60)
    print(f"Results for: {label}")
    print(f"Total Tokens Generated: {total_gen_tokens} ({reasoning_tokens} reasoning, {content_tokens} content)")
    print(f"TTFT (Prefill + First Token): {ttft:.3f} s" if ttft else "TTFT: N/A")
    print(f"Total End-to-End Time:        {total_time:.3f} s")
    print(f"Decode Time (after TTFT):     {decode_time:.3f} s")
    print(f"Decode Throughput:            {decode_tok_per_sec:.2f} tok/s")
    if token_times:
        avg_step_ms = (sum(token_times) / len(token_times)) * 1000
        min_step_ms = min(token_times) * 1000
        max_step_ms = max(token_times) * 1000
        sorted_times = sorted(token_times)
        median_ms = sorted_times[len(sorted_times) // 2] * 1000
        print(f"Token Latency (ms):           avg={avg_step_ms:.1f}ms, med={median_ms:.1f}ms, min={min_step_ms:.1f}ms, max={max_step_ms:.1f}ms")
        first_5 = [f"{t*1000:.1f}ms" for t in token_times[:5]]
        print(f"First 5 tokens latency:       {', '.join(first_5)}")
        if len(token_times) > 1:
            steady_times = token_times[1:]
            steady_avg_ms = (sum(steady_times) / len(steady_times)) * 1000
            steady_tok_per_sec = len(steady_times) / sum(steady_times)
            print(f"Steady-State (Excl. Token 1): {steady_tok_per_sec:.2f} tok/s (avg {steady_avg_ms:.1f}ms)")
    print(f"{'='*60}\n")
    return {
        "label": label,
        "total_tokens": total_gen_tokens,
        "ttft": ttft,
        "total_time": total_time,
        "decode_throughput": decode_tok_per_sec
    }

if __name__ == "__main__":
    # Run 1: Warm cache on same topic
    r1 = run_bench("Explain what a neural network is in 2 concise sentences.", max_tokens=60, label="Run 1: Warm Cache / Same Topic")
    time.sleep(2)
    # Run 2: Topic shift (forces different expert paths)
    r2 = run_bench("Write a quick Python function that finds the longest palindrome in a string.", max_tokens=80, label="Run 2: Topic Shift (Python Code Generation)")
    time.sleep(2)
    # Run 3: Continued coding context (warm cache code expert paths)
    r3 = run_bench("Now optimize that palindrome algorithm to run in linear time with Manacher's algorithm.", max_tokens=80, label="Run 3: Coding Continuation (Warm Code Experts)")
