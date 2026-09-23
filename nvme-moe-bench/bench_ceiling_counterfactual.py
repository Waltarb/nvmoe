#!/usr/bin/env python3
"""
Ceiling Counterfactual and Miss-Reduction Response Curve Benchmark
Measures decode throughput, per-token latency, and NVMe exposure across:
1. Artificial storage latency ratios (100%, 75%, 50%, 25%, 0%)
2. Forced host residency percentages (0%, 25%, 50%, 75%, 100%)
"""

import json
import os
import sys
import time
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-flash-next-nvfp4-ftw"
PROMPT = "Count from 1 to 200, one number per line, no other text."
SERVER_LOG = "server_qwen.log"


def cleanup_flags():
    for p in ["/tmp/ft_artificial_ratio.txt", "/tmp/ft_force_pinned_pct.txt", "/tmp/ft_reset_stats.txt"]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def reset_server_stats():
    with open("/tmp/ft_reset_stats.txt", "w") as f:
        f.write("reset\n")
    time.sleep(0.5)


def get_latest_telemetry(num_lines=150):
    if not os.path.exists(SERVER_LOG):
        return {}
    with open(SERVER_LOG, "r", errors="ignore") as f:
        lines = f.readlines()[-num_lines:]
    res = {}
    for l in reversed(lines):
        if "[3-Tier Telemetry]" in l and "nvme_reads" not in res:
            res["telemetry_line"] = l.strip()
            # e.g. NVMe Reads: 26244 (33.1%)
            try:
                parts = l.split("|")
                for p in parts:
                    if "NVMe Reads:" in p:
                        res["nvme_reads"] = int(p.split("NVMe Reads:")[1].split("(")[0].strip())
                    if "In-Memory Hit:" in p:
                        res["in_mem_hit"] = p.split("In-Memory Hit:")[1].strip()
            except Exception:
                pass
        if "[Latency Breakdown]" in l and "nvme_time_s" not in res:
            res["breakdown_line"] = l.strip()
            # e.g. NVMe I/O: 28.901s (69198.0 MB @ 2394 MB/s)
            try:
                p_io = l.split("NVMe I/O:")[1].split("|")[0].strip()
                res["nvme_time_s"] = float(p_io.split("s")[0].strip())
                res["nvme_mb"] = float(p_io.split("(")[1].split("MB")[0].strip())
            except Exception:
                pass
        if "nvme_reads" in res and "nvme_time_s" in res:
            break
    return res


def run_request(max_tokens=150, label="run"):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, stream=True, timeout=600)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    ttft = None
    token_times = []
    generated_text = []
    last = time.perf_counter()
    n = 0

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
                generated_text.append(text)
                if ttft is None:
                    ttft = now - t0
                    last = now
                else:
                    token_times.append(now - last)
                    last = now
        except Exception:
            pass

    total_time = time.perf_counter() - t0
    steady = token_times[5:] if len(token_times) > 5 else token_times
    steady_avg_ms = sum(steady) / len(steady) * 1000.0 if steady else 0.0
    steady_tok_s = 1000.0 / steady_avg_ms if steady_avg_ms > 0 else 0.0

    return {
        "label": label,
        "tokens": n,
        "ttft": ttft or 0.0,
        "total_time": total_time,
        "steady_avg_ms": steady_avg_ms,
        "steady_tok_s": steady_tok_s,
        "min_ms": min(steady) * 1000.0 if steady else 0.0,
        "max_ms": max(steady) * 1000.0 if steady else 0.0,
        "text": "".join(generated_text),
    }


def main():
    cleanup_flags()
    print("=" * 80)
    print("CEILING COUNTERFACTUAL & MISS-REDUCTION BENCHMARK")
    print(f"Model: {MODEL} | Prompt: '{PROMPT}' | temp=0.0")
    print("=" * 80)

    # Phase 0: Warmup and shadow cache population
    print("\n[Phase 0] Warmup Pass: populating shadow RAM and tracing demanded miss set...")
    w1 = run_request(150, "Warmup-1")
    print(f"  Warmup-1: {w1['tokens']} tokens, TTFT={w1['ttft']:.3f}s, steady={w1['steady_avg_ms']:.1f}ms ({w1['steady_tok_s']:.2f} tok/s)")
    time.sleep(1)

    reset_server_stats()
    w2 = run_request(150, "Warmup-2 (Cached Baseline)")
    telem_base = get_latest_telemetry()
    print(f"  Warmup-2: {w2['tokens']} tokens, TTFT={w2['ttft']:.3f}s, steady={w2['steady_avg_ms']:.1f}ms ({w2['steady_tok_s']:.2f} tok/s)")
    print(f"  Baseline Telemetry: NVMe Reads={telem_base.get('nvme_reads')}, NVMe I/O Time={telem_base.get('nvme_time_s')}s, MB={telem_base.get('nvme_mb')} MB")
    golden_text = w2["text"]

    # Phase 1: Artificial Storage Delay Sweep (Simulating Faster Drives)
    print("\n" + "=" * 80)
    print("[Phase 1] Artificial Storage Latency Sweep (Simulating Faster SSDs)")
    print("Ratios: 1.00 (Micron 3400) -> 0.75 (SN850X) -> 0.50 (T700 Gen5) -> 0.25 (Fast Gen5) -> 0.00 (Zero-NVMe)")
    print("=" * 80)

    ratios = [1.00, 0.75, 0.50, 0.25, 0.00]
    phase1_results = []

    for r in ratios:
        with open("/tmp/ft_artificial_ratio.txt", "w") as f:
            f.write(f"{r:.2f}\n")
        reset_server_stats()
        time.sleep(0.5)

        res = run_request(150, f"Storage-Ratio-{r:.2f}")
        telem = get_latest_telemetry()
        match = "MATCH" if res["text"] == golden_text else "MISMATCH!"
        nvme_s = telem.get("nvme_time_s", 0.0)
        nvme_reads = telem.get("nvme_reads", 0)
        mb = telem.get("nvme_mb", 0.0)
        ms_per_tok_nvme = (nvme_s / max(1, res["tokens"])) * 1000.0

        phase1_results.append({
            "ratio": r,
            "tok_s": res["steady_tok_s"],
            "avg_ms": res["steady_avg_ms"],
            "nvme_s": nvme_s,
            "ms_nvme_per_tok": ms_per_tok_nvme,
            "non_nvme_ms": max(0.0, res["steady_avg_ms"] - ms_per_tok_nvme),
            "match": match,
            "nvme_reads": nvme_reads,
        })
        print(f"  Ratio {r:4.2f}: {res['steady_tok_s']:5.2f} tok/s | {res['steady_avg_ms']:5.1f} ms/tok | "
              f"NVMe={ms_per_tok_nvme:5.1f} ms/tok | Non-NVMe={max(0.0, res['steady_avg_ms'] - ms_per_tok_nvme):5.1f} ms/tok | [{match}]")

    cleanup_flags()
    time.sleep(1)

    # Phase 2: Host Residency Sweep (Progressive Miss Elimination)
    print("\n" + "=" * 80)
    print("[Phase 2] Forced Host Residency Sweep (Progressive Miss Elimination)")
    print("Pinned: 0% extra -> 25% -> 50% -> 75% -> 100% (Zero-NVMe)")
    print("=" * 80)

    pcts = [0.0, 25.0, 50.0, 75.0, 100.0]
    phase2_results = []

    for p in pcts:
        with open("/tmp/ft_force_pinned_pct.txt", "w") as f:
            f.write(f"{p:.1f}\n")
        reset_server_stats()
        time.sleep(0.5)

        res = run_request(150, f"Residency-{p:.0f}%")
        telem = get_latest_telemetry()
        match = "MATCH" if res["text"] == golden_text else "MISMATCH!"
        nvme_s = telem.get("nvme_time_s", 0.0)
        nvme_reads = telem.get("nvme_reads", 0)
        ms_per_tok_nvme = (nvme_s / max(1, res["tokens"])) * 1000.0

        phase2_results.append({
            "pct": p,
            "tok_s": res["steady_tok_s"],
            "avg_ms": res["steady_avg_ms"],
            "nvme_s": nvme_s,
            "ms_nvme_per_tok": ms_per_tok_nvme,
            "non_nvme_ms": max(0.0, res["steady_avg_ms"] - ms_per_tok_nvme),
            "match": match,
            "nvme_reads": nvme_reads,
        })
        print(f"  Pinned {p:5.1f}%: {res['steady_tok_s']:5.2f} tok/s | {res['steady_avg_ms']:5.1f} ms/tok | "
              f"NVMe Reads={nvme_reads:5d} | NVMe={ms_per_tok_nvme:5.1f} ms/tok | [{match}]")

    cleanup_flags()

    # Final Summary Table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: COUNTERFACTUAL CEILING AND RESPONSE CURVE")
    print("=" * 80)
    print("\n--- Method 1: Storage Latency Scaling (Simulated Drives) ---")
    print("| Ratio | Effective SSD Class | Decode tok/s | Latency (ms) | Exposed NVMe (ms) | Non-NVMe Floor (ms) | Output Match |")
    print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    ssd_names = {
        1.00: "Micron 3400 (Baseline)",
        0.75: "SN850X / 990 Pro Class",
        0.50: "Crucial T700 Gen5 Class",
        0.25: "Ultra Gen5 (PCIe saturated)",
        0.00: "Zero-NVMe (Instant Storage)",
    }
    for r in phase1_results:
        name = ssd_names.get(r["ratio"], "")
        print(f"| {r['ratio']:5.2f} | {name:27s} | {r['tok_s']:6.2f} | {r['avg_ms']:6.1f} ms | {r['ms_nvme_per_tok']:6.1f} ms | {r['non_nvme_ms']:6.1f} ms | {r['match']} |")

    print("\n--- Method 2: Forced Host Residency (Miss Elimination) ---")
    print("| Extra Pinned | NVMe Reads/tok | Decode tok/s | Latency (ms) | Exposed NVMe (ms) | Output Match |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for p in phase2_results:
        reads_per_tok = p["nvme_reads"] / 148.0
        print(f"| {p['pct']:5.1f}% | {reads_per_tok:6.1f} | {p['tok_s']:6.2f} | {p['avg_ms']:6.1f} ms | {p['ms_nvme_per_tok']:6.1f} ms | {p['match']} |")

    # Save results as JSON
    out_file = "ceiling_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "phase1_storage_ratio": phase1_results,
            "phase2_host_residency": phase2_results,
        }, f, indent=2)
    print(f"\nSaved raw results to {out_file}")


if __name__ == "__main__":
    main()
