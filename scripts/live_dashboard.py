#!/usr/bin/env python3
import http.server
import socketserver
import json
import subprocess
import time
import re
import os
import sys
from pathlib import Path

PORT = 8085
BENCH_DIR = Path(__file__).resolve().parent.parent / "bench"
LIVE_STATUS_FILE = BENCH_DIR / ".live_status.json"
LIVE_STREAM_FILE = BENCH_DIR / ".live_stream.txt"

prev_bytes = 0
prev_time = time.time()
cur_tok_s = 0.0

def get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util": float(parts[0]),
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
            "temp_c": float(parts[3]),
            "power_w": float(parts[4])
        }
    except Exception:
        return {"gpu_util": 0, "vram_used_mb": 0, "vram_total_mb": 16035, "temp_c": 0, "power_w": 0}

def get_socket_stats():
    global prev_bytes, prev_time, cur_tok_s
    try:
        out = subprocess.check_output(["ss", "-ti", "sport = :8080 or dport = :8080"], text=True)
        m = re.search(r"bytes_sent:(\d+)", out)
        if m:
            total_bytes = int(m.group(1))
            now = time.time()
            dt = now - prev_time
            if dt >= 1.0 and prev_bytes > 0 and total_bytes >= prev_bytes:
                dbytes = total_bytes - prev_bytes
                # ~120 bytes per JSON SSE delta chunk on average
                tokens = dbytes / 120.0
                cur_tok_s = round(tokens / dt, 2)
                prev_bytes = total_bytes
                prev_time = now
            elif prev_bytes == 0:
                prev_bytes = total_bytes
                prev_time = now
            est_tokens = int(total_bytes / 120.0)
            return {"active": True, "bytes_sent": total_bytes, "est_tokens": est_tokens, "cur_tok_s": cur_tok_s}
    except Exception:
        pass
    return {"active": False, "bytes_sent": 0, "est_tokens": 0, "cur_tok_s": 0.0}

def get_live_data():
    gpu = get_gpu_stats()
    sock = get_socket_stats()

    live_status = {}
    if LIVE_STATUS_FILE.exists():
        try:
            live_status = json.loads(LIVE_STATUS_FILE.read_text())
        except Exception:
            pass

    stream_text = ""
    if LIVE_STREAM_FILE.exists():
        try:
            txt = LIVE_STREAM_FILE.read_text()
            stream_text = txt[-2000:]
        except Exception:
            pass

    # Read latest summary results
    results_summary = []
    try:
        res_dir = BENCH_DIR / "results"
        if res_dir.exists():
            for cfg in sorted(os.listdir(res_dir)):
                cfg_path = res_dir / cfg
                if cfg_path.is_dir():
                    files = sorted(cfg_path.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
                    if files:
                        lines = [json.loads(l) for l in files[0].read_text().splitlines() if l.strip()]
                        if lines:
                            solved = sum(1 for x in lines if x.get("solved"))
                            total = len(lines)
                            results_summary.append({
                                "config": cfg,
                                "solved": f"{solved}/{total}",
                                "score": round((solved / total) * 100, 1),
                                "last_task": lines[-1].get("task", ""),
                                "file": files[0].name
                            })
    except Exception as e:
        pass

    return {
        "timestamp": time.time(),
        "gpu": gpu,
        "socket": sock,
        "live": live_status,
        "stream_tail": stream_text,
        "results": results_summary
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>NVMoE Real-Time Benchmark Monitor</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @keyframes pulse-fast { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .animate-live { animation: pulse-fast 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-[#0f172a] text-slate-100 min-h-screen p-6 font-sans">
  <div class="max-w-6xl mx-auto space-y-6">
    <!-- Header -->
    <header class="flex items-center justify-between border-b border-slate-800 pb-4">
      <div class="flex items-center space-x-3">
        <div class="w-3.5 h-3.5 rounded-full bg-emerald-500 animate-live"></div>
        <h1 class="text-2xl font-bold tracking-tight text-white">NVMoE Real-Time Benchmark Monitor</h1>
        <span class="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">LIVE SSE</span>
      </div>
      <div class="text-xs text-slate-400 font-mono" id="clock">--:--:--</div>
    </header>

    <!-- Top Metric Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Decode Throughput</div>
        <div class="mt-2 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-cyan-400 font-mono" id="tok-s">0.00</span>
          <span class="text-sm font-semibold text-slate-400">tok/s</span>
        </div>
        <div class="mt-1 text-xs text-slate-500 font-mono" id="stream-rate">Stream active</div>
      </div>

      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Generated Tokens</div>
        <div class="mt-2 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-emerald-400 font-mono" id="est-tokens">0</span>
          <span class="text-sm font-semibold text-slate-400">tokens</span>
        </div>
        <div class="mt-1 text-xs text-slate-500 font-mono" id="bytes-sent">0 KB transfer</div>
      </div>

      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">GPU VRAM & Util</div>
        <div class="mt-2 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-violet-400 font-mono" id="vram-val">0.0</span>
          <span class="text-sm font-semibold text-slate-400">/ 16 GB</span>
        </div>
        <div class="mt-1 text-xs text-slate-500 font-mono" id="gpu-util">Util: 0%</div>
      </div>

      <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-sm">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">GPU Temp & Power</div>
        <div class="mt-2 flex items-baseline space-x-2">
          <span class="text-3xl font-extrabold text-amber-400 font-mono" id="temp-val">0</span>
          <span class="text-sm font-semibold text-slate-400">°C</span>
        </div>
        <div class="mt-1 text-xs text-slate-500 font-mono" id="power-val">Power: 0 W</div>
      </div>
    </div>

    <!-- Active Stream / Live Output Terminal -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      <div class="bg-slate-950 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full bg-red-500/80"></div>
          <div class="w-3 h-3 rounded-full bg-yellow-500/80"></div>
          <div class="w-3 h-3 rounded-full bg-green-500/80"></div>
          <span class="text-xs text-slate-400 font-mono ml-2">Live Output Stream (OpenAI SSE Port 8080)</span>
        </div>
        <span class="text-xs text-slate-500 font-mono" id="model-tag">Model: Active</span>
      </div>
      <pre id="stream-output" class="p-4 text-xs font-mono text-emerald-300 bg-slate-950/80 overflow-y-auto max-h-96 whitespace-pre-wrap break-all leading-relaxed">Waiting for token stream...</pre>
    </div>

    <!-- Results Scoreboard -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider mb-3">Model Scoreboard Overview</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs font-mono">
          <thead class="bg-slate-950/60 text-slate-400 border-b border-slate-800">
            <tr>
              <th class="py-2.5 px-3">Model Configuration</th>
              <th class="py-2.5 px-3">Solved / Total</th>
              <th class="py-2.5 px-3">Score (0-100)</th>
              <th class="py-2.5 px-3">Latest Task</th>
              <th class="py-2.5 px-3">Log File</th>
            </tr>
          </thead>
          <tbody id="results-table" class="divide-y divide-slate-800/60 text-slate-300">
            <tr><td colspan="5" class="py-3 px-3 text-slate-500">Loading results...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    async function update() {
      try {
        const res = await fetch('/api/status');
        const d = await res.json();
        
        document.getElementById('clock').textContent = new Date().toLocaleTimeString();

        // Speed & Tokens
        const tokS = (d.socket.cur_tok_s || (d.live && d.live.tokS) || 0).toFixed(2);
        document.getElementById('tok-s').textContent = tokS;
        document.getElementById('est-tokens').textContent = d.socket.est_tokens || (d.live && d.live.outputTokens) || 0;
        document.getElementById('bytes-sent').textContent = (d.socket.bytes_sent / 1024).toFixed(1) + ' KB transfer';

        // GPU
        const vramGb = (d.gpu.vram_used_mb / 1024).toFixed(1);
        document.getElementById('vram-val').textContent = vramGb;
        document.getElementById('gpu-util').textContent = 'Util: ' + d.gpu.gpu_util + '%';
        document.getElementById('temp-val').textContent = d.gpu.temp_c;
        document.getElementById('power-val').textContent = 'Power: ' + d.gpu.power_w + ' W';

        // Output stream
        if (d.stream_tail) {
          const outEl = document.getElementById('stream-output');
          outEl.textContent = d.stream_tail;
          outEl.scrollTop = outEl.scrollHeight;
        }

        // Scoreboard
        if (d.results && d.results.length) {
          const tbody = document.getElementById('results-table');
          tbody.innerHTML = d.results.map(r => `
            <tr class="hover:bg-slate-800/30">
              <td class="py-2 px-3 font-semibold text-cyan-400">${r.config}</td>
              <td class="py-2 px-3">${r.solved}</td>
              <td class="py-2 px-3 font-bold ${r.score >= 70 ? 'text-emerald-400' : (r.score >= 40 ? 'text-amber-400' : 'text-slate-400')}">${r.score}%</td>
              <td class="py-2 px-3 text-slate-400">${r.last_task}</td>
              <td class="py-2 px-3 text-slate-500">${r.file}</td>
            </tr>
          `).join('');
        }
      } catch (e) {
        console.error(e);
      }
    }
    setInterval(update, 1000);
    update();
  </script>
</body>
</html>
"""

class MonitorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/api/status":
            data = get_live_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_cli():
    print("\033[2J\033[H", end="")
    while True:
        d = get_live_data()
        gpu = d["gpu"]
        sock = d["socket"]
        tok_s = sock.get("cur_tok_s", 0.0)
        vram_gb = gpu["vram_used_mb"] / 1024.0
        
        print("\033[H", end="")
        print("================================================================================")
        print("                 NVMoE REAL-TIME BENCHMARK MONITOR                              ")
        print("================================================================================")
        print(f" TIME: {time.strftime('%H:%M:%S')}  |  PORT: 8080 (SSE)  |  STATUS: {'DECODING' if sock['active'] else 'IDLE'}")
        print("--------------------------------------------------------------------------------")
        print(f" SPEED:       \033[1;36m{tok_s:>5.2f} tok/s\033[0m   |  EST GENERATED TOKENS: \033[1;32m{sock['est_tokens']:>5}\033[0m")
        print(f" VRAM:        \033[1;35m{vram_gb:>5.2f} / 15.6 GiB\033[0m (Hard Cap: < 14 GiB)")
        print(f" GPU UTIL:    \033[1;33m{gpu['gpu_util']:>3.0f}%\033[0m             |  GPU TEMP:  \033[1;31m{gpu['temp_c']:>2.0f}°C\033[0m  |  PWR: {gpu['power_w']:>3.0f}W")
        print("--------------------------------------------------------------------------------")
        if d.get("stream_tail"):
            print(" OUTPUT SNIPPET (tail):")
            lines = d["stream_tail"].splitlines()[-6:]
            for l in lines:
                print("   " + l[:75])
        else:
            print(" OUTPUT: Streaming SSE chunks over localhost:8080...")
        print("================================================================================")
        time.sleep(1.0)

if __name__ == "__main__":
    if "--cli" in sys.argv:
        try:
            run_cli()
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", PORT), MonitorHandler) as httpd:
            print(f"Dashboard server started at http://localhost:{PORT}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
