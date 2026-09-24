# Hard Core SWE Benchmark: Architecture, Tasks & External Model Guide

The **Hard Core SWE Suite** is a high-difficulty, fast-running agentic coding benchmark designed to evaluate reasoning, systems programming, and compiler invariant fidelity without saturating frontier models (e.g. Gemini 3.8 Flash scores **70/100**, not 100/100), while executing in under **10–15 minutes** rather than hours.

---

## 1. Benchmark Task Catalog (Core Tier)

All 10 core tasks (`tasks/06` through `tasks/15`) feature multi-file TypeScript architectures, hidden edge-case suites, and strict `tsc` compiler invariants.

| # | Task ID | Domain | Challenge & Invariants | Visible Tests | Hidden Tests | Strict `tsc` Check |
|---|---|---|---|:---:|:---:|:---:|
| **06** | `06-async-batch-scheduler` | Concurrency | Coalescing batch scheduler with per-item timeout, cancellation AbortSignals, and listener leak prevention | 4 | 8 | Yes |
| **07** | `07-segmented-lru-cache` | Memory Systems | 2Q / Segmented LRU cache with probation, protected, and ghost eviction buffers + TTL eviction | 4 | 6 | Yes |
| **08** | `08-websocket-frame-codec` | Networking | RFC 6455 binary frame streaming encoder & chunked decoder with client XOR masking | 4 | 7 | Yes |
| **09** | `09-mvcc-transaction-kv` | Database Systems | In-memory MVCC Snapshot Isolation with first-committer-wins write conflict detection and garbage vacuuming | 4 | 6 | Yes |
| **10** | `10-raft-state-machine` | Distributed Systems | Raft consensus state machine: election safety, term rollover, uncommitted log repair & truncation | 4 | 7 | Yes |
| **11** | `11-expression-ast-optimizer` | Compilers & ASTs | Boolean & arithmetic expression tree simplifier: De Morgan laws, constant folding, structural equality | 4 | 6 | Yes |
| **12** | `12-token-bucket-rate-limiter` | Distributed Rate Limiting | Fractional token replenishment with burst caps, atomic consumption, and wall-clock skew protection | 4 | 5 | Yes |
| **13** | `13-json-schema-validator` | Type Systems & Schema | JSON Schema engine handling cyclic/recursive `$ref` graphs (binary tree definitions) and `oneOf` | 4 | 5 | Yes |
| **14** | `14-diff-patch-engine` | Developer Tooling | Unified diff parser & 3-way merge engine with cumulative line drift adjustment and conflict markers | 4 | 5 | Yes |
| **15** | `15-event-emitter-typecheck` | Advanced Types & Events | Strongly-typed hierarchical event bus with wildcard pattern matching and strict TypeScript generics | 4 | 6 | Yes |

---

## 2. Scoring Methodology & Rubric

A task is strictly graded as **PASS (1)** or **FAIL (0)**:

$$\text{Task Solved} \iff (\text{Hidden Tests Passed} == \text{Total Hidden Tests}) \land (\text{tsc Typecheck Exit Code} == 0)$$

### Metrics Reported:
1. **`score100`**: Solved tasks / Total tasks $\times 100$.
2. **`hiddenPct`**: Total hidden assertions passed across all tasks / Total hidden assertions.
3. **`passRate`**: Ratio of successful task runs.
4. **`decodeTokS`**: Real-time decode throughput (tokens/second) during generation.
5. **`ttftS`**: Time-to-first-token (prefill latency in seconds).
6. **`testsPer1kOut`**: Token efficiency metric ($\frac{\text{Hidden Tests Passed}}{\text{Total Completion Tokens}} \times 1000$).
7. **`medWallSolvedMin`**: Median wall clock time in minutes for solved tasks.

---

## 3. How to Run External Models Against the Benchmark

The harness (`bench/harness/`) natively supports any **OpenAI-compatible endpoint** (including Ollama, vLLM, llama-server, OpenRouter, Together AI, Groq, DeepSeek, and local NVMoE).

### Prerequisites
```bash
cd bench
pnpm install
```

### Option A: Local NVMoE Engine (Qwen 3.8 IQ2 / NVFP4 / GLM-5.3)
1. Start the NVMoE local server:
   ```bash
   # Example: Qwen 3.8 Flash IQ2
   GGML_CUDA_DISABLE_GRAPHS=1 \
   NVMOE_CACHE_SIZE=108 NVMOE_GPU_PINNED_EXPERTS=64 \
   NVMOE_HOST_CACHE_SIZE=168 NVMOE_PINNED_EXPERTS=120 \
   NVMOE_PRUNE_NVME_THRESH=0.45 NVMOE_PRUNE_HOST_THRESH=0.04 \
   NVMOE_PRUNE_MIN_KEEP=2 NVMOE_PRUNE_MIN_MASS=0.45 \
   NVMOE_FREQ_PATH=../models/freq_qwen38.bin \
   ../moe_cache_probe --server --port 8080 \
     -m ../models/qwen-3.8-flash-unsloth/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
     -c 4096 -b 2048 --temp 0
   ```

2. Run the Hard Core benchmark:
   ```bash
   BENCH_ENDPOINT=http://localhost:8080/v1 \
   BENCH_MODEL=local \
   pnpm bench --config qwen38-iq2 --tier core
   ```

---

### Option B: Any OpenAI-Compatible Provider (OpenRouter, Groq, DeepSeek, Together, vLLM)
Set the standard OpenAI endpoint environment variables:

```bash
# Example: DeepSeek-V3 / R1 via OpenRouter
BENCH_ENDPOINT=https://openrouter.ai/api/v1 \
BENCH_API_KEY=your_api_key_here \
BENCH_MODEL="deepseek/deepseek-chat" \
pnpm bench --config deepseek-v3 --tier core

# Example: Groq (Llama-3.3-70B)
BENCH_ENDPOINT=https://api.groq.com/openai/v1 \
BENCH_API_KEY=your_groq_key \
BENCH_MODEL="llama-3.3-70b-versatile" \
pnpm bench --config groq-llama33-70b --tier core

# Example: Local vLLM instance
BENCH_ENDPOINT=http://localhost:8000/v1 \
BENCH_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct" \
pnpm bench --config vllm-qwen32b --tier core
```

---

### Option C: Evaluating Gemini Frontier Models (Native SDK)
To evaluate Google Gemini models directly via `@google/genai`:
```bash
GEMINI_API_KEY="your-gemini-key" \
tsx harness/eval-gemini-flash.ts
```
The script runs the complete 10-task core suite, records token usage, latency, and full traces to `results/gemini-3.8-flash-high/baseline.jsonl`.

---

### Option D: Running Specific Tasks or Smoke Tier
```bash
# Run a single task for quick iteration
pnpm bench --config my-model --task 06-async-batch-scheduler

# Run only the smoke tier (Tasks 01 to 05)
pnpm bench --config my-model --tier smoke

# Run both smoke and core tiers
pnpm bench --config my-model --tier smoke+core
```

---

## 4. Comparing Results Side-by-Side

After running any benchmark, execute:
```bash
pnpm compare
```
This automatically scans all `.jsonl` files in `bench/results/*` and prints an aggregated comparison table.

---

## 5. Agent Output Format & Protocol

The harness communicates with models using a strict, zero-ambiguity SEARCH/REPLACE editing protocol:

```markdown
src/scheduler.ts
<<<<<<< SEARCH
export class AsyncBatchScheduler {
  // original code
}
=======
export class AsyncBatchScheduler {
  // replacement code
}
>>>>>>> REPLACE
```

### Protocol Invariants:
1. **Exact Matching**: The `SEARCH` chunk must match the existing file verbatim, including indentation.
2. **Whole File Replacement / Creation**: Leaving `SEARCH` empty creates a new file or replaces the entire file.
3. **Spec Integrity**: Edits modifying `spec/` or `hidden/` are rejected automatically by the harness to prevent test tampering.
4. **Completion Signal**: Writing `DONE` on its own line ends the interaction early when the model believes all tests are satisfied.
5. **Feedback Loop**: When an edit fails to make tests green, Vitest output and compiler errors are fed back into the conversation for up to `maxTurns` (default: 3).

---

## 6. Comparative Results & Benchmark Analysis

| Metric | Qwen 3.8 Flash NVFP4 | Gemini Flash 3.8 (High) | Qwen 3.8 Flash IQ2 | GLM-5.3-Flash IQ2_XXS |
|---|---|---|---|---|
| **Tier Evaluated** | **Core (Tasks 06–15)** | **Core (Tasks 06–15)** | **Core (Tasks 06–15)** | **Smoke (Tasks 01–05)** |
| **Score (0–100)** | **80.0** | **70.0** | **40.0** | **100.0** |
| **Tasks Solved (Turn 1)** | **8/10 (80%)** | **7/10 (70%)** | **4/10 (40%)** | **5/5 (100%)** |
| **Hidden Assertion Rate** | **98.4%** (60/61) | **91.8%** (56/61) | **37.7%** (23/61) | **100%** (46/46) |
| **Decode Throughput** | **5.37 tok/s** | 113.0 tok/s | **30.30 tok/s** | 1.76 – 8.93 tok/s |
| **Prompt TTFT** | 210.2s (3.5 min) | 0.4s | **19.7s** | 1.4s – 263.4s |
| **Median Solved Time** | 9.1 min | **0.1 min** | **1.6 min** | 8.1 min |
| **Token Efficiency** | 3.51 tests / 1k out | 6.48 tests / 1k out | 0.97 tests / 1k out | **20.03 tests / 1k out** |
| **VRAM Consumption** | **12.47 GiB** (< 14 GiB) | Cloud API | **12.64 GiB** (< 14 GiB)| **12.13 GiB** (< 14 GiB) |
| **Host RAM Consumption** | **14.5 GiB** (< 20 GiB) | Cloud API | **15.1 GiB** (< 20 GiB)| **4.1 GiB** (< 20 GiB) |

### Task-by-Task Breakdown (Hard Core Suite):

| Task # | Task ID | Gemini Flash 3.8 (High) | Qwen 3.8 Flash NVFP4 (Local) | Qwen 3.8 Flash IQ2 (Local) | Key Domain / Systems Invariant |
|---|---|:---:|:---:|:---:|---|
| **06** | `06-async-batch-scheduler` | **PASS** (8/8) | **PASS** (8/8) | FAIL (0/8) | Async request coalescing, listener leaks, AbortSignal |
| **07** | `07-segmented-lru-cache` | **PASS** (6/6) | **PASS** (6/6) | **PASS** (6/6) | 2Q / Segmented LRU cache with probation/protected/ghost |
| **08** | `08-websocket-frame-codec` | **PASS** (7/7) | **PASS** (7/7) | FAIL (0/7) | RFC 6455 binary frame streaming encoder & chunked decoder |
| **09** | `09-mvcc-transaction-kv` | **PASS** (6/6) | FAIL (6/6, tsc) | **PASS** (6/6) | MVCC snapshot isolation, first-committer conflict detection |
| **10** | `10-raft-state-machine` | **PASS** (7/7) | **PASS** (7/7) | FAIL (0/7) | Raft election safety, term rollover, uncommitted log repair |
| **11** | `11-expression-ast-optimizer` | **PASS** (6/6) | **PASS** (6/6) | FAIL (0/6) | Boolean & arithmetic AST simplification, De Morgan laws |
| **12** | `12-token-bucket-rate-limiter`| **PASS** (5/5) | **PASS** (5/5) | **PASS** (5/5) | Fractional token bucket with burst mitigation & clock skew |
| **13** | `13-json-schema-validator` | FAIL (5/5, TS2304) | FAIL (4/5) | FAIL (0/5) | Recursive/cyclic `$ref` binary tree schema validation |
| **14** | `14-diff-patch-engine` | FAIL (0/5, regex) | **PASS** (5/5) | FAIL (0/5) | Unified diff 3-way merge with cumulative line drift |
| **15** | `15-event-emitter-typecheck` | FAIL (6/6, TS2304) | **PASS** (6/6) | **PASS** (6/6) | Strictly-typed hierarchical event bus with wildcard patterns |
| **Total** | | **70.0% (7/10)** | **80.0% (8/10)** | **40.0% (4/10)** | *NVFP4 takes #1 spot with 8/10 solved & 98.4% hidden pass rate!* |

### Why Gemini Flash 3.8 Scores 70/100 (Non-Saturating Benchmark):
1. **Tasks 06 to 12 (Solved Cleanly)**: Concurrency, segmented LRU eviction, binary WebSocket frames, MVCC snapshot isolation, Raft consensus, AST optimization, and token bucket arithmetic were synthesized flawlessly.
2. **Tasks 13 & 15 (Compiler Type Invariants)**: On `13-json-schema-validator` and `15-event-emitter-typecheck`, the generated code passed 100% of hidden test assertions, but missed type imports (`import type { ValidationError }` and `import type { EventContext }`), failing the strict compiler check (`tsc -p tsconfig.json`).
3. **Task 14 (Unified Diff Regex Escaping)**: In `14-diff-patch-engine`, regex patterns for unified diff hunk headers (`@@ -l,s +l,s @@`) encountered escaping nuances, demonstrating real-world software engineering friction.

---

## 7. Hardware Safety Invariants (Local Models)

When evaluating models locally with NVMoE on laptop/workstation hardware:
- **VRAM Hard Cap (< 14 GiB)**: Always keep model buffer + KV cache strictly under 14 GiB to avoid CUDA driver resets.
- **Host RAM Hard Cap (< 20 GiB)**: DDR5 pinned RAM cache must stay under 20 GiB to prevent triggering the Linux OOM Killer.
- **Always Explicit Context**: Pass `-c 4096 -b 2048` (or `-c 512`) to prevent llama.cpp from preallocating 30+ GiB for default 256k-token contexts.
- **Always Set `GGML_CUDA_DISABLE_GRAPHS=1`**: Disables CUDA graph capture conflicts with dynamic H2D weight updates.

