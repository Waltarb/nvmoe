# nvmoe-bench

Small, token-budgeted webdev bench for slow local models (built to stay usable at ~2 tok/s).
Each task = tiny TS repo + ticket + visible spec tests + hidden tests. Score = hidden tests pass AND `tsc` clean.

## Setup

```fish
cd bench
pnpm install
```

## Run

Any OpenAI-compatible endpoint (nvmoe, llama-server, Ollama `/v1`):

```fish
set -x BENCH_ENDPOINT http://localhost:8080/v1
set -x BENCH_MODEL local
# optional: disable/limit thinking, merged straight into the request body
set -x BENCH_EXTRA_BODY '{"chat_template_kwargs":{"enable_thinking":false}}'

pnpm bench --config qwen120b-iq2xxs              # smoke tier (default)
pnpm bench --config qwen120b-iq2xxs --tier core   # smoke + core
pnpm bench --config qwen27b-q4 --runs 3           # repeat to check variance
pnpm bench --config test --task 02-slugify-feature
pnpm compare                                       # latest run of every config side by side
```

Results: `results/<config>/<timestamp>.jsonl`, one line per task. Work dirs stay in `.work/` for inspection.

## How a task runs

1. Repo is copied to `.work/`; all `src/` + `spec/` files are pasted into the prompt (no exploration turns).
2. The model answers with Aider-style SEARCH/REPLACE blocks; the harness applies them.
3. Spec tests run after every turn; only pass/fail + the first failure (20 lines) is fed back.
4. Stops on: spec green, `DONE`, output-token budget, wall-clock timeout, or max turns.
5. Hidden tests + `tsc` decide the score. The model never sees them.

The system prompt is byte-identical for every task, so prefix caching (if your backend has it) pays for it once.

## Metrics (per config)

| Metric | Meaning |
| --- | --- |
| passRate / solved | Hidden tests all green + tsc clean |
| testsPer1kOut | Hidden tests passed per 1k output tokens: rewards terse solutions |
| medWallSolvedMin | Median wall-clock minutes for solved tasks |
| decodeTokS / ttftS | Median decode speed and time to first token |
| formatErr / applyErr | Replies without valid blocks / SEARCH blocks that didn't match |
| budgetOrTimeout | Tasks killed by the token or time budget (count as fails) |

`usageEstimated` in the JSONL means the server sent no usage data; tokens were estimated as chars/4.

## Budgets

Set per task in `task.json`: `maxTurns`, `outputBudget` (completion tokens across all turns, reasoning included), `timeoutSec`.
The harness warns when a prompt goes over ~4k tokens.

## Adding tasks

```
tasks/<nn-name>/
  task.json   { "tier": "smoke|core|full", "category": "...", "maxTurns": 3, "outputBudget": 1500, "timeoutSec": 1800 }
  prompt.md   ticket-style description
  repo/src/   starting code (keep it to 1-3 small files)
  repo/spec/  visible tests (1-2, the happy path)
  hidden/     scoring tests; import from "../src/..."
```

Before trusting a new task: write a reference solution and confirm the hidden tests pass, and that the untouched repo fails them.

## Smoke tier

| Task | Category | Hidden tests |
| --- | --- | --- |
| 01-paginate-bugfix | bugfix | 7 |
| 02-slugify-feature | feature | 11 |
| 03-validate-signup | api validation | 10 |
| 04-cart-reducer | state: bugfix + feature + immutability | 10 |
| 05-user-name-split | multi-file refactor + tsc | 8 |

## Core tier ("Hard Core SWE Suite")

High-difficulty systems programming tasks designed so frontier models do not saturate at 100/100:

| Task | Category | Description | Hidden tests |
| --- | --- | --- | --- |
| 06-async-batch-scheduler | concurrency | Async request batching with per-item cancellation, timeout, and listener leak prevention | 8 |
| 07-segmented-lru-cache | memory | 2Q / Segmented LRU cache with probation, protected, and ghost buffers + TTL | 6 |
| 08-websocket-frame-codec | networking | RFC 6455 binary WebSocket frame encoder & streaming chunked decoder with XOR masking | 7 |
| 09-mvcc-transaction-kv | database | In-memory MVCC Snapshot Isolation with write conflict detection and vacuuming | 6 |
| 10-raft-state-machine | distributed | Raft consensus state machine: RequestVote, AppendEntries, term rollover, and uncommitted log truncation | 7 |
| 11-expression-ast-optimizer | ast | Algebraic and boolean AST optimizer with De Morgan laws, constant folding, and fixpoint | 6 |
| 12-token-bucket-rate-limiter | distributed | Fractional token bucket rate limiter with burst spike mitigation and clock skew protection | 5 |
| 13-json-schema-validator | serialization | Lightweight JSON Schema validator with cyclic `$ref` (binary tree) and `oneOf` | 5 |
| 14-diff-patch-engine | tooling | Unified diff parser & 3-way merge with cumulative line drift compensation and conflict markers | 5 |
| 15-event-emitter-typecheck | types | Strongly-typed hierarchical event bus with wildcard namespaces and strict `tsc` checks | 6 |

## Benchmark Scoreboard & Hard Core Guide

See [HARDCORE_BENCHMARK.md](./HARDCORE_BENCHMARK.md) for full task specifications, architectural invariants, scoring methodology, and instructions for evaluating external models (Ollama, vLLM, DeepSeek, Claude, GPT-4o, etc.).

Run `pnpm compare` to view live results:

| Config | Tier | Score (0-100) | Hidden Assertions % | Solved | Decode (tok/s) | TTFT (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen38-nvfp4` (NVMoE Local) | core | **80.0** | **98.4%** | **8/10** | 5.37 | 210.2s |
| `gemini-3.8-flash-high` (Frontier Baseline) | core | **70.0** | **91.8%** | 7/10 | 113.0 | 0.4s |
| `qwen38-iq2` (NVMoE Local) | core | **40.0** | **37.7%** | 4/10 | **30.3** | **19.7s** |
| `qwen38-flash-nvfp4` (NVMoE Local) | smoke | **100.0** | **100%** | 5/5 | 6.89 | 62.6s |
| `glm53-flash-iq2xxs` (NVMoE Local) | smoke | **100.0** | **100%** | 5/5 | 1.76 | 263.4s |


