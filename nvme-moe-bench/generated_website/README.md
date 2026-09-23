# Sample output

`index.html` is a single-file, self-contained dashboard written by
**Qwen3.8-Flash-Next-NVFP4-FTW** while served by this engine on the reference system
(RTX 3080 Ti Laptop, 16 GB VRAM / 32 GB RAM — a machine with roughly a quarter of the
memory the model's weights occupy on disk).

It is here as a qualitative counterpart to the throughput tables in
[`performance.md`](../performance.md): those show the engine is *fast enough*, this shows
the model is still fully coherent across a long multi-turn generation while nearly every
expert it routes to is being paged off NVMe on demand. Offloading changes when weights
arrive, not what the model computes — output is bit-identical to running the same
configuration without the cache, which is verified as a correctness gate on every change.

Open it directly in a browser; it has no build step and no external dependencies.
