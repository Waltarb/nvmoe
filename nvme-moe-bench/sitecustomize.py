"""Global FreeToken customization hook for nvmoe (3-Tier NVMe MoE Cache).

Enables the 3-tier NVMe expert cache for models when FREETOKEN_NVME_TIER=1 is set.
"""
from __future__ import annotations

import os
import sys

if os.getenv("FREETOKEN_NVME_TIER", "0") in ("1", "true", "yes", "on"):
    try:
        repo_dir = os.getenv("NVMOE_DIR", os.getenv("NVME_MOE_BENCH_DIR"))
        if repo_dir and repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        # Fallback: check current directory if nvme_offload_cache is present
        cwd = os.getcwd()
        if os.path.isfile(os.path.join(cwd, "nvme_offload_cache.py")) and cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Fallback: check script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isfile(os.path.join(script_dir, "nvme_offload_cache.py")) and script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        from freetoken.models.glm5_next.model import Glm5NextForCausalLM
        from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
        from nvme_offload_cache import install_glm5_pruning_patch, make_glm5_offload_cache

        Glm5NextForCausalLM.make_offload_moe_cache = staticmethod(make_glm5_offload_cache)
        Qwen4ExpForCausalLM.make_offload_moe_cache = staticmethod(make_glm5_offload_cache)
        install_glm5_pruning_patch()

        if os.getenv("FREETOKEN_DISABLE_PLE", "0") == "1":
            def dummy_load_host_tables(self, engine_config):
                from freetoken.models.qwen4_exp.ple import ZeroTable, derive_ngram_hash_constants
                import torch
                for ple in self.model.ple_layers:
                    args = ple.args
                    mult, sizes, offsets = derive_ngram_hash_constants(
                        vocab_size=self._config.vocab_size,
                        ngram_size=args.ngram_size,
                        num_ngram_heads=args.num_ngram_heads,
                        ngram_vocab_size_base=args.ngram_vocab_size_base,
                        ple_layer_index=ple.ple_index,
                    )
                    emb = ple.ple_embedding
                    emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                    emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                    emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                    emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
                from freetoken.utils import init_logger
                init_logger(__name__).info("[ABLATION] FREETOKEN_DISABLE_PLE=1: attached ZeroTable to PLE layers")
                return 0
            Qwen4ExpForCausalLM.load_host_tables = dummy_load_host_tables

        from freetoken.utils import init_logger
        init_logger(__name__).info("FREETOKEN_NVME_TIER: registered make_offload_moe_cache and GLM-5 pruning patch")
    except Exception as exc:
        sys.stderr.write(f"Warning: Failed to install FREETOKEN_NVME_TIER hook: {exc}\n")
