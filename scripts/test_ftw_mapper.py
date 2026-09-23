#!/usr/bin/env python3
import json, os, re, bisect, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/waltarb/nvmoe-llamacpp/llama.cpp")
sys.path.insert(0, "/home/waltarb/nvmoe-llamacpp/llama.cpp/gguf-py")
import gguf

MODEL_DIR = Path("/home/waltarb/.freetoken/models/qwen3.8-flash-next-nvfp4-ftw")

with open(MODEL_DIR / "freetoken_weight.json") as f:
    manifest = json.load(f)

with open(MODEL_DIR / "config.json") as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)

shards = sorted(manifest["shards"], key=lambda s: s["global_off"])
shard_offsets = [s["global_off"] for s in shards]
tensors_by_name = {t["name"]: t for t in manifest["tensors"]}

print(f"Loaded config: layers={tc['num_hidden_layers']}, experts={tc['num_experts']}, top_k={tc['num_experts_per_tok']}")
print(f"Found {len(manifest['tensors'])} manifest entries across {len(shards)} shards")
