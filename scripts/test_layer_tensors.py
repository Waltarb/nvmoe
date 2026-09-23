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

# Helper to read raw bytes from shards
shard_fps = [open(MODEL_DIR / s["file"], "rb") for s in shards]

def read_tensor_bytes(name):
    t = tensors_by_name[name]
    idx = bisect.bisect_right(shard_offsets, t["global_off"]) - 1
    shard = shards[idx]
    off_in_shard = t["global_off"] - shard["global_off"]
    fp = shard_fps[idx]
    fp.seek(off_in_shard)
    return fp.read(t["nbytes"]), t

print("Checking full attention vs linear attention layers...")
full_attn_interval = tc.get("full_attention_interval", 4)
for il in range(tc["num_hidden_layers"]):
    is_recr = (il + 1) % full_attn_interval != 0
    if not is_recr:
        # full attention layer
        assert f"model.layers.{il}.self_attn.qkv_proj.weight" in tensors_by_name
        assert f"model.layers.{il}.self_attn.o_proj.weight" in tensors_by_name
        assert f"model.layers.{il}.self_attn.index_qk_proj.weight" in tensors_by_name
    else:
        # linear attention layer
        assert f"model.layers.{il}.linear_attn.in_proj.weight" in tensors_by_name
        assert f"model.layers.{il}.linear_attn.out_proj.weight" in tensors_by_name
        assert f"model.layers.{il}.linear_attn.conv1d.weight" in tensors_by_name
        assert f"model.layers.{il}.linear_attn.A_log" in tensors_by_name
        assert f"model.layers.{il}.linear_attn.dt_bias" in tensors_by_name
    # MoE experts
    assert f"gate_up_packed#L{il:05d}" in tensors_by_name
    assert f"down_packed#L{il:05d}" in tensors_by_name
    assert f"gate_up_global#L{il:05d}" in tensors_by_name
    assert f"down_global#L{il:05d}" in tensors_by_name

print("All 48 layers have all expected tensors!")
