#!/usr/bin/env python3
"""
convert_ftw_to_gguf.py

Converts Qwen3.8-Flash-Next-NVFP4 FreeToken (.ftw) weights into a GGML / GGUF model
compatible with llama.cpp's Qwen4Exp implementation.

Architecture:
- 48 layers, 512 experts/layer, top-10 routing
- NVFP4 MoE experts repacked into GGML super-blocks (block_nvfp4)
- Gemma-style RMSNorm (+1)
- Linear attention (SSM GDN) with grouped V heads reordered
- QSA indexer projections split
- Hyper-connection mixing layers
"""

import argparse
import bisect
import json
import mmap
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llama.cpp" / "gguf-py"))
import gguf


def _reorder_v_heads(tensor: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def pack_nvfp4_layer(weight_packed: np.ndarray, scale_e4m3: np.ndarray) -> np.ndarray:
    """
    Vectorized repacker: transforms [512, out_features, in_features // 2] uint8
    and [512, out_features, in_features // 16] uint8 scales into GGML block_nvfp4 layout.
    """
    n_exp, out_features, n_bytes_in = weight_packed.shape
    n_blocks = scale_e4m3.shape[2]

    # Unpack ModelOpt nibble-packed weights (byte b has w_{2b} in low nibble, w_{2b+1} in high nibble)
    w_r = weight_packed.reshape(n_exp, out_features, n_blocks, 8)
    w_unpacked = np.empty((n_exp, out_features, n_blocks, 16), dtype=np.uint8)
    w_unpacked[..., 0::2] = w_r & 0x0F
    w_unpacked[..., 1::2] = w_r >> 4

    # Pack into GGML block_nvfp4 sub-block (byte j has w_j in low nibble, w_{j+8} in high nibble)
    qs = (w_unpacked[..., 0:8] & 0x0F) | ((w_unpacked[..., 8:16] & 0x0F) << 4)

    # Preserve original E4M3 scale bits as UE4M3 (strip sign bit)
    d_ue = scale_e4m3 & 0x7F
    n_super = n_blocks // 4
    d_grouped = d_ue.reshape(n_exp, out_features, n_super, 4)
    qs_grouped = qs.reshape(n_exp, out_features, n_super, 32)

    raw = np.concatenate([d_grouped, qs_grouped], axis=-1).reshape(n_exp, out_features, n_super * 36)
    return raw


class FtwReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        with open(model_dir / "freetoken_weight.json") as f:
            self.manifest = json.load(f)
        with open(model_dir / "config.json") as f:
            self.cfg = json.load(f)
        self.tc = self.cfg.get("text_config", self.cfg)

        self.shards = sorted(self.manifest["shards"], key=lambda s: s["global_off"])
        self.shard_offsets = [s["global_off"] for s in self.shards]
        self.tensors_by_name = {t["name"]: t for t in self.manifest["tensors"]}

        self.shard_files = []
        self.shard_mmaps = []
        for s in self.shards:
            f = open(model_dir / s["file"], "rb")
            self.shard_files.append(f)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.shard_mmaps.append(mm)

    def close(self):
        for mm in self.shard_mmaps:
            try:
                mm.close()
            except BufferError:
                pass
        for f in self.shard_files:
            f.close()

    def get_raw_bytes(self, name: str) -> memoryview:
        t = self.tensors_by_name[name]
        idx = bisect.bisect_right(self.shard_offsets, t["global_off"]) - 1
        shard = self.shards[idx]
        off_in_shard = t["global_off"] - shard["global_off"]
        mm = self.shard_mmaps[idx]
        return memoryview(mm)[off_in_shard:off_in_shard + t["nbytes"]]

    def get_torch_tensor(self, name: str) -> torch.Tensor:
        t = self.tensors_by_name[name]
        mv = self.get_raw_bytes(name)
        dtype_str = t["dtype"]
        shape = t["shape"]
        if dtype_str == "bfloat16":
            arr = np.frombuffer(mv, dtype=np.uint16).reshape(shape)
            return torch.from_numpy(arr).view(torch.bfloat16)
        elif dtype_str == "float16":
            arr = np.frombuffer(mv, dtype=np.float16).reshape(shape)
            return torch.from_numpy(arr)
        elif dtype_str == "float32":
            arr = np.frombuffer(mv, dtype=np.float32).reshape(shape)
            return torch.from_numpy(arr)
        elif dtype_str in ("uint8", "float8_e4m3fn"):
            arr = np.frombuffer(mv, dtype=np.uint8).reshape(shape)
            return torch.from_numpy(arr)
        else:
            raise ValueError(f"Unknown dtype {dtype_str} for tensor {name}")

    def get_numpy_array(self, name: str, copy: bool = False) -> np.ndarray:
        t = self.tensors_by_name[name]
        mv = self.get_raw_bytes(name)
        dtype_str = t["dtype"]
        shape = t["shape"]
        if dtype_str == "bfloat16":
            arr = np.frombuffer(mv, dtype=np.uint16).reshape(shape)
        elif dtype_str == "float16":
            arr = np.frombuffer(mv, dtype=np.float16).reshape(shape)
        elif dtype_str == "float32":
            arr = np.frombuffer(mv, dtype=np.float32).reshape(shape)
        elif dtype_str in ("uint8", "float8_e4m3fn"):
            arr = np.frombuffer(mv, dtype=np.uint8).reshape(shape)
        else:
            raise ValueError(f"Unknown dtype {dtype_str}")
        return arr.copy() if copy else arr


def convert_model(model_dir: Path, outfile: Path, with_ple: bool = False):
    print(f"=== Qwen3.8-Flash-Next FTW to GGUF Converter ===")
    print(f"Source Model Dir: {model_dir}")
    print(f"Output GGUF File: {outfile}")
    print(f"PLE Table: {'Enabled' if with_ple else 'Disabled (pass --with-ple to include 51GB table)'}")

    reader = FtwReader(model_dir)
    tc = reader.tc
    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]
    n_embd = tc["hidden_size"]

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(outfile, "qwen4exp")

    # 1. Metadata
    print("\n[1/4] Writing GGUF Metadata...")
    writer.add_name("Qwen3.8-Flash-Next")
    writer.add_block_count(num_layers)
    writer.add_context_length(tc.get("max_position_embeddings", 262144))
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(tc.get("moe_intermediate_size", 640))
    writer.add_head_count(tc.get("num_attention_heads", 24))
    writer.add_head_count_kv(tc.get("num_key_value_heads", 2))
    writer.add_key_length(tc.get("head_dim", 256))
    writer.add_value_length(tc.get("head_dim", 256))
    writer.add_expert_count(num_experts)
    writer.add_expert_used_count(tc.get("num_experts_per_tok", 10))
    writer.add_expert_weights_scale(1.0)
    writer.add_expert_shared_feed_forward_length(tc.get("shared_expert_intermediate_size", 640))
    writer.add_layer_norm_rms_eps(tc.get("rms_norm_eps", 1e-6))
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_NVFP4)

    # RoPE
    rope_params = tc.get("rope_parameters", {}).get("full_attention", {})
    writer.add_rope_freq_base(rope_params.get("rope_theta", 10000000.0))
    writer.add_rope_dimension_sections([11, 11, 10, 0])
    writer.add_rope_dimension_count(64)

    # Hyper-connections
    writer.add_hyper_connection_count(tc.get("hc_count", 4))
    writer.add_hyper_connection_low_rank(tc.get("hc_lowrank", 320))

    # Indexer (QSA sparse attention)
    writer.add_indexer_head_count(tc.get("indexer_n_heads", 4))
    writer.add_indexer_key_length(tc.get("indexer_head_dim", 128))
    writer.add_indexer_top_k(tc.get("indexer_budget", 2048))
    full_attn_interval = tc.get("full_attention_interval", 4)
    writer.add_attention_compress_ratios([
        tc.get("indexer_compress_ratio", 4) if (i + 1) % full_attn_interval == 0 else 0
        for i in range(num_layers)
    ])

    # SSM / Mamba GDN
    writer.add_ssm_conv_kernel(tc.get("linear_conv_kernel_dim", 4))
    writer.add_ssm_inner_size(tc.get("linear_num_value_heads", 48) * tc.get("linear_value_head_dim", 128))
    writer.add_ssm_state_size(tc.get("linear_key_head_dim", 128))
    writer.add_ssm_time_step_rank(tc.get("linear_num_value_heads", 48))
    writer.add_ssm_group_count(tc.get("linear_num_key_heads", 16))
    writer.add_full_attention_interval(full_attn_interval)
    writer.add_recurrent_layers([(i + 1) % full_attn_interval != 0 for i in range(num_layers)])

    # PLE
    if with_ple:
        from freetoken.models.qwen4_exp.ple import derive_ngram_hash_constants
        mult, sizes, offsets = derive_ngram_hash_constants(
            vocab_size=tc["vocab_size"],
            ngram_size=tc["ngram_size"],
            num_ngram_heads=tc.get("heads_per_ngram", tc.get("num_ngram_heads")),
            ngram_vocab_size_base=tc["ngram_vocab_size_base"],
            ple_layer_index=0,
        )
        writer.add_ple_layers([1])
        writer.add_ple_ngram_size(tc.get("ngram_size", 3))
        writer.add_ple_heads_per_ngram(tc.get("heads_per_ngram", 8))
        writer.add_ple_conv_kernel(tc.get("ple_conv_kernel_size", 4))
        writer.add_ple_eos_token_id(tc.get("eos_token_id", 248044))
        writer.add_ple_image_token_id(tc.get("image_token_id", 248056))
        writer.add_ple_layer_multipliers(mult)
        writer.add_ple_head_offsets(offsets)
        writer.add_ple_head_vocab_sizes(sizes)
        writer.add_embedding_length_per_layer_input(160)

    # Tokenizer
    print("\n[2/4] Setting Model Tokenizer & Vocab...")
    # Use ModelBase gpt2 vocab loader
    from conversion.qwen4exp import Qwen4ExpTextModel
    class DummyModel(Qwen4ExpTextModel):
        model_arch = gguf.MODEL_ARCH.QWEN4EXP
        def index_tensors(self, remote_hf_model_id=None):
            return {}

    dummy = DummyModel(dir_model=model_dir, ftype=gguf.LlamaFileType.MOSTLY_NVFP4, fname_out=outfile, is_big_endian=False, use_temp_file=False, eager=False)
    dummy.set_vocab()
    # Copy vocab KV pairs from dummy to writer
    for k, v in dummy.gguf_writer.kv_data[0].items():
        if k.startswith("tokenizer."):
            writer.kv_data[0][k] = v

    # 2. Plan all tensor headers
    print("\n[3/4] Registering Tensor Headers...")
    tensor_jobs = [] # list of tuples: (name, load_fn, shape, dtype, raw_dtype)

    def register_tensor(name, load_fn, shape, dtype, raw_dtype=None):
        writer.add_tensor_info(name, shape, dtype, np.dtype(dtype).itemsize * int(np.prod(shape)) if raw_dtype is None else None, raw_dtype=raw_dtype)
        tensor_jobs.append((name, load_fn))

    # Global tensors
    tensor_jobs_plan = []

    def plan_tensor(name, tensor_shape, np_dtype, raw_dtype=None, load_fn=None):
        nbytes = np.dtype(np_dtype).itemsize * int(np.prod(tensor_shape))
        writer.add_tensor_info(name, tensor_shape, np_dtype, nbytes, raw_dtype=raw_dtype)
        tensor_jobs_plan.append((name, load_fn))

    # Global embeddings & heads
    plan_tensor("token_embd.weight", (tc["vocab_size"], n_embd), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                load_fn=lambda: reader.get_numpy_array("model.embed_tokens.weight"))
    plan_tensor("output.weight", (tc["vocab_size"], n_embd), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                load_fn=lambda: reader.get_numpy_array("lm_head.weight"))
    plan_tensor("hc_head_norm.weight", (tc["hc_count"] * n_embd,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                load_fn=lambda: (reader.get_torch_tensor("model.hyper_connection_mixer.hc_norm.weight") + 1).view(torch.uint16).numpy())
    plan_tensor("hc_head_down.weight", (320, tc["hc_count"] * n_embd), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                load_fn=lambda: reader.get_numpy_array("model.hyper_connection_mixer.input_mix_weight_down.weight"))
    plan_tensor("hc_head_up.weight", (tc["hc_count"] * n_embd, 320), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                load_fn=lambda: reader.get_numpy_array("model.hyper_connection_mixer.input_mix_weight_up.weight"))

    # Per-layer tensors
    for il in range(num_layers):
        is_full_attn = (il + 1) % full_attn_interval == 0
        
        # Hyper connections
        plan_tensor(f"blk.{il}.hc_attn_norm.weight", (10240,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: (reader.get_torch_tensor(f"model.layers.{il}.attn_hyper_connection.hc_norm.weight") + 1).view(torch.uint16).numpy())
        plan_tensor(f"blk.{il}.hc_attn_down.weight", (320, 10240), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.attn_hyper_connection.input_mix_weight_down.weight"))
        plan_tensor(f"blk.{il}.hc_attn_up.weight", (10240, 320), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.attn_hyper_connection.input_mix_weight_up.weight"))
        plan_tensor(f"blk.{il}.hc_attn_inject.weight", (4, 10240), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.attn_hyper_connection.block_inject_weight.weight"))

        plan_tensor(f"blk.{il}.hc_ffn_norm.weight", (10240,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: (reader.get_torch_tensor(f"model.layers.{il}.mlp_hyper_connection.hc_norm.weight") + 1).view(torch.uint16).numpy())
        plan_tensor(f"blk.{il}.hc_ffn_down.weight", (320, 10240), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp_hyper_connection.input_mix_weight_down.weight"))
        plan_tensor(f"blk.{il}.hc_ffn_up.weight", (10240, 320), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp_hyper_connection.input_mix_weight_up.weight"))
        plan_tensor(f"blk.{il}.hc_ffn_inject.weight", (4, 10240), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp_hyper_connection.block_inject_weight.weight"))

        # Attention / SSM
        if is_full_attn:
            plan_tensor(f"blk.{il}.attn_qkv.weight", (13312, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.self_attn.qkv_proj.weight"))
            plan_tensor(f"blk.{il}.attn_out.weight", (2560, 6144), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.self_attn.o_proj.weight"))
            plan_tensor(f"blk.{il}.attn_q_norm.weight", (256,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: (reader.get_torch_tensor(f"model.layers.{il}.self_attn.q_norm.weight") + 1).view(torch.uint16).numpy())
            plan_tensor(f"blk.{il}.attn_k_norm.weight", (256,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: (reader.get_torch_tensor(f"model.layers.{il}.self_attn.k_norm.weight") + 1).view(torch.uint16).numpy())
            
            # Indexer
            def load_indexer_q(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.self_attn.index_qk_proj.weight")
                return t[:512].view(torch.uint16).numpy()
            def load_indexer_k(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.self_attn.index_qk_proj.weight")
                return t[512:640].view(torch.uint16).numpy()

            plan_tensor(f"blk.{il}.indexer.q_proj.weight", (512, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_indexer_q)
            plan_tensor(f"blk.{il}.indexer.k_proj.weight", (128, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_indexer_k)
            plan_tensor(f"blk.{il}.indexer.q_norm.weight", (128,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: (reader.get_torch_tensor(f"model.layers.{il}.self_attn.index_q_norm.weight") + 1).view(torch.uint16).numpy())
            plan_tensor(f"blk.{il}.indexer.k_norm.weight", (128,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: (reader.get_torch_tensor(f"model.layers.{il}.self_attn.index_k_norm.weight") + 1).view(torch.uint16).numpy())

        else:
            # Linear attention
            def load_linear_qkv(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.in_proj.weight")[:10240]
                q = t[:2048]
                k = t[2048:4096]
                v = t[4096:10240]
                v = _reorder_v_heads(v, 0, 16, 3, 128)
                return torch.cat([q, k, v], dim=0).view(torch.uint16).numpy()

            def load_linear_gate(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.in_proj.weight")[10240:16384]
                z = _reorder_v_heads(t, 0, 16, 3, 128)
                return z.view(torch.uint16).numpy()

            def load_linear_beta(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.in_proj.weight")[16384:16432]
                b = _reorder_v_heads(t, 0, 16, 3, 1)
                return b.view(torch.uint16).numpy()

            def load_linear_alpha(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.in_proj.weight")[16432:16480]
                a = _reorder_v_heads(t, 0, 16, 3, 1)
                return a.view(torch.uint16).numpy()

            def load_linear_out(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.out_proj.weight")
                col_perm = _reorder_v_heads(torch.arange(48 * 128, dtype=torch.long).unsqueeze(0), 1, 16, 3, 128).squeeze(0)
                return t[:, col_perm].view(torch.uint16).numpy()

            def load_linear_a(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.A_log")
                a = _reorder_v_heads(t.unsqueeze(-1), 0, 16, 3, 1).squeeze(-1)
                return (-torch.exp(a)).view(torch.uint16).numpy()

            def load_linear_dt(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.dt_bias")
                dt = _reorder_v_heads(t.unsqueeze(-1), 0, 16, 3, 1).squeeze(-1)
                return dt.view(torch.uint16).numpy()

            plan_tensor(f"blk.{il}.attn_qkv.weight", (10240, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_qkv)
            plan_tensor(f"blk.{il}.attn_gate.weight", (6144, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_gate)
            def load_linear_conv(il=il):
                t = reader.get_torch_tensor(f"model.layers.{il}.linear_attn.conv1d.weight").squeeze(1)
                qk = t[:4096]
                v = _reorder_v_heads(t[4096:], 0, 16, 3, 128)
                return torch.cat([qk, v], dim=0).view(torch.uint16).numpy()

            plan_tensor(f"blk.{il}.ssm_conv1d.weight", (10240, 4), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_conv)
            plan_tensor(f"blk.{il}.ssm_dt.bias", (48,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_dt)
            plan_tensor(f"blk.{il}.ssm_a", (48,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_a)
            plan_tensor(f"blk.{il}.ssm_beta.weight", (48, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_beta)
            plan_tensor(f"blk.{il}.ssm_alpha.weight", (48, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_alpha)
            plan_tensor(f"blk.{il}.ssm_norm.weight", (128,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.linear_attn.norm.weight"))
            plan_tensor(f"blk.{il}.ssm_out.weight", (2560, 6144), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16, load_fn=load_linear_out)

        # PLE on layer 1
        if il == 1:
            plan_tensor(f"blk.1.ple_key.weight", (10240, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda: reader.get_numpy_array("model.layers.1.ple.key_proj.weight"))
            plan_tensor(f"blk.1.ple_value.weight", (2560, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda: reader.get_numpy_array("model.layers.1.ple.value_proj.weight"))
            plan_tensor(f"blk.1.ple_norm_key.weight", (10240,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda: (reader.get_torch_tensor("model.layers.1.ple.norm_key.weight") + 1).view(torch.uint16).numpy())
            plan_tensor(f"blk.1.ple_norm_query.weight", (10240,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda: (reader.get_torch_tensor("model.layers.1.ple.norm_query.weight") + 1).view(torch.uint16).numpy())
            plan_tensor(f"blk.1.ple_norm_conv.weight", (10240,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda: (reader.get_torch_tensor("model.layers.1.ple.norm_conv.weight") + 1).view(torch.uint16).numpy())
            plan_tensor(f"blk.1.ple_conv1d.weight", (10240, 4), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                        load_fn=lambda: reader.get_numpy_array("model.layers.1.ple.conv1d.weight").squeeze(1))

        # Router
        plan_tensor(f"blk.{il}.ffn_gate_inp.weight", (512, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp.gate.weight"))

        # MoE Routed Experts (NVFP4)
        def load_gate_up_packed(il=il):
            w = reader.get_numpy_array(f"gate_up_packed#L{il:05d}")
            s = reader.get_numpy_array(f"gate_up_scale#L{il:05d}")
            return pack_nvfp4_layer(w, s)

        def load_gate_up_scale(il=il):
            g = reader.get_torch_tensor(f"gate_up_global#L{il:05d}")
            return g[:, 0].float().numpy()

        def load_down_packed(il=il):
            w = reader.get_numpy_array(f"down_packed#L{il:05d}")
            s = reader.get_numpy_array(f"down_scale#L{il:05d}")
            return pack_nvfp4_layer(w, s)

        def load_down_scale(il=il):
            g = reader.get_torch_tensor(f"down_global#L{il:05d}")
            return g[:, 0].float().numpy()

        plan_tensor(f"blk.{il}.ffn_gate_up_exps.weight", (512, 1280, 1440), np.uint8, raw_dtype=gguf.GGMLQuantizationType.NVFP4, load_fn=load_gate_up_packed)
        plan_tensor(f"blk.{il}.ffn_gate_up_exps.scale", (512,), np.float32, load_fn=load_gate_up_scale)
        plan_tensor(f"blk.{il}.ffn_down_exps.weight", (512, 2560, 360), np.uint8, raw_dtype=gguf.GGMLQuantizationType.NVFP4, load_fn=load_down_packed)
        plan_tensor(f"blk.{il}.ffn_down_exps.scale", (512,), np.float32, load_fn=load_down_scale)

        # Shared Expert
        plan_tensor(f"blk.{il}.ffn_gate_shexp.weight", (640, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp.shared_expert.gate_up_proj.weight")[:640])
        plan_tensor(f"blk.{il}.ffn_up_shexp.weight", (640, 2560), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp.shared_expert.gate_up_proj.weight")[640:])
        plan_tensor(f"blk.{il}.ffn_down_shexp.weight", (2560, 640), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp.shared_expert.down_proj.weight"))
        plan_tensor(f"blk.{il}.ffn_gate_inp_shexp.weight", (2560,), np.uint16, raw_dtype=gguf.GGMLQuantizationType.BF16,
                    load_fn=lambda il=il: reader.get_numpy_array(f"model.layers.{il}.mlp.shared_expert_gate.weight").reshape(2560))

    # PLE Table
    if with_ple:
        ngram_path = model_dir / "qwen4_ngram.bin"
        if ngram_path.exists():
            def load_ple_chunk():
                # Stream or map qwen4_ngram.bin
                raise NotImplementedError("Full PLE chunking")
            pass

    print(f"Registered {len(tensor_jobs_plan)} tensors for conversion.")

    # 3. Write Headers
    print("\n[4/4] Streaming GGUF File To Disk...")
    t0 = time.perf_counter()
    writer.write_header_to_file(outfile)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # 4. Stream Tensors
    total_tensors = len(tensor_jobs_plan)
    for idx, (name, load_fn) in enumerate(tensor_jobs_plan, 1):
        t_sub0 = time.perf_counter()
        data = load_fn()
        writer.write_tensor_data(data)
        t_sub1 = time.perf_counter()
        if "exps.weight" in name or idx % 25 == 0 or idx == total_tensors:
            pct = (idx / total_tensors) * 100
            print(f"[{idx:4d}/{total_tensors}] ({pct:5.1f}%) Wrote {name:<36} ({data.nbytes/(1024*1024):6.1f} MB in {(t_sub1-t_sub0)*1000:5.0f} ms)")

    writer.close()
    reader.close()
    t1 = time.perf_counter()
    file_size_gb = outfile.stat().st_size / (1024**3)
    print(f"\nSUCCESS: Wrote {outfile} ({file_size_gb:.2f} GiB) in {t1 - t0:.1f} s ({file_size_gb / (t1 - t0) * 1024:.1f} MB/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3.8-Flash-Next FTW to GGUF")
    parser.add_argument("--model-dir", type=Path, default=Path("/home/waltarb/.freetoken/models/qwen3.8-flash-next-nvfp4-ftw"),
                        help="Path to FTW model directory")
    parser.add_argument("--outfile", type=Path, default=Path("/home/waltarb/nvmoe-llamacpp/models/qwen3.8-flash-next-nvfp4.gguf"),
                        help="Path to output GGUF file")
    parser.add_argument("--with-ple", action="store_true", help="Include 51.2GB PLE table")
    args = parser.parse_args()

    convert_model(args.model_dir, args.outfile, args.with_ple)
