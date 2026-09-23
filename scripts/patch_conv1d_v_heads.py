#!/usr/bin/env python3
"""
Patch blk.{il}.ssm_conv1d.weight in models/qwen3.8-flash-next-nvfp4.gguf to reorder
the V channel portion from grouped to tiled order, matching _LinearAttentionVReorderBase.
"""

import mmap
import sys
from pathlib import Path
import numpy as np
import torch

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

def main():
    gguf_path = Path("models/qwen3.8-flash-next-nvfp4.gguf")
    print(f"Reading {gguf_path}...")
    reader = gguf.GGUFReader(str(gguf_path))

    # Find all ssm_conv1d tensors
    conv_tensors = {}
    for t in reader.tensors:
        if t.name.endswith(".ssm_conv1d.weight"):
            conv_tensors[t.name] = (t.data_offset, t.n_bytes)

    print(f"Found {len(conv_tensors)} ssm_conv1d tensors.")
    assert len(conv_tensors) == 36, f"Expected 36 linear attention layers, got {len(conv_tensors)}"

    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = 128
    head_v_dim = 128
    num_v_per_k = num_v_heads // num_k_heads  # 3
    qk_channels = head_k_dim * num_k_heads * 2 # 4096
    v_channels = head_v_dim * num_v_heads     # 6144

    with open(gguf_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        for name, (data_offset, n_bytes) in sorted(conv_tensors.items()):
            # Read 10240 x 4 uint16 array
            arr = np.frombuffer(mm, dtype=np.uint16, count=10240 * 4, offset=data_offset).reshape(10240, 4)
            v_part = arr[qk_channels:]
            
            # Reorder V heads
            v_torch = torch.from_numpy(v_part.copy())
            v_reordered = _reorder_v_heads(v_torch, 0, num_k_heads, num_v_per_k, head_v_dim)
            
            # Write back in-place into mmap buffer
            v_bytes = v_reordered.numpy().tobytes()
            v_offset = data_offset + qk_channels * 4 * 2 # 4096 * 8 bytes
            mm[v_offset : v_offset + len(v_bytes)] = v_bytes
            print(f"Patched {name} at offset {v_offset}")
        mm.flush()
        mm.close()

    print("All 36 ssm_conv1d tensors successfully patched in-place!")

if __name__ == "__main__":
    main()
