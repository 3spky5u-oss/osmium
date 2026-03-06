#!/usr/bin/env python3
"""Verify Marlin GEMV kernel correctness for a single expert.

Compares GPU kernel output (marlin_gemv_int4) against a CPU reference
implementation that reads the same Marlin-packed weights and performs
the inverse permutation + dequantization + FP32 matmul.

This isolates the kernel from all other MoE logic (routing, SiLU, accumulation).
"""

import sys
import os
import numpy as np

import torch
torch.set_default_dtype(torch.bfloat16)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel
from krasis.tokenizer import Tokenizer
from krasis.kv_cache import SequenceKVState

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")


def bf16_to_f32(u16_list):
    arr = np.array(u16_list, dtype=np.uint16)
    buf = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return buf.float()


def compare(name, cpu_t, gpu_t, n=8):
    cos = torch.nn.functional.cosine_similarity(
        cpu_t.unsqueeze(0), gpu_t.unsqueeze(0)
    ).item()
    diff = (cpu_t - gpu_t).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cpu_norm = cpu_t.norm().item()
    gpu_norm = gpu_t.norm().item()

    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "DIVERGED")
    print(f"  {name}: cos={cos:.6f} max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} "
          f"cpu_norm={cpu_norm:.4f} gpu_norm={gpu_norm:.4f} [{status}]")
    if status != "OK":
        print(f"    CPU[0:{n}]: {cpu_t[:n].tolist()}")
        print(f"    GPU[0:{n}]: {gpu_t[:n].tolist()}")
    return cos


def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    print(f"Loading model on GPU {gpu_id}...")
    quant = QuantConfig(
        gpu_expert_bits=4,
        cpu_expert_bits=4,
        attention="bf16",
        shared_expert="int8",
        dense_mlp="int8",
        lm_head="int8",
    )

    model = KrasisModel(
        model_path=MODEL_PATH,
        pp_partition=[48],
        num_gpus=1,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=16,
        quant_cfg=quant,
        layer_group_size=2,
        gpu_prefill_threshold=1,
        stream_attention=True,
    )
    model.load()
    tokenizer = Tokenizer(MODEL_PATH)

    prompt = "What is 2+2?"
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # Prefill to get meaningful hidden state
    seq_states = [
        SequenceKVState(c, seq_id=0) if c is not None else None
        for c in model.kv_caches
    ]
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()

    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

    with torch.inference_mode():
        logits = model.forward(prompt_tensor, positions, seq_states)
        first_token = logits[-1:, :].argmax(dim=-1).item()

    print(f"First token: {first_token} = '{tokenizer.decode([first_token])}'")

    # Set up GPU decode store
    model.setup_gpu_decode_store()
    model._update_la_state_ptrs()
    model._export_kv_to_rust(seq_states, len(prompt_tokens))

    store = model._gpu_decode_store

    # Upload a known hidden state (embedding of first_token)
    with torch.inference_mode():
        hidden = model.embedding[first_token].unsqueeze(0).to(device)
    hidden_bf16 = hidden[0].cpu().contiguous().view(torch.int16).numpy().astype(np.uint16).tolist()
    store.upload_hidden_bf16(hidden_bf16)

    print(f"\nHidden state uploaded: norm={hidden[0].float().norm().item():.4f}")

    # Test expert 319 (most common from routing) for layer 0
    test_experts = [319, 106, 0, 1]  # mix of common and index-boundary experts
    layer_idx = 0

    for eid in test_experts:
        print(f"\n{'='*60}")
        print(f"EXPERT {eid}, LAYER {layer_idx}")
        print(f"{'='*60}")

        # GPU kernel: w13 GEMV
        gpu_gate_up_u16 = store.test_single_expert_w13(layer_idx, eid)
        gpu_gate_up = bf16_to_f32(gpu_gate_up_u16)

        # CPU reference: same Marlin weights, inverse perm + dequant + FP32 matmul
        cpu_gate_up = torch.tensor(
            store.test_cpu_reference_w13(layer_idx, eid),
            dtype=torch.float32,
        )

        intermediate = len(gpu_gate_up) // 2
        print(f"  Output size: {len(gpu_gate_up)} (2 x {intermediate})")

        # Compare full gate_up
        cos_full = compare("gate_up (full)", cpu_gate_up, gpu_gate_up)

        # Compare gate and up separately
        cpu_gate = cpu_gate_up[:intermediate]
        cpu_up = cpu_gate_up[intermediate:]
        gpu_gate = gpu_gate_up[:intermediate]
        gpu_up = gpu_gate_up[intermediate:]

        cos_gate = compare("gate part", cpu_gate, gpu_gate)
        cos_up = compare("up part", cpu_up, gpu_up)

        # Show first few values
        print(f"\n  CPU gate[0:8]: {cpu_gate[:8].tolist()}")
        print(f"  GPU gate[0:8]: {gpu_gate[:8].tolist()}")
        print(f"  CPU up[0:8]:   {cpu_up[:8].tolist()}")
        print(f"  GPU up[0:8]:   {gpu_up[:8].tolist()}")

        if cos_full < 0.99:
            # Find worst element
            diff = (cpu_gate_up - gpu_gate_up).abs()
            worst_idx = diff.argmax().item()
            print(f"\n  Worst element: idx={worst_idx} cpu={cpu_gate_up[worst_idx]:.6f} "
                  f"gpu={gpu_gate_up[worst_idx]:.6f} diff={diff[worst_idx]:.6f}")

    # Cleanup
    for s in seq_states:
        if s is not None:
            s.free()

    print("\nDone.")


if __name__ == "__main__":
    main()
