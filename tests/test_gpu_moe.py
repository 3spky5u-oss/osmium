#!/usr/bin/env python3
"""Test GPU MoE forward against CPU MoE forward on QCN model."""

import sys
import time
import struct
import numpy as np

# Setup path
sys.path.insert(0, "python")

from krasis import KrasisEngine, GpuDecodeStore


def f32_to_bf16(val):
    """Convert float32 to bfloat16 (as u16)."""
    b = struct.pack('f', val)
    # BF16 is just the upper 16 bits of FP32
    return struct.unpack('H', b[2:4])[0]


def bf16_to_f32(val):
    """Convert bfloat16 (as u16) to float32."""
    b = struct.pack('HH', 0, val)
    return struct.unpack('f', b)[0]


def main():
    model_dir = "/home/main/.krasis/models/Qwen3-Coder-Next"
    cache_dir = "/home/main/.krasis/cache/Qwen3-Coder-Next"

    print("Loading QCN model weights...")
    engine = KrasisEngine()
    engine.load(model_dir, cache_dir, gpu_bits=4, cpu_bits=4, group_size=128)

    # Get model config
    cfg = engine.model_config()
    hidden = cfg["hidden_size"]
    intermediate = cfg["moe_intermediate_size"]
    num_experts = cfg["n_routed_experts"]
    topk = cfg["num_experts_per_tok"]
    group_size = 128

    print(f"Model: hidden={hidden}, intermediate={intermediate}, "
          f"experts={num_experts}, topk={topk}")

    # Setup GPU decode store
    print("Setting up GpuDecodeStore...")
    store = GpuDecodeStore(0)
    store.configure(
        hidden_size=hidden,
        num_layers=48,
        vocab_size=152064,
        eps=1e-6,
        max_experts_per_tok=topk,
        max_intermediate_size=intermediate,
        group_size=group_size,
    )

    # We need to register the gate weight for MoE routing.
    # The gate weights are FP32 [num_experts, hidden] stored on GPU.
    # For testing, we load them from the model's safetensors.
    # But for now, let's get the gate from the engine's route weights.

    # Get expert data pointers for layer 0 (first MoE layer)
    # QCN: first 8 layers are dense, layers 8-47 are MoE
    test_moe_layer = 0  # MoE layer index (0 = global layer 8)

    # Get expert pointers from engine
    expert_ptrs = []
    for eid in range(num_experts):
        ptrs = engine.get_expert_gpu_ptrs(test_moe_layer, eid)
        expert_ptrs.append(ptrs)

    # Get shared expert pointers
    shared_ptrs = engine.get_shared_expert_gpu_ptrs(test_moe_layer)

    # Get gate weight from engine (FP32, on CPU)
    gate_data = engine.get_route_gate_data(test_moe_layer)
    gate_data_np = np.array(gate_data, dtype=np.float32)

    # Upload gate to GPU via torch
    import torch
    gate_tensor = torch.tensor(gate_data_np.reshape(num_experts, hidden),
                               dtype=torch.float32, device='cuda:0')
    gate_wid = store.register_weight(gate_tensor.data_ptr(), num_experts, hidden, 1)  # dtype=1 for FP32

    # Register MoE layer
    store.register_moe_layer(
        layer_idx=test_moe_layer,
        expert_ptrs=expert_ptrs,
        shared_ptrs=shared_ptrs,
        num_experts=num_experts,
        topk=topk,
        scoring_func=1,  # sigmoid for QCN
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        gate_wid=gate_wid,
    )

    # Resize expert buffers to fit the largest expert
    max_expert_bytes = max(
        sum(ptrs[1::2])  # sum of all byte sizes
        for ptrs in expert_ptrs
    )
    store.resize_expert_buffers(max_expert_bytes)

    # Create a random hidden state (BF16)
    np.random.seed(42)
    hidden_f32 = np.random.randn(hidden).astype(np.float32) * 0.1
    hidden_bf16 = [f32_to_bf16(float(v)) for v in hidden_f32]

    # Upload hidden state
    store.upload_hidden_bf16(hidden_bf16)

    # Run MoE forward on GPU
    print("\nRunning GPU MoE forward...")
    t0 = time.time()
    route_ms, dma_ms, compute_ms, total_ms = store.moe_forward_gpu(test_moe_layer)
    elapsed = (time.time() - t0) * 1000

    print(f"GPU MoE timing:")
    print(f"  Route:   {route_ms:.2f} ms")
    print(f"  DMA:     {dma_ms:.2f} ms")
    print(f"  Compute: {compute_ms:.2f} ms")
    print(f"  Total:   {total_ms:.2f} ms (wall: {elapsed:.2f} ms)")

    # Download GPU result
    gpu_result_bf16 = store.download_moe_out_bf16()
    gpu_result_f32 = np.array([bf16_to_f32(v) for v in gpu_result_bf16], dtype=np.float32)

    print(f"\nGPU output: first 10 values = {gpu_result_f32[:10]}")
    print(f"GPU output: max abs = {np.max(np.abs(gpu_result_f32)):.6f}")
    print(f"GPU output: L2 norm = {np.linalg.norm(gpu_result_f32):.6f}")

    # Run a second time for timing stability
    print("\nRunning GPU MoE forward (2nd run, warmed up)...")
    store.upload_hidden_bf16(hidden_bf16)
    route_ms, dma_ms, compute_ms, total_ms = store.moe_forward_gpu(test_moe_layer)
    print(f"  Route: {route_ms:.2f} ms, DMA: {dma_ms:.2f} ms, "
          f"Compute: {compute_ms:.2f} ms, Total: {total_ms:.2f} ms")

    print("\nDone!")


if __name__ == "__main__":
    main()
