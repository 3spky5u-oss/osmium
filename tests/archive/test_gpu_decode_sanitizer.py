#!/usr/bin/env python3
"""Minimal GPU decode test for compute-sanitizer.

Usage:
    cd krasis
    python tests/test_gpu_decode_sanitizer.py
    compute-sanitizer --tool memcheck python tests/test_gpu_decode_sanitizer.py
"""
import os
import sys
import time
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("RUST_LOG", "krasis=info")

from krasis.config import QuantConfig
from krasis.model import KrasisModel


def main():
    model_path = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")

    print("Loading model...")
    t0 = time.time()

    model = KrasisModel(
        model_path=model_path,
        pp_partition=[48],
        num_gpus=1,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=16,
        quant_cfg=QuantConfig(gpu_expert_bits=4, cpu_expert_bits=4),
        layer_group_size=2,
        gpu_prefill_threshold=1,
        force_load=False,
    )
    print(f"Model constructed in {time.time() - t0:.1f}s")

    print("Loading weights...")
    t1 = time.time()
    model.load(gpu_only=True)
    print(f"Weights loaded in {time.time() - t1:.1f}s")

    # Setup GPU decode store
    print("Setting up GPU decode store...")
    model.setup_gpu_decode_store()
    store = model._gpu_decode_store
    print("GPU decode store ready")

    # Warm up
    model.warmup_cuda_runtime([str(d) for d in model.all_devices])

    # Prefill
    print("\n=== Prefill ===")
    prompt_tokens = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        add_generation_prompt=True,
    )
    first_token, prompt_len, kv_overflow = store.rust_prefill_tokens(
        prompt_tokens,
        temperature=0.0,
    )
    if kv_overflow:
        raise RuntimeError(f"Prompt length {prompt_len} exceeded Rust KV capacity")
    stop_ids = [model.cfg.eos_token_id] + list(model.cfg.extra_stop_token_ids)

    print(f"Prefill: prompt_len={prompt_len}, first_token={first_token}")

    # Decode
    print("\n=== GPU Decode (15 tokens) ===")
    tokens = store.gpu_generate_batch(
        first_token=first_token,
        start_position=prompt_len,
        max_tokens=15,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        stop_ids=stop_ids,
        presence_penalty=0.0,
    )
    print(f"Token IDs: {tokens}")

    # Decode to text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    all_tokens = [first_token] + tokens
    text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    print(f"Text: {repr(text)}")

    # Show per-token
    for i, tid in enumerate(all_tokens):
        t = tokenizer.decode([tid])
        print(f"  token[{i}] = {tid:>6} = {repr(t)}")

    model.server_cleanup()
    print("\nDone.")


if __name__ == "__main__":
    main()
