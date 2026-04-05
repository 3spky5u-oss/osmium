# Osmium

High-performance MoE inference engine for consumer GPUs. Fork of [Krasis](https://github.com/brontoguana/krasis) with WriteCombined DMA staging, SM120 (Blackwell) native kernels, and Python GPU prefill preservation.

Runs **Qwen3.5-122B-A10B at 3800 tok/s prefill and 60 tok/s decode on a single RTX 5090.**

## Benchmarks

### RTX 5090 32GB, PCIe Gen5 x16

Qwen3.5-122B-A10B (122B params, 234 GB BF16 / 56 GB INT4). Single GPU, AWQ attention, FP8 KV cache, WriteCombined DMA.

| KV Cache | Context Capacity | Prefill (tok/s) | Decode (tok/s) | HCS Coverage |
|----------|:----------------:|:---------------:|:--------------:|:------------:|
| 1000 MB  | 85K tokens       | **3,862**       | **60.1**       | 34.3%        |
| 3400 MB  | 256K tokens      | **3,742**       | **59.5**       | 27.9%        |

### What makes this fast

- **WriteCombined DMA staging** (`--wc-alloc`): Expert weights allocated via `cuMemHostAlloc(WRITECOMBINED)` bypass the CPU cache hierarchy. PCIe Gen5 DMA reads hit ~46 GB/s vs ~28 GB/s for regular pinned memory. Per-component layout with incremental heap freeing keeps peak RAM at ~74 GB on a 91 GB system.
- **Python GPU prefill** via sglang's `fused_marlin_moe`: Full INT4 Marlin GEMM prefill at 3800+ tok/s. The upstream Rust prefill path is 10x slower (364 tok/s) due to lack of optimized GEMM.
- **Dual-PTX decode kernels**: SM80 (Ampere/Ada) and SM120 (Blackwell) PTX compiled at build time, selected at runtime based on GPU compute capability.
- **Vectorized INT4 GEMV**: `uint32` loads for 4x fewer iterations and full 128-byte cache line utilization per warp.
- **HCS (Hot Cache Strategy)**: Frequently-accessed MoE experts cached in VRAM, reducing PCIe DMA for hot paths.

## Requirements

- **Linux** (Fedora 43, Ubuntu 24.04+, or WSL2)
- **NVIDIA GPU** with CUDA 12.8+ drivers (SM80+ for Ampere, SM120 for Blackwell)
- **Python 3.13** (3.10+ should work)
- **System RAM**: ~74 GB with `--wc-alloc` for 122B models (91 GB physical + 64 GB NVMe swap recommended)
- **NVMe swap**: Required for WC mode. `sudo fallocate -l 64G /swapfile_osmium && sudo chmod 600 /swapfile_osmium && sudo mkswap /swapfile_osmium && sudo swapon /swapfile_osmium`

## Quick Start

```bash
git clone https://github.com/3spky5u-oss/osmium.git
cd osmium
git checkout v0.1.64-sm120

# Create venv and install dependencies
python3.13 -m venv .venv
.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install "sglang[all]==0.5.9" flashinfer-python==0.6.3

# Build (requires Rust toolchain + CUDA toolkit)
CUDA_HOME=/usr/local/cuda .venv/bin/pip install maturin
CUDA_HOME=/usr/local/cuda .venv/bin/maturin develop --release

# Download a model
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3.5-122B-A10B

# Ensure swap is active
sudo swapon /swapfile_osmium

# Run with benchmark
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m krasis.server \
  --model-path ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/* \
  --benchmark --wc-alloc \
  --gpu-expert-bits 4 --attention-quant awq --kv-dtype fp8_e4m3 --kv-cache-mb 1000
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--wc-alloc` | WriteCombined DMA staging. ~2x decode speed, requires swap. |
| `--gpu-expert-bits 4` | INT4 Marlin experts on GPU (default) |
| `--attention-quant awq` | AWQ attention quantization (frees VRAM for HCS) |
| `--kv-dtype fp8_e4m3` | FP8 KV cache (halves KV memory vs BF16) |
| `--kv-cache-mb N` | KV cache size. 1000 = 85K ctx, 3400 = 256K ctx |
| `--benchmark` | Run prefill + decode benchmark after loading |
| `--hcs` | Hot Cache Strategy (on by default with `--wc-alloc`) |

## Architecture

Osmium is a hybrid Python/Rust/CUDA system:

- **Python orchestration** (`python/krasis/`): Model loading, GPU prefill via sglang's fused Marlin MoE, server (OpenAI-compatible API), launcher TUI
- **Rust core** (`src/`): GPU decode engine with CUDA graphs, MoE expert routing + DMA scheduling, weight loading (safetensors, GGUF, Marlin cache)
- **CUDA kernels** (`src/cuda/`): Decode kernels (RMSNorm, GQA attention, INT4/INT8 GEMV, expert routing), compiled to PTX at build time

MoE expert weights live in system RAM (56 GB for 122B INT4). During decode, the top-K experts per layer are DMA'd to GPU via PCIe. WriteCombined memory makes this transfer ~46 GB/s on Gen5, fast enough for 60 tok/s.

## What Osmium Changes vs Krasis v0.1.64

| Feature | Krasis v0.1.64 | Osmium v0.1 |
|---------|---------------|-------------|
| Decode kernels | SM80 PTX only | Dual SM80 + SM120 PTX |
| Expert DMA | Pinned memory (~28 GB/s) | WriteCombined (~46 GB/s) |
| INT4 GEMV | Scalar loads | Vectorized uint32 loads |
| GGUF loading | Use-after-free bug | Fixed (layer_backings_gpu) |
| GGUF gpu_only | Not forwarded | Proper skip_cpu plumbing |
| Marlin cache load | 2x peak RSS | Per-layer MADV_DONTNEED |
| RAM watchdog | 5% floor, no swap | 0.5% floor, counts SwapFree |

## API

OpenAI-compatible at `http://localhost:8012/v1/chat/completions` with SSE streaming. Works with Cursor, OpenCode, Continue, and any OpenAI SDK client.

```bash
curl http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

## License

SSPL-1.0 (inherited from Krasis)
