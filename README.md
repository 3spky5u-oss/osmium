# Osmium

High-performance MoE inference engine for consumer GPUs. Fork of [Krasis](https://github.com/brontoguana/krasis) with WriteCombined DMA staging for +50% decode throughput on PCIe Gen5.

Runs **Qwen3.5-122B-A10B at 57 tok/s decode on a single RTX 5090** (vs 38 tok/s upstream).

## Benchmarks

### RTX 5090 32GB, PCIe Gen5 x16

Qwen3.5-122B-A10B (122B params, 234 GB BF16 / 56 GB INT4). Single GPU, AWQ attention, FP8 KV cache, WriteCombined DMA.

| Config | Decode (tok/s) | Improvement |
|--------|:-:|:-:|
| Krasis v0.1.66 (baseline) | 38.7 | — |
| **Osmium (+ `--wc-alloc`)** | **57.0** | **+47%** |

### What makes this fast

- **WriteCombined DMA staging** (`--wc-alloc`): Expert weights allocated via `cuMemHostAlloc(WRITECOMBINED)` bypass the CPU cache hierarchy. PCIe Gen5 DMA reads hit ~46 GB/s vs ~28 GB/s for regular pinned memory. Per-component layout with incremental heap freeing keeps peak RAM manageable.
- **HCS (Hot Cache Strategy)**: Frequently-accessed MoE experts cached in VRAM, reducing PCIe DMA for hot paths.

Everything else (prefill, attention, quantization, KV cache) comes directly from upstream Krasis.

## Requirements

- **Linux** (Fedora 43, Ubuntu 24.04+, or WSL2)
- **NVIDIA GPU** with CUDA 12.8+ drivers
- **Python 3.13** (3.10+ should work)
- **System RAM**: ~74 GB with `--wc-alloc` for 122B models (91 GB physical + 64 GB NVMe swap recommended)
- **NVMe swap**: Required for WC mode. `sudo fallocate -l 64G /swapfile_osmium && sudo chmod 600 /swapfile_osmium && sudo mkswap /swapfile_osmium && sudo swapon /swapfile_osmium`

## Quick Start

```bash
git clone https://github.com/3spky5u-oss/osmium.git
cd osmium
git checkout osmium-v0.3.0

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

## What Osmium Changes vs Krasis

| Feature | Krasis v0.1.66 | Osmium v0.3 |
|---------|---------------|-------------|
| Expert DMA | Pinned memory (~28 GB/s) | WriteCombined (~46 GB/s) |
| Decode (122B, 1 GPU) | 38.7 tok/s | 57.0 tok/s |
| RAM management | Standard allocation | Incremental per-layer WC migration |
| RAM watchdog | 5% floor, no swap awareness | 0.5% floor, counts SwapFree |

Osmium is a single commit (`--wc-alloc`) on top of upstream Krasis. [PR #19](https://github.com/brontoguana/krasis/pull/19) is open upstream.

## Key Flags

| Flag | Description |
|------|-------------|
| `--wc-alloc` | WriteCombined DMA staging. +47% decode speed, requires swap. |
| `--gpu-expert-bits 4` | INT4 Marlin experts on GPU (default) |
| `--attention-quant awq` | AWQ attention quantization (frees VRAM for HCS) |
| `--kv-dtype fp8_e4m3` | FP8 KV cache (halves KV memory vs BF16) |
| `--kv-cache-mb N` | KV cache size. 1000 = 85K ctx, 3400 = 256K ctx |
| `--benchmark` | Run prefill + decode benchmark after loading |
| `--hcs` | Hot Cache Strategy (on by default with `--wc-alloc`) |

## API

OpenAI-compatible at `http://localhost:8012/v1/chat/completions` with SSE streaming. Works with Cursor, OpenCode, Continue, and any OpenAI SDK client.

```bash
curl http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

## License

SSPL-1.0 (inherited from Krasis)
