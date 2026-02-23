# Token Scaling Analysis — Qwen3-Coder-Next (1 GPU, INT4/INT4)

Measures how prefill speed scales with prompt length.
Identifies fixed per-prefill costs vs linear scaling.

Config: 1x RTX 2000 Ada, layer_group_size=2, stream_attention, FP8 KV, 3 runs/size

## Results

|   Tokens |   TTFT (s) |      tok/s |     ms/tok |   Fixed cost est |
| -------- | ---------- | ---------- | ---------- | ---------------- |
|      100 |      3.618 |         28 |     36.178 |                  |
|      500 |      7.868 |         64 |     15.736 |                  |
|    2,000 |      7.652 |        261 |      3.826 |                  |
|   10,000 |      9.842 |       1017 |      0.984 |                  |
|   25,000 |     26.754 |        934 |      1.070 |                  |

## Analysis

Linear regression (100 → 10,000 tokens):
- **Fixed cost per prefill**: 3554.9 ms
- **Marginal cost per token**: 0.6287 ms (1591 tok/s throughput)
- **At 100 tokens**: fixed cost is 98% of total time
- **At 10K tokens**: fixed cost is 36% of total time
- **At 25K tokens**: predicted 19.27s, actual 26.75s (SUPERLINEAR)

## Per-Run Detail

### 100 tokens
- Run 1: TTFT=4.260s, wall=4.261s, 23 tok/s
- Run 2: TTFT=3.180s, wall=3.180s, 31 tok/s
- Run 3: TTFT=3.413s, wall=3.413s, 29 tok/s

### 500 tokens
- Run 1: TTFT=7.641s, wall=7.641s, 65 tok/s
- Run 2: TTFT=7.589s, wall=7.589s, 66 tok/s
- Run 3: TTFT=8.374s, wall=8.374s, 60 tok/s

### 2,000 tokens
- Run 1: TTFT=7.645s, wall=7.646s, 262 tok/s
- Run 2: TTFT=7.668s, wall=7.669s, 261 tok/s
- Run 3: TTFT=7.642s, wall=7.643s, 262 tok/s

### 10,000 tokens
- Run 1: TTFT=10.367s, wall=10.367s, 965 tok/s
- Run 2: TTFT=9.582s, wall=9.583s, 1044 tok/s
- Run 3: TTFT=9.578s, wall=9.578s, 1044 tok/s

### 25,000 tokens
- Run 1: TTFT=26.534s, wall=26.535s, 942 tok/s
- Run 2: TTFT=26.790s, wall=26.791s, 933 tok/s
- Run 3: TTFT=26.938s, wall=26.938s, 928 tok/s
