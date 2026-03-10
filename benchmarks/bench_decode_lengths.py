#!/usr/bin/env python3
"""Benchmark decode speed at multiple OUTPUT lengths for Krasis and llama.cpp.

Tests 100, 500, 1000, 2500 token decodes with a fixed short prompt.
Measures pure decode speed (tok/s) for fair comparison.

Usage:
    # Krasis (server must be running):
    python3 bench_decode_lengths.py krasis

    # llama.cpp:
    python3 bench_decode_lengths.py llama
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("PORT", 8080))
SERVER = f"http://localhost:{PORT}"
MODEL = os.environ.get("MODEL", "Qwen3-Coder-Next")

LLAMA_CLI = os.path.expanduser(
    "~/Documents/Claude/llama/llama.cpp/build/bin/llama-cli"
)
LLAMA_MODEL = os.path.expanduser(
    "~/.krasis/models/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M/"
    "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
)

# Short prompt so prefill is minimal and we mostly measure decode
PROMPT = "Write a detailed essay about the history of computing, from the earliest mechanical calculators through modern quantum computers. Cover all major developments, key figures, and technological breakthroughs in chronological order."

DECODE_LENGTHS = [100, 500, 1000, 2500]


def run_krasis(n_tokens):
    """Run Krasis decode and return tok/s."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": n_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    engine_timing = None

    with urllib.request.urlopen(req, timeout=3600) as resp:
        buffer = b""
        for chunk in iter(lambda: resp.read(1), b""):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        obj = json.loads(line[6:])
                        if "krasis_timing" in obj:
                            engine_timing = obj["krasis_timing"]
                            continue
                        delta = obj["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.perf_counter()
                            if t_first_token is None:
                                t_first_token = now
                            token_count += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    t_end = time.perf_counter()

    if t_first_token and token_count > 1:
        decode_time = t_end - t_first_token
        decode_tokens = token_count - 1
        http_tok_s = decode_tokens / decode_time
    else:
        http_tok_s = 0

    eng_tok_s = 0
    if engine_timing:
        eng_tok_s = engine_timing.get("decode_tok_s", 0)

    return {
        "tokens": token_count,
        "http_tok_s": round(http_tok_s, 1),
        "engine_tok_s": round(eng_tok_s, 1),
        "ttft_s": round((t_first_token - t_start) if t_first_token else 0, 2),
        "total_s": round(t_end - t_start, 2),
    }


def run_llama(n_tokens, ngl=99):
    """Run llama.cpp decode and return tok/s."""
    cmd = [
        LLAMA_CLI,
        "-m", LLAMA_MODEL,
        "-p", PROMPT,
        "-n", str(n_tokens),
        "-ngl", str(ngl),
        "-c", "4096",
        "--temp", "0.7",
        "-s", "42",
        "--no-display-prompt",
        "-t", "16",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        stderr = result.stderr

        # Parse llama.cpp timing from stderr
        # llama_perf_sampler_print:    sampling time =     X.XX ms /   N runs   (    X.XX ms per token, XXXX.XX tokens per second)
        # llama_perf_context_print:        eval time =     X.XX ms /   N runs   (    X.XX ms per token, XXXX.XX tokens per second)

        eval_match = re.search(
            r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
            stderr,
        )

        prompt_match = re.search(
            r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
            stderr,
        )

        if eval_match:
            eval_ms = float(eval_match.group(1))
            eval_tokens = int(eval_match.group(2))
            tok_s = float(eval_match.group(4))
        else:
            eval_ms = 0
            eval_tokens = 0
            tok_s = 0

        prefill_ms = 0
        if prompt_match:
            prefill_ms = float(prompt_match.group(1))

        return {
            "tokens": eval_tokens,
            "tok_s": round(tok_s, 1),
            "eval_ms": round(eval_ms, 1),
            "prefill_ms": round(prefill_ms, 1),
            "total_s": round((eval_ms + prefill_ms) / 1000, 2),
        }
    except subprocess.TimeoutExpired:
        return {"tokens": 0, "tok_s": 0, "error": "timeout"}
    except Exception as e:
        return {"tokens": 0, "tok_s": 0, "error": str(e)}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("krasis", "llama"):
        print("Usage: python3 bench_decode_lengths.py [krasis|llama]")
        sys.exit(1)

    mode = sys.argv[1]

    print(f"=== Decode Length Benchmark: {mode} ===")
    print(f"Prompt: {len(PROMPT)} chars (~{len(PROMPT)//4} tokens)")
    print()

    if mode == "krasis":
        # Check server
        try:
            req = urllib.request.Request(f"{SERVER}/health")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            print(f"ERROR: Krasis server not running at {SERVER}")
            sys.exit(1)

        # Warmup
        print("Warming up...", end="", flush=True)
        run_krasis(20)
        print(" done\n")

        print(f"{'Tokens':<10} {'Engine tok/s':<14} {'HTTP tok/s':<12} {'TTFT(s)':<10} {'Total(s)'}")
        print("-" * 56)

        for n in DECODE_LENGTHS:
            r = run_krasis(n)
            print(
                f"{r['tokens']:<10} {r['engine_tok_s']:<14} {r['http_tok_s']:<12} "
                f"{r['ttft_s']:<10} {r['total_s']}"
            )

    elif mode == "llama":
        # Warmup
        print("Warming up (short run)...", end="", flush=True)
        run_llama(20)
        print(" done\n")

        print(f"{'Tokens':<10} {'tok/s':<10} {'Decode(ms)':<12} {'Prefill(ms)':<12} {'Total(s)'}")
        print("-" * 56)

        for n in DECODE_LENGTHS:
            r = run_llama(n)
            if "error" in r:
                print(f"{n:<10} ERROR: {r['error']}")
            else:
                print(
                    f"{r['tokens']:<10} {r['tok_s']:<10} {r['eval_ms']:<12} "
                    f"{r['prefill_ms']:<12} {r['total_s']}"
                )


if __name__ == "__main__":
    main()
