"""Minimal timing flags for Python prefill instrumentation.

Decode timing is handled entirely by Rust (KRASIS_DECODE_TIMING env var).
These flags gate optional synchronize+perf_counter blocks in Python prefill code.
"""

import os


class _TimingFlags:
    __slots__ = ("prefill", "decode", "diag")

    def __init__(self):
        self.prefill = False
        self.decode = False
        self.diag = False


TIMING = _TimingFlags()
