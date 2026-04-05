"""Paged KV cache supporting both MLA and GQA attention.

MLA (DeepSeek/Kimi): compresses all KV heads into a single latent vector per token.
  Split (FlashInfer MLA):
    - ckv_cache: [num_layers, num_pages, page_size, kv_lora_rank]
    - kpe_cache: [num_layers, num_pages, page_size, qk_rope_head_dim]
  Combined (TRTLLM MLA):
    - kv_cache: [num_layers, num_pages, page_size, kv_lora_rank + qk_rope_head_dim]

GQA (Qwen3): standard K/V caches with head dimension (NHD layout).
    - k_cache: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
    - v_cache: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import torch

from krasis.config import ModelConfig

logger = logging.getLogger(__name__)

PAGE_SIZE = 16  # tokens per page

# TRTLLM kernel constraint: block_num % (128 / page_size) == 0
TRTLLM_BLOCK_CONSTRAINT = 128


class PagedKVCache:
    """Manages paged KV cache for a set of layers on one GPU.

    Allocates a fixed pool of pages at init. Sequences claim pages
    from the free list as they grow; pages are returned on free.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_layers: int,
        device: torch.device,
        max_pages: Optional[int] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        page_size: int = PAGE_SIZE,
        combined: bool = False,
        max_mb: Optional[int] = None,
        kv_format: str = "fp8",  # "fp8" or "fp4"
    ):
        self.cfg = cfg
        self.num_layers = num_layers
        self.device = device
        self.kv_format = kv_format
        self.page_size = page_size
        self.kv_dtype = kv_dtype
        self.combined = combined
        self.attention_type = cfg.attention_type  # "mla" or "gqa"

        # Compute cache dimensions based on attention type
        if cfg.is_mla:
            self.ckv_dim = cfg.kv_lora_rank        # 512
            self.kpe_dim = cfg.qk_rope_head_dim    # 64
            self.kv_cache_dim = self.ckv_dim + self.kpe_dim  # 576
            self.num_kv_heads = None
            self.gqa_head_dim = None
        else:
            # GQA: standard K/V with head dimension
            self.ckv_dim = None
            self.kpe_dim = None
            self.num_kv_heads = cfg.num_key_value_heads
            self.gqa_head_dim = cfg.gqa_head_dim
            self.kv_cache_dim = cfg.num_key_value_heads * cfg.gqa_head_dim * 2  # K + V

        # Size from max_mb (preferred) or max_pages (explicit)
        if max_pages is None:
            if max_mb is None:
                max_mb = 2000  # default 2 GB
            budget_bytes = max_mb * 1024 * 1024
            bytes_per_page = self._bytes_per_page()

            # Cap to actual free VRAM minus computed safety margin.
            # Safety = FlashInfer workspace (256 MB) + max prefill intermediate
            # (MoE or dense MLP, whichever is larger at 5K token chunk) + 200 MB base.
            free_bytes, _ = torch.cuda.mem_get_info(device)
            chunk_est = 5000
            # MoE intermediates: [M*topk, 2*moe_inter] + [M*topk, hidden], bf16
            moe_inter = cfg.moe_intermediate_size
            top_k = cfg.num_experts_per_tok
            hidden = cfg.hidden_size
            moe_ws = (chunk_est * top_k * 2 * moe_inter * 2 +
                       chunk_est * top_k * hidden * 2) if moe_inter > 0 and top_k > 0 else 0
            # Dense MLP intermediates: gate + up + cat, peak = M * inter * 8
            dense_inter = cfg.intermediate_size
            dense_ws = chunk_est * dense_inter * 8 if dense_inter > 0 else 0
            prefill_ws = max(moe_ws, dense_ws)
            flashinfer_ws = 256 * 1024 * 1024  # matches attention.py _get_workspace
            base_headroom = 200 * 1024 * 1024
            safety_bytes = flashinfer_ws + prefill_ws + base_headroom
            safety_mb = safety_bytes / (1024 * 1024)
            available_bytes = max(0, free_bytes - safety_bytes)
            if budget_bytes > available_bytes:
                old_mb = budget_bytes / (1024 * 1024)
                budget_bytes = available_bytes
                new_mb = budget_bytes / (1024 * 1024)
                logger.warning(
                    "KV cache: requested %d MB but only %.0f MB available "
                    "(%.0f MB free - %.0f MB safety [%.0f prefill + 256 FlashInfer + 200 base]), "
                    "capping to %.0f MB",
                    int(old_mb), available_bytes / (1024 * 1024),
                    free_bytes / (1024 * 1024), safety_mb,
                    prefill_ws / (1024 * 1024), new_mb,
                )

            max_pages = max(64, budget_bytes // bytes_per_page)
            logger.info(
                "KV cache: %d MB → %d pages (%.1fK tokens)",
                max_mb, max_pages, max_pages * page_size / 1000,
            )

        self.max_pages = max_pages

        # GQA caches (separate K and V)
        self.k_cache = None
        self.v_cache = None
        # FP4 caches: packed data + per-group scale factors
        self.k_data_fp4 = None
        self.k_scale_fp4 = None
        self.v_data_fp4 = None
        self.v_scale_fp4 = None
        # MLA caches
        self.ckv_cache = None
        self.kpe_cache = None
        self.kv_cache = None

        if cfg.is_gqa and kv_format == "fp4":
            # FP4 GQA: packed uint8 data (2 values per byte) + FP8 scales (1 per 16 values)
            kv_dim = self.num_kv_heads * self.gqa_head_dim  # flat KV dimension
            self.fp4_group_size = 16
            self.k_data_fp4 = torch.zeros(
                num_layers, max_pages, page_size, kv_dim // 2,
                dtype=torch.uint8, device=device,
            )
            self.k_scale_fp4 = torch.zeros(
                num_layers, max_pages, page_size, kv_dim // self.fp4_group_size,
                dtype=torch.float8_e4m3fn, device=device,
            )
            self.v_data_fp4 = torch.zeros(
                num_layers, max_pages, page_size, kv_dim // 2,
                dtype=torch.uint8, device=device,
            )
            self.v_scale_fp4 = torch.zeros(
                num_layers, max_pages, page_size, kv_dim // self.fp4_group_size,
                dtype=torch.float8_e4m3fn, device=device,
            )
            alloc_mb = sum(t.nbytes for t in [
                self.k_data_fp4, self.k_scale_fp4, self.v_data_fp4, self.v_scale_fp4
            ]) / (1024**2)
            layout_str = "gqa-fp4"
            # FP8 shadow for FlashInfer prefill (same pages, both always allocated)
            self.k_cache = torch.zeros(
                num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                dtype=torch.float8_e4m3fn, device=device,
            )
            self.v_cache = torch.zeros(
                num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                dtype=torch.float8_e4m3fn, device=device,
            )
            alloc_mb += (self.k_cache.nbytes + self.v_cache.nbytes) / (1024**2)
        elif cfg.is_gqa:
            # GQA: separate K and V caches [layers, pages, page_size, heads, head_dim]
            self.k_cache = torch.zeros(
                num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                dtype=kv_dtype, device=device,
            )
            self.v_cache = torch.zeros(
                num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = (self.k_cache.nbytes + self.v_cache.nbytes) / (1024**2)
            layout_str = "gqa-split"
        elif combined:
            # TRTLLM MLA format: single combined cache
            self.kv_cache = torch.zeros(
                num_layers, max_pages, page_size, self.kv_cache_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = self.kv_cache.nbytes / (1024**2)
            layout_str = "mla-combined"
        else:
            # FlashInfer MLA format: split ckv + kpe caches
            self.ckv_cache = torch.zeros(
                num_layers, max_pages, page_size, self.ckv_dim,
                dtype=kv_dtype, device=device,
            )
            self.kpe_cache = torch.zeros(
                num_layers, max_pages, page_size, self.kpe_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = (self.ckv_cache.nbytes + self.kpe_cache.nbytes) / (1024**2)
            layout_str = "mla-split"

        logger.info(
            "KV cache allocated: %d layers × %d pages × %d tokens = %.0f MB (%s, %s)",
            num_layers, max_pages, page_size, alloc_mb,
            self.attention_type, layout_str,
        )

        # Free page tracking
        self._free_pages: List[int] = list(range(max_pages))
        self._free_pages.reverse()  # pop from end

    def _bytes_per_page(self) -> int:
        if self.kv_format == "fp4":
            # FP4 mode needs both FP8 (prefill) and FP4 (decode) per page.
            kv_dim = self.num_kv_heads * self.gqa_head_dim
            fp8_bytes = self.page_size * kv_dim * 2  # K + V, 1 byte each
            fp4_data = self.page_size * kv_dim // 2 * 2  # K + V packed
            fp4_scale = self.page_size * kv_dim // 16 * 2  # K + V scales
            return (fp8_bytes + fp4_data + fp4_scale) * self.num_layers
        elem_size = 1 if self.kv_dtype == torch.float8_e4m3fn else 2
        return self.page_size * self.kv_cache_dim * elem_size * self.num_layers

    @property
    def max_context_tokens(self) -> int:
        """Maximum number of tokens this cache can hold."""
        return self.max_pages * self.page_size

    @property
    def free_page_count(self) -> int:
        return len(self._free_pages)

    def alloc_pages(self, n: int) -> List[int]:
        """Allocate n pages from the free pool."""
        if n > len(self._free_pages):
            raise RuntimeError(
                f"KV cache exhausted: need {n} pages, have {len(self._free_pages)}"
            )
        pages = [self._free_pages.pop() for _ in range(n)]
        return pages

    def free_pages(self, pages: List[int]):
        """Return pages to the free pool.

        Page table indirection in the decode kernels handles non-contiguous
        page access, so ordering no longer matters.
        """
        self._free_pages.extend(pages)

    # ── MLA cache access ──

    def get_layer_caches(self, layer_offset: int):
        """Get split cache tensors for MLA (FlashInfer).

        Returns (ckv_cache, kpe_cache) each [max_pages, page_size, dim].
        """
        assert self.attention_type == "mla" and not self.combined
        return self.ckv_cache[layer_offset], self.kpe_cache[layer_offset]

    def get_combined_layer_cache(self, layer_offset: int) -> torch.Tensor:
        """Get combined KV cache for MLA (TRTLLM format)."""
        assert self.attention_type == "mla" and self.combined
        return self.kv_cache[layer_offset].unsqueeze(0)

    # ── GQA cache access ──

    def get_gqa_layer_caches(self, layer_offset: int):
        """Get (k_cache, v_cache) for GQA (FlashInfer standard paged attention).

        Returns (k, v) each [max_pages, page_size, num_kv_heads, head_dim].
        """
        assert self.attention_type == "gqa"
        return self.k_cache[layer_offset], self.v_cache[layer_offset]

    # ── FP4 shadow cache management ──

    def alloc_fp8_shadow(self):
        """Allocate FP8 shadow caches for FlashInfer prefill (FP4 mode only).
        Called before prefill; freed after FP4 quantization via free_fp8_shadow().
        """
        if self.kv_format != "fp4" or getattr(self, '_fp8_shadow_allocated', False):
            return
        self.k_cache = torch.zeros(
            self.num_layers, self.max_pages, self.page_size,
            self.num_kv_heads, self.gqa_head_dim,
            dtype=torch.float8_e4m3fn, device=self.device,
        )
        self.v_cache = torch.zeros(
            self.num_layers, self.max_pages, self.page_size,
            self.num_kv_heads, self.gqa_head_dim,
            dtype=torch.float8_e4m3fn, device=self.device,
        )
        self._fp8_shadow_allocated = True
        mb = (self.k_cache.nbytes + self.v_cache.nbytes) / (1024**2)
        logger.info("FP4: allocated FP8 shadow cache (%.0f MB) for prefill", mb)

    def free_fp8_shadow(self):
        """Free FP8 shadow caches after FP4 quantization (FP4 mode only).
        Frees VRAM for HCS/decode.
        """
        if self.kv_format != "fp4" or not getattr(self, '_fp8_shadow_allocated', False):
            return
        mb = (self.k_cache.nbytes + self.v_cache.nbytes) / (1024**2)
        del self.k_cache, self.v_cache
        self.k_cache = None
        self.v_cache = None
        self._fp8_shadow_allocated = False
        torch.cuda.empty_cache()
        logger.info("FP4: freed FP8 shadow cache (%.0f MB)", mb)


class SequenceKVState:
    """KV cache state for a single sequence (request).

    Tracks which pages are allocated and current position.
    Provides FlashInfer-compatible index arrays.
    """

    def __init__(self, cache: PagedKVCache, seq_id: int = 0):
        self.cache = cache
        self.seq_id = seq_id
        self.pages: List[int] = []
        self.seq_len: int = 0  # number of tokens in cache

    def ensure_capacity(self, new_tokens: int):
        """Ensure we have enough pages for new_tokens more tokens."""
        total_needed = self.seq_len + new_tokens
        pages_needed = (total_needed + self.cache.page_size - 1) // self.cache.page_size
        if pages_needed > len(self.pages):
            extra = pages_needed - len(self.pages)
            new_pages = self.cache.alloc_pages(extra)
            self.pages.extend(new_pages)

    def advance(self, num_tokens: int):
        """Record that num_tokens were appended to the cache."""
        self.seq_len += num_tokens

    def free(self):
        """Release all pages back to the pool."""
        if self.pages:
            self.cache.free_pages(self.pages)
            self.pages = []
            self.seq_len = 0

    def kv_indices(self, device: torch.device) -> torch.Tensor:
        """Page indices for FlashInfer: all allocated pages."""
        return torch.tensor(self.pages, dtype=torch.int32, device=device) if self.pages else torch.zeros(0, dtype=torch.int32, device=device)

    def kv_indptr(self, device: torch.device) -> torch.Tensor:
        """Page indptr for FlashInfer (single sequence): [0, num_allocated_pages]."""
        return torch.tensor([0, len(self.pages)], dtype=torch.int32, device=device)

    def kv_len_arr(self, device: torch.device) -> torch.Tensor:
        """Sequence length array: [seq_len]."""
        return torch.tensor([self.seq_len], dtype=torch.int32, device=device)

    def last_page_len(self) -> int:
        """Number of valid tokens in the last page."""
        if self.seq_len == 0:
            return 0
        rem = self.seq_len % self.cache.page_size
        return rem if rem > 0 else self.cache.page_size

    def last_page_len_tensor(self, device: torch.device) -> torch.Tensor:
        """Last page length as tensor (for FlashInfer decode)."""
        return torch.tensor([self.last_page_len()], dtype=torch.int32, device=device)

    def block_tables(self, device: torch.device, pad_to_multiple: int = 8) -> torch.Tensor:
        """Block (page) indices for TRTLLM decode kernel.

        Returns [1, padded_num_blocks] int32.
        """
        num_blocks = len(self.pages)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.cache.page_size
        padded = math.ceil(num_blocks / constraint) * constraint if num_blocks > 0 else constraint
        table = torch.full((1, padded), -1, dtype=torch.int32, device=device)
        if num_blocks > 0:
            table[0, :num_blocks] = torch.tensor(self.pages, dtype=torch.int32, device=device)
        return table

    def store_kv_combined(
        self,
        layer_offset: int,
        kv_combined: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Store combined KV [M, kv_cache_dim] into the paged cache (TRTLLM MLA)."""
        assert self.cache.combined, "store_kv_combined requires combined cache"
        page_size = self.cache.page_size
        pages_tensor = torch.tensor(self.pages, dtype=torch.long, device=kv_combined.device)

        page_indices = pages_tensor[positions.long() // page_size]
        slots = (positions.long() % page_size)

        self.cache.kv_cache[layer_offset, page_indices, slots] = kv_combined.to(self.cache.kv_dtype)


class SequenceManager:
    """Manages multiple concurrent sequences sharing a single PagedKVCache pool.

    Each sequence gets its own SequenceKVState with independent page allocation.
    Switching the active sequence updates the GPU page table so Rust decode
    addresses the correct physical pages.

    Typical usage:
        mgr = SequenceManager(cache, gpu_store)
        sid = mgr.create("conv-1")        # allocate a new sequence
        mgr.activate("conv-1")            # push page table to GPU
        # ... prefill + decode ...
        mgr.park("conv-1")               # save LA state, keep pages
        mgr.activate("conv-2")           # switch to another sequence
    """

    def __init__(self, cache: PagedKVCache, gpu_store=None, max_sequences: int = 3):
        self.cache = cache
        self.gpu_store = gpu_store  # GpuDecodeStore (Rust)
        self.max_sequences = max_sequences
        self.sequences: Dict[str, SequenceKVState] = {}
        self.active_id: Optional[str] = None
        self._last_used: Dict[str, float] = {}  # conv_id → timestamp
        self._la_snapshots: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def create(self, conv_id: str) -> SequenceKVState:
        """Create a new sequence. Evicts LRU if at capacity."""
        if conv_id in self.sequences:
            return self.sequences[conv_id]
        if len(self.sequences) >= self.max_sequences:
            evicted = self.evict_lru()
            if evicted:
                logger.info("SequenceManager: evicted '%s' to make room for '%s'", evicted, conv_id)
        state = SequenceKVState(self.cache, seq_id=len(self.sequences))
        self.sequences[conv_id] = state
        self._last_used[conv_id] = time.monotonic()
        logger.info("SequenceManager: created '%s' (%d/%d sequences)",
                     conv_id, len(self.sequences), self.max_sequences)
        return state

    def get(self, conv_id: str) -> Optional[SequenceKVState]:
        """Get a sequence by ID, or None if not found."""
        return self.sequences.get(conv_id)

    def activate(self, conv_id: str):
        """Make conv_id the active decode target. Pushes page table to GPU."""
        if conv_id not in self.sequences:
            raise KeyError(f"Unknown sequence '{conv_id}'")
        state = self.sequences[conv_id]
        self._last_used[conv_id] = time.monotonic()

        if self.gpu_store is not None and state.pages:
            self.gpu_store.set_page_table(state.pages)
            self.gpu_store.set_kv_position(state.seq_len)

        self.active_id = conv_id
        logger.debug("SequenceManager: activated '%s' (seq_len=%d, pages=%d)",
                      conv_id, state.seq_len, len(state.pages))

    def park(self, conv_id: str):
        """Park a sequence (keep pages allocated, stop decoding)."""
        if conv_id == self.active_id:
            self.active_id = None

    def destroy(self, conv_id: str):
        """Free all resources for a sequence."""
        if conv_id in self.sequences:
            self.sequences[conv_id].free()
            del self.sequences[conv_id]
            self._last_used.pop(conv_id, None)
            self._la_snapshots.pop(conv_id, None)
            if self.active_id == conv_id:
                self.active_id = None
            logger.info("SequenceManager: destroyed '%s' (%d sequences remain)",
                         conv_id, len(self.sequences))

    def evict_lru(self) -> Optional[str]:
        """Evict the least-recently-used non-active sequence. Returns evicted ID."""
        candidates = [
            (t, cid) for cid, t in self._last_used.items()
            if cid != self.active_id
        ]
        if not candidates:
            return None
        candidates.sort()
        _, evict_id = candidates[0]
        self.destroy(evict_id)
        return evict_id

    def page_budget_report(self) -> dict:
        """Return page usage stats."""
        used = sum(len(s.pages) for s in self.sequences.values())
        return {
            "total_pages": self.cache.max_pages,
            "free_pages": self.cache.free_page_count,
            "used_pages": used,
            "sequences": {
                cid: {"pages": len(s.pages), "seq_len": s.seq_len,
                       "swapped": getattr(s, 'swapped', False)}
                for cid, s in self.sequences.items()
            },
        }

    # ── CPU Swap ──

    def swap_out(self, conv_id: str):
        """Swap a sequence's KV pages to CPU pinned memory, freeing GPU pages.

        After swap-out the sequence's pages list is empty (GPU pages freed)
        but seq_len is preserved. swap_in() restores the pages.
        """
        if conv_id not in self.sequences:
            raise KeyError(f"Unknown sequence '{conv_id}'")
        state = self.sequences[conv_id]
        if not state.pages:
            return  # nothing to swap
        if getattr(state, 'swapped', False):
            return  # already swapped

        cache = self.cache
        t0 = time.monotonic()

        # Create CPU pinned storage for this sequence's pages
        swap = _SwapHandle(state.pages[:], state.seq_len, cache)
        swap.copy_gpu_to_cpu(cache)

        # Free GPU pages
        state.free()
        state.seq_len = swap.seq_len  # restore seq_len (free() zeroes it)
        state.swapped = True
        state._swap_handle = swap

        if conv_id == self.active_id:
            self.active_id = None

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("SequenceManager: swapped out '%s' (%d pages, %d tokens, %.1fms)",
                     conv_id, len(swap.pages), swap.seq_len, elapsed_ms)

    def swap_in(self, conv_id: str):
        """Restore a swapped-out sequence's KV pages from CPU to GPU."""
        if conv_id not in self.sequences:
            raise KeyError(f"Unknown sequence '{conv_id}'")
        state = self.sequences[conv_id]
        if not getattr(state, 'swapped', False):
            return  # not swapped
        swap: _SwapHandle = state._swap_handle

        cache = self.cache
        t0 = time.monotonic()

        # Allocate fresh GPU pages (may be different physical pages)
        new_pages = cache.alloc_pages(len(swap.pages))
        swap.copy_cpu_to_gpu(cache, new_pages)

        state.pages = new_pages
        state.seq_len = swap.seq_len
        state.swapped = False
        del state._swap_handle

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("SequenceManager: swapped in '%s' (%d pages, %d tokens, %.1fms)",
                     conv_id, len(new_pages), state.seq_len, elapsed_ms)


class _SwapHandle:
    """Holds CPU-side copies of swapped-out KV cache pages."""

    def __init__(self, pages: List[int], seq_len: int, cache: PagedKVCache):
        self.pages = pages
        self.seq_len = seq_len
        self.num_layers = cache.num_layers
        self.is_gqa = cache.attention_type == "gqa"
        # CPU pinned storage — allocated per-layer as contiguous blocks
        # covering only the pages belonging to this sequence.
        self.k_cpu: Optional[torch.Tensor] = None
        self.v_cpu: Optional[torch.Tensor] = None
        # For MLA
        self.ckv_cpu: Optional[torch.Tensor] = None
        self.kpe_cpu: Optional[torch.Tensor] = None

    def copy_gpu_to_cpu(self, cache: PagedKVCache):
        """Copy this sequence's pages from GPU cache to CPU pinned tensors."""
        pages_t = torch.tensor(self.pages, dtype=torch.long)
        if self.is_gqa:
            # k_cache shape: [num_layers, max_pages, page_size, nkv, hd]
            # Gather pages for all layers at once, copy to CPU pinned memory.
            # .cpu() does D2H copy, .pin_memory() re-allocates into pinned pages.
            self.k_cpu = cache.k_cache[:, pages_t].contiguous().cpu().pin_memory()
            self.v_cpu = cache.v_cache[:, pages_t].contiguous().cpu().pin_memory()
        else:
            if cache.ckv_cache is not None:
                self.ckv_cpu = cache.ckv_cache[:, pages_t].contiguous().cpu().pin_memory()
                self.kpe_cpu = cache.kpe_cache[:, pages_t].contiguous().cpu().pin_memory()
            elif cache.kv_cache is not None:
                self.k_cpu = cache.kv_cache[:, pages_t].contiguous().cpu().pin_memory()

    def copy_cpu_to_gpu(self, cache: PagedKVCache, new_pages: List[int]):
        """Copy CPU pinned tensors back to GPU cache at new page locations."""
        new_t = torch.tensor(new_pages, dtype=torch.long)
        device = cache.device
        if self.is_gqa:
            cache.k_cache[:, new_t] = self.k_cpu.to(device, non_blocking=True)
            cache.v_cache[:, new_t] = self.v_cpu.to(device, non_blocking=True)
        else:
            if self.ckv_cpu is not None:
                cache.ckv_cache[:, new_t] = self.ckv_cpu.to(device, non_blocking=True)
                cache.kpe_cache[:, new_t] = self.kpe_cpu.to(device, non_blocking=True)
            elif self.k_cpu is not None:
                cache.kv_cache[:, new_t] = self.k_cpu.to(device, non_blocking=True)
        # Sync to ensure copy completes before the tensors are used
        torch.cuda.synchronize(device)
