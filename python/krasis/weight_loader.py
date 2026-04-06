"""Streaming BF16→INT8 weight loader for Krasis standalone server.

Loads non-expert weights from HF safetensors one tensor at a time,
quantizes to INT8 per-channel symmetric, and places on target GPU.
Peak transient RAM: ~50 MB (one tensor buffer at a time).

Expert weights are loaded by the Krasis Rust engine (INT4, CPU RAM).
"""

import json
import logging
import os
import struct
import time
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from krasis.config import ModelConfig, PPRankConfig, QuantConfig

logger = logging.getLogger(__name__)


def quantize_to_int8(
    weight_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight to INT8 per-channel symmetric.

    Args:
        weight_bf16: [out_features, in_features] BF16 tensor

    Returns:
        (weight_int8, scale) where:
        - weight_int8: [out_features, in_features] torch.int8
        - scale: [out_features] torch.bfloat16 (per-channel)
    """
    w = weight_bf16.float()
    # Per-channel (per-row) max absolute value
    amax = w.abs().amax(dim=1).clamp(min=1e-10)
    scale = amax / 127.0
    w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.to(torch.bfloat16)


def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A8 INT8 matmul with on-the-fly activation quantization.

    Both weights and activations are INT8, matmul accumulates in INT32 via
    torch._int_mm for maximum precision. Works well for MLP/shared_expert/lm_head
    but NOT suitable for attention projections (use attention="bf16" instead).

    Args:
        x: [..., in_features] BF16 input
        weight_int8: [out_features, in_features] INT8
        scale: [out_features] BF16 per-channel scale

    Returns:
        [..., out_features] BF16
    """
    orig_shape = x.shape[:-1]
    in_features = x.shape[-1]
    x_2d = x.reshape(-1, in_features)
    M = x_2d.shape[0]

    # Quantize activation to INT8 per-token
    x_float = x_2d.float()
    x_amax = x_float.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    x_scale = x_amax / 127.0
    x_int8 = (x_float / x_scale).round().clamp(-128, 127).to(torch.int8)

    # _int_mm requires M >= 17 and M to be a multiple of 8 on SM89
    padded_M = max(M, 17)
    padded_M = (padded_M + 7) & ~7  # round up to multiple of 8
    if padded_M != M:
        pad = padded_M - M
        x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad))
        x_scale = torch.nn.functional.pad(x_scale, (0, 0, 0, pad))

    # INT8 matmul: [M, K] @ [K, N] -> [M, N] INT32
    out_int32 = torch._int_mm(x_int8, weight_int8.t())

    if padded_M != M:
        out_int32 = out_int32[:M]
        x_scale = x_scale[:M]

    # Dequantize: multiply by x_scale * w_scale
    out = out_int32.float() * (x_scale * scale.float().unsqueeze(0))
    out = out.to(torch.bfloat16).reshape(*orig_shape, -1)

    if bias is not None:
        out = out + bias

    return out


class WeightLoader:
    """Streaming weight loader — loads one tensor at a time from safetensors.

    Uses safe_open with framework="pt" for random-access tensor reading.
    Each tensor is read, quantized to INT8, placed on GPU, then the CPU
    buffer is freed. Peak transient RAM: ~50 MB.
    """

    def __init__(self, cfg: ModelConfig, quant_cfg: QuantConfig = None):
        self.cfg = cfg
        self.quant_cfg = quant_cfg or QuantConfig()
        self.model_path = cfg.model_path

        # Load safetensors index
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self._weight_map: Dict[str, str] = index["weight_map"]
        self._handles: Dict[str, object] = {}

    def _get_handle(self, shard_name: str):
        """Get or open a safetensors file handle (cached)."""
        if shard_name not in self._handles:
            path = os.path.join(self.model_path, shard_name)
            self._handles[shard_name] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_name]

    def _read_tensor(self, name: str) -> torch.Tensor:
        """Read a single tensor from safetensors by name."""
        shard_name = self._weight_map[name]
        handle = self._get_handle(shard_name)
        return handle.get_tensor(name)

    def _load_and_quantize(
        self, name: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a weight tensor, quantize to INT8, place on GPU.

        Returns (weight_int8, scale) both on device.
        """
        w = self._read_tensor(name).to(torch.bfloat16)
        w_int8, scale = quantize_to_int8(w)
        del w
        return w_int8.to(device), scale.to(device)

    def _load_bf16(self, name: str, device: torch.device) -> torch.Tensor:
        """Read a weight tensor and place on GPU as BF16."""
        w = self._read_tensor(name).to(torch.bfloat16)
        return w.to(device)



    def load_embedding(self, device: torch.device) -> torch.Tensor:
        """Load embedding table (BF16, ~2.2 GB for Kimi K2.5)."""
        name = f"{self.cfg.layers_prefix}.embed_tokens.weight"
        logger.info("Loading embedding: %s", name)
        return self._load_bf16(name, device)

    def load_final_norm(self, device: torch.device) -> torch.Tensor:
        """Load final RMSNorm weight (BF16, tiny)."""
        name = f"{self.cfg.layers_prefix}.norm.weight"
        logger.info("Loading final norm: %s", name)
        w = self._load_bf16(name, device)
        if self.cfg.norm_bias_one:
            w = w + 1.0  # Qwen3NextRMSNorm convention: (1 + weight) * x
        return w

    def load_lm_head(self, device: torch.device):
        """Load LM head weight.

        Returns (weight_int8, scale) if INT8, or plain BF16 tensor if BF16.
        """
        # lm_head location depends on model — try multiple naming conventions
        name = "lm_head.weight"
        if name not in self._weight_map:
            if self.cfg.layers_prefix != "model":
                prefix = self.cfg.layers_prefix.rsplit(".", 1)[0]
                name = f"{prefix}.lm_head.weight"
        logger.info("Loading LM head: %s (precision=%s)", name, self.quant_cfg.lm_head)
        if self.quant_cfg.lm_head == "bf16":
            return self._load_bf16(name, device)
        return self._load_and_quantize(name, device)

    def load_attention_weights(
        self, layer_idx: int, device: torch.device,
        proj_device: torch.device = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load attention weights for one layer (MLA or GQA).

        Args:
            device: Device for all weights by default (norms, biases, etc).
            proj_device: Device for large projection weights (q/k/v/o_proj).
                         Defaults to device. Pass CPU for AWQ mode where
                         projections stay on CPU for quantization, while
                         small weights (norms, biases) go directly to GPU.
        """
        if proj_device is None:
            proj_device = device
        if self.cfg.is_gqa:
            return self._load_gqa_attention(layer_idx, device, proj_device)
        return self._load_mla_attention(layer_idx, device)

    def _load_mla_attention(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load MLA attention weights for one layer.

        Handles both:
        - q_lora path (Kimi K2.5): q_a_proj + q_a_layernorm + q_b_proj
        - direct q path (V2-Lite): q_proj only

        kv_b_proj kept BF16 (split into w_kc, w_vc for quality).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        if self.cfg.has_q_lora:
            for proj in ["q_a_proj", "q_b_proj"]:
                weights[proj] = load_proj(f"{prefix}.{proj}.weight", device)
            weights["q_a_layernorm"] = self._load_bf16(f"{prefix}.q_a_layernorm.weight", device)
        else:
            weights["q_proj"] = load_proj(f"{prefix}.q_proj.weight", device)

        weights["kv_a_proj_with_mqa"] = load_proj(
            f"{prefix}.kv_a_proj_with_mqa.weight", device)
        weights["o_proj"] = load_proj(f"{prefix}.o_proj.weight", device)
        weights["kv_a_layernorm"] = self._load_bf16(f"{prefix}.kv_a_layernorm.weight", device)

        kv_b = self._load_bf16(f"{prefix}.kv_b_proj.weight", device)
        n_heads = self.cfg.num_attention_heads
        qk_nope = self.cfg.qk_nope_head_dim
        v_head = self.cfg.v_head_dim

        kv_b = kv_b.reshape(n_heads, qk_nope + v_head, self.cfg.kv_lora_rank)
        weights["w_kc"] = kv_b[:, :qk_nope, :].contiguous()
        weights["w_vc"] = kv_b[:, qk_nope:, :].contiguous()
        del kv_b

        return weights

    def _load_gqa_attention(
        self, layer_idx: int, device: torch.device,
        proj_device: torch.device = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load GQA attention weights for one layer (Qwen3, GLM-4.7).

        Loads: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
        Also loads bias tensors when present (GLM-4.7 has attention_bias=true).

        Args:
            device: Device for small weights (norms, biases) — always GPU.
            proj_device: Device for large projection weights. Defaults to device.
                         CPU when AWQ will quantize them before GPU upload.
        """
        if proj_device is None:
            proj_device = device
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[proj] = load_proj(f"{prefix}.{proj}.weight", proj_device)
            # Load bias if present (GLM-4.7) — biases are small, always on GPU
            bias_name = f"{prefix}.{proj}.bias"
            if bias_name in self._weight_map:
                weights[f"{proj}_bias"] = self._load_bf16(bias_name, device)

        # QK-Norm (Qwen3 and GLM-4.7 use RMSNorm on Q and K)
        q_norm_name = f"{prefix}.q_norm.weight"
        k_norm_name = f"{prefix}.k_norm.weight"
        if q_norm_name in self._weight_map:
            w = self._load_bf16(q_norm_name, device)
            if self.cfg.norm_bias_one:
                w = w + 1.0  # Qwen3NextRMSNorm convention
            weights["q_norm"] = w
        if k_norm_name in self._weight_map:
            w = self._load_bf16(k_norm_name, device)
            if self.cfg.norm_bias_one:
                w = w + 1.0  # Qwen3NextRMSNorm convention
            weights["k_norm"] = w

        # Attention sinks (GPT OSS: learnable logits for attention normalization)
        sinks_name = f"{prefix}.sinks"
        if sinks_name in self._weight_map:
            weights["sinks"] = self._load_bf16(sinks_name, device)

        return weights

    def load_layer_norms(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load input_layernorm and post_attention_layernorm (BF16)."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}"
        norms = {
            "input_layernorm": self._load_bf16(
                f"{prefix}.input_layernorm.weight", device),
            "post_attention_layernorm": self._load_bf16(
                f"{prefix}.post_attention_layernorm.weight", device),
        }
        if self.cfg.norm_bias_one:
            # Qwen3NextRMSNorm convention: (1 + weight) * x
            for k in norms:
                norms[k] = norms[k] + 1.0
        return norms

    def load_dense_mlp(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load dense MLP weights for a non-MoE layer.

        Uses quant_cfg.dense_mlp ("int8" or "bf16").
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp"
        load = self._load_and_quantize if self.quant_cfg.dense_mlp == "int8" else self._load_bf16
        return {
            "gate_proj": load(f"{prefix}.gate_proj.weight", device),
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

    def load_moe_gate(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Load MoE router gate weight and optional biases (BF16).

        Supports both naming conventions:
        - "mlp.gate" (DeepSeek/Kimi/Qwen3)
        - "mlp.router" (GPT OSS)
        """
        # Detect naming: "gate" vs "router"
        prefix_gate = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.gate"
        prefix_router = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.router"
        if f"{prefix_gate}.weight" in self._weight_map:
            prefix = prefix_gate
        else:
            prefix = prefix_router
        result = {
            "weight": self._load_bf16(f"{prefix}.weight", device),
        }
        # Router bias (GPT OSS)
        bias_name = f"{prefix}.bias"
        if bias_name in self._weight_map:
            result["bias"] = self._load_bf16(bias_name, device)
        # e_score_correction_bias (Kimi K2.5)
        corr_name = f"{prefix}.e_score_correction_bias"
        if corr_name in self._weight_map:
            result["e_score_correction_bias"] = self._load_bf16(corr_name, device)
        return result

    def load_shared_expert(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load shared expert MLP weights for a MoE layer.

        Uses quant_cfg.shared_expert ("int8" or "bf16").
        Handles both naming conventions:
        - "shared_experts" (plural, DeepSeek/Kimi)
        - "shared_expert" (singular, Qwen3-Next)
        """
        # Detect naming convention from weight map
        prefix_plural = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_experts"
        prefix_singular = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert"
        if f"{prefix_plural}.gate_proj.weight" in self._weight_map:
            prefix = prefix_plural
        else:
            prefix = prefix_singular

        load = self._load_and_quantize if self.quant_cfg.shared_expert == "int8" else self._load_bf16
        result = {
            "gate_proj": load(f"{prefix}.gate_proj.weight", device),
            "up_proj": load(f"{prefix}.up_proj.weight", device),
            "down_proj": load(f"{prefix}.down_proj.weight", device),
        }

        # Shared expert gate (Qwen3-Next): sigmoid gate on shared expert output
        # Weight: [1, hidden_size] — projects hidden → scalar per token
        gate_name = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        if gate_name in self._weight_map:
            result["shared_expert_gate"] = self._load_bf16(gate_name, device)

        return result

    def load_linear_attention_weights(
        self, layer_idx: int, device: torch.device
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Load Gated DeltaNet linear attention weights for one layer.

        Loads: in_proj_qkvz, in_proj_ba, out_proj (quantizable),
               conv1d.weight, A_log, dt_bias, norm.weight (always BF16).
        """
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.linear_attn"
        weights = {}
        # Attention weights always loaded as BF16 — adaptive FP8 conversion
        # for decode is handled separately in model.py after loading.
        load_proj = self._load_bf16

        # Quantizable projections — handle fused (QCN) or separate (Qwen3.5) format
        fused_qkvz = f"{prefix}.in_proj_qkvz.weight"
        if fused_qkvz in self._weight_map:
            # Fused format: in_proj_qkvz = [Q, K, V, Z], in_proj_ba = [B, A]
            weights["in_proj_qkvz"] = load_proj(fused_qkvz, device)
            weights["in_proj_ba"] = load_proj(f"{prefix}.in_proj_ba.weight", device)
        else:
            # Separate format (Qwen3.5): rearrange into fused interleaved format
            # QKV is flat [Q_all, K_all, V_all], Z/B/A are flat per-head
            # Must interleave per key-head group to match QCN's fused layout
            qkv_raw = self._load_bf16(f"{prefix}.in_proj_qkv.weight", device)
            z_raw = self._load_bf16(f"{prefix}.in_proj_z.weight", device)
            b_raw = self._load_bf16(f"{prefix}.in_proj_b.weight", device)
            a_raw = self._load_bf16(f"{prefix}.in_proj_a.weight", device)

            nk = self.cfg.linear_num_key_heads    # 16
            dk = self.cfg.linear_key_head_dim      # 128
            hr = self.cfg.linear_num_value_heads // nk  # 2
            dv = self.cfg.linear_value_head_dim    # 128
            key_dim = nk * dk   # 2048
            val_dim = self.cfg.linear_num_value_heads * dv  # 4096

            # Interleave QKVZ per key-head group: [q_i, k_i, v_i, z_i] for each group i
            qkvz_parts = []
            for i in range(nk):
                qkvz_parts.append(qkv_raw[i * dk : (i + 1) * dk])              # q group i
                qkvz_parts.append(qkv_raw[key_dim + i * dk : key_dim + (i + 1) * dk])  # k group i
                qkvz_parts.append(qkv_raw[key_dim * 2 + i * hr * dv : key_dim * 2 + (i + 1) * hr * dv])  # v group i
                qkvz_parts.append(z_raw[i * hr * dv : (i + 1) * hr * dv])      # z group i
            weights["in_proj_qkvz"] = torch.cat(qkvz_parts, dim=0)

            # Interleave BA per key-head group: [b_i, a_i] for each group i
            ba_parts = []
            for i in range(nk):
                ba_parts.append(b_raw[i * hr : (i + 1) * hr])
                ba_parts.append(a_raw[i * hr : (i + 1) * hr])
            weights["in_proj_ba"] = torch.cat(ba_parts, dim=0)
        weights["out_proj"] = load_proj(f"{prefix}.out_proj.weight", device)

        # Small/critical weights — always BF16
        weights["conv1d_weight"] = self._load_bf16(f"{prefix}.conv1d.weight", device)
        weights["A_log"] = self._load_bf16(f"{prefix}.A_log", device)
        weights["dt_bias"] = self._load_bf16(f"{prefix}.dt_bias", device)
        weights["norm_weight"] = self._load_bf16(f"{prefix}.norm.weight", device)

        return weights

    def load_layer(
        self, layer_idx: int, device: torch.device,
        attn_device: torch.device = None,
    ) -> Dict[str, any]:
        """Load all GPU weights for one layer.

        Args:
            device: Device for norms, gate, shared expert, dense MLP.
            attn_device: Device for attention weights. Defaults to device.
                         Pass torch.device('cpu') to keep attention in system RAM.

        Returns a dict with:
        - "norms": {input_layernorm, post_attention_layernorm}
        - "attention" or "linear_attention": attention weights
        - "layer_type": "linear_attention" or "full_attention"
        - "mlp": dense MLP weights (if dense layer) OR MoE gate + shared expert
        - "is_moe": bool
        """
        if attn_device is None:
            attn_device = device
        start = time.perf_counter()

        is_linear = self.cfg.is_linear_attention_layer(layer_idx)
        if is_linear:
            layer_type = "linear_attention"
        elif self.cfg.is_sliding_attention_layer(layer_idx):
            layer_type = "sliding_attention"
        else:
            layer_type = "full_attention"

        result = {
            "norms": self.load_layer_norms(layer_idx, device),
            "is_moe": self.cfg.is_moe_layer(layer_idx),
            "layer_type": layer_type,
        }

        # Load attention weights based on layer type
        # Linear attention is NOT affected by AWQ (AWQ only applies to GQA layers),
        # so linear attention always loads to the primary GPU device.
        if is_linear:
            result["linear_attention"] = self.load_linear_attention_weights(layer_idx, device)
        else:
            result["attention"] = self.load_attention_weights(
                layer_idx, device, proj_device=attn_device)

        if result["is_moe"]:
            result["gate"] = self.load_moe_gate(layer_idx, device)
            if self.cfg.n_shared_experts > 0:
                result["shared_expert"] = self.load_shared_expert(layer_idx, device)
        else:
            result["dense_mlp"] = self.load_dense_mlp(layer_idx, device)

        elapsed = time.perf_counter() - start
        alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
        logger.info(
            "Layer %d loaded in %.1fs (GPU alloc: %.0f MB, moe=%s, type=%s)",
            layer_idx, elapsed, alloc_mb, result["is_moe"], layer_type,
        )
        return result

    def close(self):
        """Close all safetensors handles."""
        self._handles.clear()


class GgufWeightLoader:
    """Weight loader that reads ALL non-expert weights from a GGUF file.

    Replaces WeightLoader (safetensors) when --gguf-path is provided.
    Uses the Rust GGUF parser (via gguf_read_tensor) to dequantize tensors
    to FP32, then converts to BF16 and optionally quantizes to INT8.

    Tensor name mapping: HF safetensors → llama.cpp GGUF convention.
    """

    # HF name component → GGUF name component
    _GGUF_NAME_MAP = {
        "embed_tokens": "token_embd",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
        "mlp.gate": "ffn_gate_inp",  # MoE router
        "mlp.shared_experts.gate_proj": "ffn_gate_shexp",
        "mlp.shared_experts.up_proj": "ffn_up_shexp",
        "mlp.shared_experts.down_proj": "ffn_down_shexp",
        "mlp.shared_expert_gate": "ffn_shexp_gate",
        "mlp.gate_proj": "ffn_gate",  # dense MLP
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }

    def __init__(self, cfg: ModelConfig, gguf_path: str, quant_cfg: QuantConfig = None):
        self.cfg = cfg
        self.gguf_path = gguf_path
        self.quant_cfg = quant_cfg or QuantConfig()
        self.model_path = cfg.model_path

        # Verify GGUF is readable and cache tensor list
        from krasis import gguf_list_tensors
        self._tensor_list = gguf_list_tensors(gguf_path)
        self._tensor_names = {name for name, _, _ in self._tensor_list}
        logger.info("GgufWeightLoader: %s — %d tensors", gguf_path, len(self._tensor_names))

    def _gguf_name(self, hf_name: str) -> str:
        """Convert HF tensor name to GGUF tensor name."""
        # Handle top-level tensors
        if hf_name == f"{self.cfg.layers_prefix}.embed_tokens.weight":
            return "token_embd.weight"
        if hf_name == f"{self.cfg.layers_prefix}.norm.weight":
            return "output_norm.weight"
        if hf_name == "lm_head.weight":
            return "output.weight"

        # Handle per-layer tensors: model.layers.{L}.{component}.weight
        prefix = f"{self.cfg.layers_prefix}.layers."
        if hf_name.startswith(prefix):
            rest = hf_name[len(prefix):]  # "{L}.{component}.weight"
            parts = rest.split(".", 1)  # ["{L}", "{component}.weight"]
            layer_idx = parts[0]
            component_and_suffix = parts[1]  # e.g. "self_attn.q_proj.weight"

            # Try each mapping
            for hf_comp, gguf_comp in self._GGUF_NAME_MAP.items():
                hf_pattern = f"{hf_comp}.weight"
                if component_and_suffix == hf_pattern:
                    return f"blk.{layer_idx}.{gguf_comp}.weight"
                # Handle bias
                hf_bias = f"{hf_comp}.bias"
                if component_and_suffix == hf_bias:
                    return f"blk.{layer_idx}.{gguf_comp}.bias"

        # Fallback: return as-is (will fail on lookup, giving a clear error)
        return hf_name

    def _read_tensor(self, hf_name: str) -> torch.Tensor:
        """Read tensor from GGUF, dequantize to FP32, return as torch tensor."""
        gguf_name = self._gguf_name(hf_name)
        if gguf_name not in self._tensor_names:
            raise KeyError(f"Tensor '{hf_name}' (GGUF: '{gguf_name}') not found in {self.gguf_path}")
        from krasis import gguf_read_tensor
        data, dims = gguf_read_tensor(self.gguf_path, gguf_name)
        shape = list(reversed(dims))  # GGUF stores dims in reverse (column-major)
        t = torch.tensor(data, dtype=torch.float32).reshape(shape)
        return t

    def _has_tensor(self, hf_name: str) -> bool:
        """Check if a tensor exists in the GGUF file."""
        return self._gguf_name(hf_name) in self._tensor_names

    def _load_and_quantize(self, name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self._read_tensor(name).to(torch.bfloat16)
        w_int8, scale = quantize_to_int8(w)
        del w
        return w_int8.to(device), scale.to(device)

    def _load_bf16(self, name: str, device: torch.device) -> torch.Tensor:
        w = self._read_tensor(name).to(torch.bfloat16)
        return w.to(device)

    # ── Public API (same as WeightLoader) ──

    def load_embedding(self, device: torch.device) -> torch.Tensor:
        name = f"{self.cfg.layers_prefix}.embed_tokens.weight"
        logger.info("Loading embedding from GGUF: %s", self._gguf_name(name))
        return self._load_bf16(name, device)

    def load_final_norm(self, device: torch.device) -> torch.Tensor:
        name = f"{self.cfg.layers_prefix}.norm.weight"
        logger.info("Loading final norm from GGUF: %s", self._gguf_name(name))
        return self._load_bf16(name, device)

    def load_lm_head(self, device: torch.device) -> torch.Tensor:
        name = "lm_head.weight"
        logger.info("Loading lm_head from GGUF: %s", self._gguf_name(name))
        quant = self.quant_cfg.lm_head
        if quant == "int8":
            return self._load_and_quantize(name, device)
        return self._load_bf16(name, device)

    def load_attention_weights(self, layer_idx: int, device: torch.device, on_cpu: bool = False):
        """Load attention Q/K/V/O weights from GGUF."""
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.self_attn"
        target = "cpu" if on_cpu else device

        result = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            name = f"{prefix}.{proj}.weight"
            if self.quant_cfg.attention == "int8" and not on_cpu:
                w_int8, scale = self._load_and_quantize(name, target)
                result[proj] = (w_int8, scale)
            else:
                result[proj] = self._load_bf16(name, target)

        # Q/K norms (optional, Qwen3 has them)
        for norm_name in ["q_norm", "k_norm"]:
            name = f"{prefix}.{norm_name}.weight"
            if self._has_tensor(name):
                result[norm_name] = self._load_bf16(name, device)

        return result

    def load_layer_norms(self, layer_idx: int, device: torch.device):
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}"
        return {
            "input_layernorm": self._load_bf16(f"{prefix}.input_layernorm.weight", device),
            "post_attention_layernorm": self._load_bf16(f"{prefix}.post_attention_layernorm.weight", device),
        }

    def load_moe_gate(self, layer_idx: int, device: torch.device):
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp"
        name = f"{prefix}.gate.weight"
        result = {"gate": self._load_bf16(name, device)}
        # Optional bias
        bias_name = f"{prefix}.gate.bias"
        if self._has_tensor(bias_name):
            result["gate_bias"] = self._load_bf16(bias_name, device)
        # Optional e_score_correction_bias (DeepSeek V3)
        corr_name = f"{prefix}.gate.e_score_correction_bias"
        if self._has_tensor(corr_name):
            result["e_score_correction_bias"] = self._load_bf16(corr_name, device)
        return result

    def load_shared_expert(self, layer_idx: int, device: torch.device):
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_experts"
        quant = self.quant_cfg.shared_expert
        result = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            name = f"{prefix}.{proj}.weight"
            if not self._has_tensor(name):
                return None
            if quant == "int8":
                result[proj] = self._load_and_quantize(name, device)
            else:
                result[proj] = self._load_bf16(name, device)

        # Optional shared expert gate
        gate_name = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        if self._has_tensor(gate_name):
            result["shared_expert_gate"] = self._load_bf16(gate_name, device)
        return result

    def load_dense_mlp(self, layer_idx: int, device: torch.device):
        prefix = f"{self.cfg.layers_prefix}.layers.{layer_idx}.mlp"
        quant = self.quant_cfg.dense_mlp
        result = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            name = f"{prefix}.{proj}.weight"
            if quant == "int8":
                result[proj] = self._load_and_quantize(name, device)
            else:
                result[proj] = self._load_bf16(name, device)
        return result

    def load_linear_attention_weights(self, layer_idx: int, device: torch.device):
        """Load linear attention weights (DeltaNet/GDN) from GGUF.

        GGUF naming for linear attention isn't standardized in llama.cpp.
        Try common naming patterns; return None if not found (fall back to safetensors).
        """
        # Linear attention tensors aren't in standard llama.cpp GGUF
        # This is a Qwen3.5-specific extension — may not be in the GGUF at all
        prefix = f"blk.{layer_idx}"
        # Check if any LA tensor exists
        la_names = [f"{prefix}.ssm_in.weight", f"{prefix}.la_in_proj.weight"]
        if not any(n in self._tensor_names for n in la_names):
            return None  # Not in GGUF — caller should fall back
        # TODO: implement full LA weight loading from GGUF when naming is standardized
        logger.warning("Linear attention GGUF loading not yet implemented for layer %d", layer_idx)
        return None

    def load_layer(self, layer_idx: int, device: torch.device,
                    attn_device: torch.device = None) -> dict:
        """Load all weights for a single layer."""
        start = time.perf_counter()
        layer_type = self.cfg.layer_type(layer_idx)
        is_moe = self.cfg.is_moe_layer(layer_idx)

        result = {"layer_type": layer_type, "is_moe": is_moe}

        if layer_type == "linear_attention":
            la = self.load_linear_attention_weights(layer_idx, device)
            if la is None:
                return None  # Signal to caller: use safetensors fallback for this layer
            result["linear_attention"] = la
        else:
            on_cpu = attn_device is not None and str(attn_device) == "cpu"
            if not on_cpu:
                on_cpu = (self.quant_cfg.attention == "awq")
            result["attention"] = self.load_attention_weights(layer_idx, device, on_cpu=on_cpu)

        result["norms"] = self.load_layer_norms(layer_idx, device)

        if is_moe:
            result["moe_gate"] = self.load_moe_gate(layer_idx, device)
            se = self.load_shared_expert(layer_idx, device)
            if se:
                result["shared_expert"] = se
        elif layer_type != "linear_attention":
            result["dense_mlp"] = self.load_dense_mlp(layer_idx, device)

        elapsed = time.perf_counter() - start
        alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
        logger.info("Layer %d loaded from GGUF in %.1fs (GPU alloc: %.0f MB, moe=%s, type=%s)",
                     layer_idx, elapsed, alloc_mb, is_moe, layer_type)
        return result

    def close(self):
        pass
