#!/usr/bin/env python3
"""Compare HF transformers layer 0 intermediate values against Rust prefill diagnostics.

Loads the BF16 model (just enough for layer 0), runs the same 31-token reference
prompt, and captures norms at each stage of the linear attention forward pass.

Usage: KRASIS_DEV_SCRIPT=1 python3 tests/hf_layer0_compare.py
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU only
import torch
import torch.nn.functional as F

if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("Run via ./dev, not directly")
    sys.exit(1)

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")
REF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "krasis-internal",
    "reference-outputs", "output", "Qwen3-Coder-Next-sublayers", "greedy_reference.json")

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

def main():
    with open(REF_PATH) as f:
        ref = json.load(f)
    token_ids = ref["conversations"][0]["turns"][0]["input_token_ids"]
    print(f"Prompt: {len(token_ids)} tokens")

    # Load model config
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(MODEL_PATH)
    print(f"Model: {config.model_type}, hidden={config.hidden_size}, layers={config.num_hidden_layers}")

    # Load full model but only use layer 0
    print("Loading model (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Get layer 0 LA module
    layer0 = model.model.layers[0]
    la = layer0.linear_attn  # The linear attention module
    print(f"Layer 0 mixer type: {type(la).__name__}")

    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        # Get embedding
        emb = model.model.embed_tokens(input_ids)  # [1, M, hidden]
        print(f"\nEmbedding pos0 norm: {emb[0, 0].float().norm().item():.6f}")

        # Apply input norm (RMSNorm)
        hidden = model.model.layers[0].input_layernorm(emb)

        # --- Now manually trace through the LA forward ---
        M = hidden.shape[1]  # 31
        hidden_bf16 = hidden.to(torch.bfloat16)

        # 1. in_proj_qkvz GEMM
        projected_qkvz = la.in_proj_qkvz(hidden_bf16)  # [1, M, qkvz_dim]
        print(f"qkvz_gemm pos0 norm: {projected_qkvz[0, 0].float().norm().item():.6f} (dim={projected_qkvz.shape[-1]})")

        # 2. in_proj_ba GEMM
        projected_ba = la.in_proj_ba(hidden_bf16)

        # 3. fix_query_key_value_ordering
        query, key, value, z, b, a = la.fix_query_key_value_ordering(projected_qkvz, projected_ba)
        # query: [1, M, nk, dk], key: [1, M, nk, dk], value: [1, M, nv, dv], z: [1, M, nv, dv]

        # Flatten for conv input
        query_flat = query.reshape(1, M, -1)  # [1, M, key_dim]
        key_flat = key.reshape(1, M, -1)      # [1, M, key_dim]
        value_flat = value.reshape(1, M, -1)  # [1, M, value_dim]

        nk = config.linear_num_key_heads
        nv = config.linear_num_value_heads
        dk = config.linear_key_head_dim
        dv = config.linear_value_head_dim
        hr = nv // nk
        key_dim = nk * dk
        value_dim = nv * dv

        print(f"\nAfter uninterleave pos0:")
        print(f"  q: {query_flat[0, 0].float().norm().item():.6f}")
        print(f"  k: {key_flat[0, 0].float().norm().item():.6f}")
        print(f"  v: {value_flat[0, 0].float().norm().item():.6f}")
        print(f"  z: {z[0, 0].reshape(-1).float().norm().item():.6f}")

        # 4. Conv1d
        mixed_qkv = torch.cat((query_flat, key_flat, value_flat), dim=-1)  # [1, M, conv_dim]
        mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [1, conv_dim, M]

        # Use the same path as forward()
        if la.causal_conv1d_fn is not None:
            mixed_qkv_out = la.causal_conv1d_fn(
                x=mixed_qkv_t,
                weight=la.conv1d.weight.squeeze(1),
                bias=la.conv1d.bias,
                activation=la.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv_out = F.silu(la.conv1d(mixed_qkv_t)[:, :, :M])

        mixed_qkv_out = mixed_qkv_out.transpose(1, 2)  # [1, M, conv_dim]

        # Split back
        q_conv, k_conv, v_conv = torch.split(
            mixed_qkv_out, [key_dim, key_dim, value_dim], dim=-1)
        q_conv = q_conv.reshape(1, M, nk, dk)
        k_conv = k_conv.reshape(1, M, nk, dk)
        v_conv = v_conv.reshape(1, M, nv, dv)

        print(f"\nAfter conv+silu pos0 (FP32 equiv):")
        print(f"  q: {q_conv[0, 0].reshape(-1).float().norm().item():.6f}")
        print(f"  k: {k_conv[0, 0].reshape(-1).float().norm().item():.6f}")
        print(f"  v: {v_conv[0, 0].reshape(-1).float().norm().item():.6f}")

        # 5. Gate and beta
        beta = b.sigmoid()  # [1, M, nv]
        g_raw = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)  # [1, M, nv]

        # 6. Repeat interleave
        if hr > 1:
            q_conv = q_conv.repeat_interleave(hr, dim=2)  # [1, M, nv, dk]
            k_conv = k_conv.repeat_interleave(hr, dim=2)  # [1, M, nv, dk]

        # 7. Chunk gated delta rule (use the torch fallback for clarity)
        print(f"\nBefore chunk_gated_delta_rule:")
        print(f"  q_conv norm (all pos): {q_conv.float().norm().item():.6f}")
        print(f"  k_conv norm (all pos): {k_conv.float().norm().item():.6f}")
        print(f"  v_conv norm (all pos): {v_conv.float().norm().item():.6f}")

        # L2 normalize (done inside chunk_gated_delta_rule)
        q_normed = l2norm(q_conv.float(), dim=-1)
        k_normed = l2norm(k_conv.float(), dim=-1)
        scale = 1.0 / (dk ** 0.5)
        q_scaled = q_normed * scale

        print(f"\nAfter L2 norm pos0:")
        print(f"  q (all heads): {q_scaled[0, 0].reshape(-1).norm().item():.6f}")
        print(f"  k (all heads): {k_normed[0, 0].reshape(-1).norm().item():.6f}")

        # Compute v_beta and k_beta
        # Note: beta has shape [1, M, nv], v has [1, M, nv, dv]
        v_beta = v_conv.float() * beta.float().unsqueeze(-1)
        k_beta = k_normed * beta.float().unsqueeze(-1)  # k after L2 norm * beta

        print(f"\n  v_beta h0p0: {v_beta[0, 0, 0].norm().item():.6f}")
        print(f"  k_beta h0p0: {k_beta[0, 0, 0].norm().item():.6f}")

        # Actually run the chunk attention to get the output
        core_attn_out, _ = la.chunk_gated_delta_rule(
            q_conv, k_conv, v_conv, g=g_raw, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        print(f"\nRecurrence output pos0 norm: {core_attn_out[0, 0].reshape(-1).float().norm().item():.6f}")
        print(f"Recurrence output pos0 all-head-dims norm: {core_attn_out[0, 0].float().norm().item():.6f}")

        # Element-level diagnostics before gated_rmsnorm
        # core_attn_out: [1, M, nv, dv] — head 0 pos 0 is [0, 0, 0, :]
        x_h0p0 = core_attn_out[0, 0, 0].float()
        z_h0p0 = z[0, 0, 0].float()
        norm_weight = la.norm.weight.float()
        print(f"\ngrmsnorm inputs (h0p0):")
        print(f"  x[0:4]=[{x_h0p0[0]:.8f},{x_h0p0[1]:.8f},{x_h0p0[2]:.8f},{x_h0p0[3]:.8f}]")
        print(f"  z[0:4]=[{z_h0p0[0]:.8f},{z_h0p0[1]:.8f},{z_h0p0[2]:.8f},{z_h0p0[3]:.8f}]")
        print(f"  weight[0:4]=[{norm_weight[0]:.8f},{norm_weight[1]:.8f},{norm_weight[2]:.8f},{norm_weight[3]:.8f}]")

        # Manual computation matching the kernel
        variance = x_h0p0.pow(2).mean()
        rms_inv = torch.rsqrt(variance + config.rms_norm_eps)
        normed_0 = x_h0p0[0] * rms_inv * norm_weight[0]
        silu_z_0 = z_h0p0[0] / (1.0 + torch.exp(-z_h0p0[0]))
        out_0 = normed_0 * silu_z_0
        print(f"  manual: rms_inv={rms_inv.item():.8f} normed[0]={normed_0.item():.8f} silu_z[0]={silu_z_0.item():.8f} out[0]={out_0.item():.8f}")

        # Gated RMSNorm
        core_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        normed_out = la.norm(core_flat, z_flat)
        normed_out = normed_out.reshape(1, M, -1)

        # Element-level: output head 0 pos 0
        out_h0p0 = normed_out[0, 0, :dv].float()
        print(f"  out[0:4]=[{out_h0p0[0]:.8f},{out_h0p0[1]:.8f},{out_h0p0[2]:.8f},{out_h0p0[3]:.8f}]")

        print(f"\nAfter gated_rmsnorm pos0 norm: {normed_out[0, 0].float().norm().item():.6f} (dim={normed_out.shape[-1]})")

        # Out projection
        mixer_out = la.out_proj(normed_out)
        print(f"After out_proj pos0 norm: {mixer_out[0, 0].float().norm().item():.6f} (dim={mixer_out.shape[-1]})")

        # Compare to reference
        ref_mixer = ref["conversations"][0]["turns"][0]["layer_norms"]["layers"][0]["mixer"]
        ref_pos0 = [n for n in ref_mixer if n["position"] == 0][0]["l2_norm"]
        print(f"\nBF16 reference mixer pos0 norm: {ref_pos0:.6f}")
        ratio = mixer_out[0, 0].float().norm().item() / ref_pos0
        print(f"Our HF / reference ratio: {ratio:.4f}")

        # Print ALL 10 reference positions
        ref_positions = ref["conversations"][0]["turns"][0]["layer_norms"]["positions"]
        print(f"\nAll positions comparison:")
        print(f"{'pos':>4s}  {'HF_now':>10s}  {'ref':>10s}  {'ratio':>8s}")
        for pos in ref_positions:
            hf_norm = mixer_out[0, pos].float().norm().item()
            ref_norm = [n for n in ref_mixer if n["position"] == pos][0]["l2_norm"]
            r = hf_norm / ref_norm if ref_norm > 0 else float('nan')
            print(f"{pos:4d}  {hf_norm:10.6f}  {ref_norm:10.6f}  {r:8.4f}")

if __name__ == "__main__":
    main()
