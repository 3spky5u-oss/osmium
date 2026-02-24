//! CPU decode compute kernels for non-MoE layers.
//!
//! Provides quantized INT4/INT8 matmul, RMSNorm, and SiLU for CPU-only decode.
//! Weights are quantized once at prepare() time, then reused for all decode steps.
//! Activations are f32, quantized to INT16 per-call before each matmul.

use crate::kernel::avx2::{
    matmul_int4_transposed_integer, matmul_int4_transposed_integer_parallel,
    matmul_int8_transposed_integer, matmul_int8_transposed_integer_parallel,
    quantize_activation_int16_f32,
};
use crate::weights::marlin::f32_to_bf16;
use pyo3::prelude::*;

/// A single quantized weight matrix in transposed format for CPU decode.
struct TransposedWeight {
    /// Packed weight data (transposed).
    /// INT4: [K/8, N] as u32 (8 nibbles per u32)
    /// INT8: [K, N] as i8 packed into u32 container
    packed: Vec<u32>,
    /// Per-group scales in BF16 (transposed). [K/group_size, N]
    scales: Vec<u16>,
    /// Output dimension (N = rows of original weight).
    rows: usize,
    /// Input dimension (K = cols of original weight).
    cols: usize,
    group_size: usize,
    num_bits: u8,
}

/// Quantize f32 weight matrix [N, K] to transposed INT4 format.
///
/// INT4 symmetric: values mapped to [-8, 7], 8 packed per u32.
/// Output layout: packed [K/8, N], scales [K/gs, N] (both transposed).
fn quantize_f32_to_transposed_int4(
    weight: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> TransposedWeight {
    assert_eq!(weight.len(), rows * cols);
    assert!(cols % group_size == 0, "cols {} must be divisible by group_size {}", cols, group_size);
    assert!(cols % 8 == 0, "cols {} must be divisible by 8 for INT4", cols);
    assert!(group_size % 8 == 0);

    let packed_k = cols / 8;
    let num_groups = cols / group_size;

    // Quantize in row-major [N, K/8] packed, [N, K/gs] scales
    let mut packed_rm = vec![0u32; rows * packed_k];
    let mut scales_rm = vec![0u16; rows * num_groups];

    for row in 0..rows {
        let row_base = row * cols;
        for g in 0..num_groups {
            let g_start = g * group_size;

            let mut max_abs: f32 = 0.0;
            for i in 0..group_size {
                max_abs = max_abs.max(weight[row_base + g_start + i].abs());
            }

            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = if max_abs > 0.0 { 7.0 / max_abs } else { 0.0 };
            scales_rm[row * num_groups + g] = f32_to_bf16(scale);

            for pack in 0..(group_size / 8) {
                let base = g_start + pack * 8;
                let mut word: u32 = 0;
                for j in 0..8u32 {
                    let val = weight[row_base + base + j as usize];
                    let q = ((val * inv_scale).round() as i32).clamp(-8, 7);
                    let u4 = (q + 8) as u32;
                    word |= u4 << (j * 4);
                }
                packed_rm[row * packed_k + g * (group_size / 8) + pack] = word;
            }
        }
    }

    // Transpose packed: [N, K/8] -> [K/8, N]
    let mut packed = vec![0u32; packed_k * rows];
    for k in 0..packed_k {
        for n in 0..rows {
            packed[k * rows + n] = packed_rm[n * packed_k + k];
        }
    }

    // Transpose scales: [N, K/gs] -> [K/gs, N]
    let mut scales = vec![0u16; num_groups * rows];
    for g in 0..num_groups {
        for n in 0..rows {
            scales[g * rows + n] = scales_rm[n * num_groups + g];
        }
    }

    TransposedWeight { packed, scales, rows, cols, group_size, num_bits: 4 }
}

/// Quantize f32 weight matrix [N, K] to transposed INT8 format.
///
/// INT8 symmetric: values mapped to [-127, 127], stored as i8 in u32 container.
/// Output layout: data [K, N] as i8 in u32, scales [K/gs, N] (both transposed).
fn quantize_f32_to_transposed_int8(
    weight: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> TransposedWeight {
    assert_eq!(weight.len(), rows * cols);
    assert!(cols % group_size == 0, "cols {} must be divisible by group_size {}", cols, group_size);
    assert!(group_size % 2 == 0);

    let num_groups = cols / group_size;

    // Quantize in row-major
    let mut data_rm = vec![0i8; rows * cols];
    let mut scales_rm = vec![0u16; rows * num_groups];

    for row in 0..rows {
        let row_base = row * cols;
        for g in 0..num_groups {
            let g_start = g * group_size;
            let mut max_abs: f32 = 0.0;
            for i in 0..group_size {
                max_abs = max_abs.max(weight[row_base + g_start + i].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
            scales_rm[row * num_groups + g] = f32_to_bf16(scale);
            for i in 0..group_size {
                let val = weight[row_base + g_start + i];
                data_rm[row_base + g_start + i] =
                    ((val * inv_scale).round() as i32).clamp(-128, 127) as i8;
            }
        }
    }

    // Transpose data: [N, K] -> [K, N] as i8, packed into Vec<u32>
    let byte_count = cols * rows;
    let u32_count = (byte_count + 3) / 4;
    let mut transposed_bytes = vec![0i8; u32_count * 4];
    for k in 0..cols {
        for n in 0..rows {
            transposed_bytes[k * rows + n] = data_rm[n * cols + k];
        }
    }
    let packed: Vec<u32> = unsafe {
        let mut v = vec![0u32; u32_count];
        std::ptr::copy_nonoverlapping(
            transposed_bytes.as_ptr() as *const u8,
            v.as_mut_ptr() as *mut u8,
            u32_count * 4,
        );
        v
    };

    // Transpose scales: [N, K/gs] -> [K/gs, N]
    let mut scales = vec![0u16; num_groups * rows];
    for g in 0..num_groups {
        for n in 0..rows {
            scales[g * rows + n] = scales_rm[n * num_groups + g];
        }
    }

    TransposedWeight { packed, scales, rows, cols, group_size, num_bits: 8 }
}

/// CPU decode weight store — holds quantized non-MoE weights for fast matmul.
#[pyclass]
pub struct CpuDecodeStore {
    weights: Vec<TransposedWeight>,
    /// Scratch buffer for INT16 activation quantization (reused across calls).
    act_int16: Vec<i16>,
    act_scales: Vec<f32>,
    /// Current scratch size (max K seen so far).
    scratch_k: usize,
    group_size: usize,
    /// Whether to use parallel (multi-threaded) matmul for large outputs.
    parallel: bool,
    /// Whether norms use (1+w)*x instead of w*x (Qwen3-Next).
    norm_bias_one: bool,
}

#[pymethods]
impl CpuDecodeStore {
    #[new]
    #[pyo3(signature = (group_size=128, parallel=true, norm_bias_one=false))]
    pub fn new(group_size: usize, parallel: bool, norm_bias_one: bool) -> Self {
        CpuDecodeStore {
            weights: Vec::new(),
            act_int16: Vec::new(),
            act_scales: Vec::new(),
            scratch_k: 0,
            group_size,
            parallel,
            norm_bias_one,
        }
    }

    /// Store a weight matrix from f32 data. Returns weight ID.
    ///
    /// Args:
    ///   data_ptr: pointer to f32 [rows, cols] row-major
    ///   rows: output dimension (N)
    ///   cols: input dimension (K)
    ///   num_bits: 4 or 8
    pub fn store_weight_f32(
        &mut self,
        data_ptr: usize,
        rows: usize,
        cols: usize,
        num_bits: u8,
    ) -> PyResult<usize> {
        if num_bits != 4 && num_bits != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("num_bits must be 4 or 8, got {}", num_bits)));
        }
        if cols % self.group_size != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("cols {} must be divisible by group_size {}", cols, self.group_size)));
        }
        if num_bits == 4 && cols % 8 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("cols {} must be divisible by 8 for INT4", cols)));
        }

        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_ptr as *const f32, rows * cols)
        };

        let weight = match num_bits {
            4 => quantize_f32_to_transposed_int4(data, rows, cols, self.group_size),
            8 => quantize_f32_to_transposed_int8(data, rows, cols, self.group_size),
            _ => unreachable!(),
        };

        // Grow scratch buffers if needed
        if cols > self.scratch_k {
            self.scratch_k = cols;
            self.act_int16 = vec![0i16; cols];
            self.act_scales = vec![0f32; cols / self.group_size];
        }

        let id = self.weights.len();
        let bytes = weight.packed.len() * 4 + weight.scales.len() * 2;
        self.weights.push(weight);
        log::debug!("Stored weight {}: [{}x{}] INT{} transposed, {:.1} KB",
            id, rows, cols, num_bits, bytes as f64 / 1024.0);
        Ok(id)
    }

    /// Matrix-vector multiply: output[N] = W[N,K] @ input[K]
    ///
    /// Input is f32, internally quantized to INT16. Output is f32.
    pub fn matmul(
        &mut self,
        weight_id: usize,
        input_ptr: usize,
        output_ptr: usize,
    ) -> PyResult<()> {
        if weight_id >= self.weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("weight_id {} out of range ({})", weight_id, self.weights.len())));
        }
        let w = &self.weights[weight_id];
        let k = w.cols;
        let n = w.rows;
        let gs = w.group_size;

        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, k)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, n)
        };

        // Quantize input to INT16
        quantize_activation_int16_f32(
            input, gs, &mut self.act_int16[..k], &mut self.act_scales[..k / gs]);

        self.dispatch_matmul(weight_id, &self.act_int16[..k], &self.act_scales[..k / gs], output);
        Ok(())
    }

    /// Batch matmul: quantize input once, run multiple matmuls.
    ///
    /// All weights must have the same input dimension (K).
    /// weight_ids: list of weight IDs
    /// input_ptr: f32 [K]
    /// output_ptrs: list of f32 output pointers
    pub fn matmul_batch(
        &mut self,
        weight_ids: Vec<usize>,
        input_ptr: usize,
        output_ptrs: Vec<usize>,
    ) -> PyResult<()> {
        if weight_ids.len() != output_ptrs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weight_ids and output_ptrs must have same length"));
        }
        if weight_ids.is_empty() {
            return Ok(());
        }

        let k = self.weights[weight_ids[0]].cols;
        let gs = self.weights[weight_ids[0]].group_size;

        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, k)
        };

        // Quantize input once
        quantize_activation_int16_f32(
            input, gs, &mut self.act_int16[..k], &mut self.act_scales[..k / gs]);

        for i in 0..weight_ids.len() {
            let wid = weight_ids[i];
            let w = &self.weights[wid];
            assert_eq!(w.cols, k, "All weights in batch must have same K");
            let n = w.rows;
            let output: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(output_ptrs[i] as *mut f32, n)
            };
            self.dispatch_matmul(wid, &self.act_int16[..k], &self.act_scales[..k / gs], output);
        }
        Ok(())
    }

    /// Fused add + RMSNorm (in-place on both buffers).
    ///
    /// If first_call: residual = hidden, hidden = rmsnorm(residual)
    /// Else: residual += hidden, hidden = rmsnorm(residual)
    pub fn fused_add_rmsnorm(
        &self,
        hidden_ptr: usize,
        residual_ptr: usize,
        weight_ptr: usize,
        eps: f32,
        size: usize,
        first_call: bool,
    ) -> PyResult<()> {
        let hidden: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(hidden_ptr as *mut f32, size)
        };
        let residual: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(residual_ptr as *mut f32, size)
        };
        let weight: &[f32] = unsafe {
            std::slice::from_raw_parts(weight_ptr as *const f32, size)
        };

        if first_call {
            residual.copy_from_slice(hidden);
        } else {
            for i in 0..size {
                residual[i] += hidden[i];
            }
        }

        // RMSNorm
        let mut sum_sq: f32 = 0.0;
        for i in 0..size {
            sum_sq += residual[i] * residual[i];
        }
        let rms = (sum_sq / size as f32 + eps).sqrt().recip();

        if self.norm_bias_one {
            for i in 0..size {
                hidden[i] = residual[i] * rms * (1.0 + weight[i]);
            }
        } else {
            for i in 0..size {
                hidden[i] = residual[i] * rms * weight[i];
            }
        }

        Ok(())
    }

    /// Standalone RMSNorm (non-fused).
    pub fn rmsnorm(
        &self,
        input_ptr: usize,
        weight_ptr: usize,
        eps: f32,
        output_ptr: usize,
        size: usize,
    ) -> PyResult<()> {
        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, size)
        };
        let weight: &[f32] = unsafe {
            std::slice::from_raw_parts(weight_ptr as *const f32, size)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, size)
        };

        let mut sum_sq: f32 = 0.0;
        for i in 0..size {
            sum_sq += input[i] * input[i];
        }
        let rms = (sum_sq / size as f32 + eps).sqrt().recip();

        if self.norm_bias_one {
            for i in 0..size {
                output[i] = input[i] * rms * (1.0 + weight[i]);
            }
        } else {
            for i in 0..size {
                output[i] = input[i] * rms * weight[i];
            }
        }

        Ok(())
    }

    /// SiLU(gate) * up -> output, elementwise.
    pub fn silu_mul(
        &self,
        gate_ptr: usize,
        up_ptr: usize,
        output_ptr: usize,
        size: usize,
    ) -> PyResult<()> {
        let gate: &[f32] = unsafe {
            std::slice::from_raw_parts(gate_ptr as *const f32, size)
        };
        let up: &[f32] = unsafe {
            std::slice::from_raw_parts(up_ptr as *const f32, size)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, size)
        };

        for i in 0..size {
            let x = gate[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            output[i] = x * sigmoid * up[i];
        }

        Ok(())
    }

    /// Fused shared expert: gate_up_matmul → SiLU*mul → down_matmul.
    ///
    /// Does the full shared expert MLP in one Rust call, avoiding 3 FFI round-trips.
    /// input: f32 [K], gate_up_wid: fused [2*intermediate, K], down_wid: [K, intermediate]
    /// output: f32 [K] (same dim as input, since down_proj maps back to hidden)
    pub fn fused_shared_expert(
        &mut self,
        gate_up_wid: usize,
        down_wid: usize,
        input_ptr: usize,
        output_ptr: usize,
    ) -> PyResult<()> {
        // gate_up matmul: [2*intermediate] = gate_up_W @ input
        let gu_w = &self.weights[gate_up_wid];
        let k_in = gu_w.cols;
        let n_gu = gu_w.rows; // 2 * intermediate
        let gs = gu_w.group_size;
        let intermediate = n_gu / 2;

        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, k_in)
        };

        // Quantize input once for gate_up
        quantize_activation_int16_f32(
            input, gs, &mut self.act_int16[..k_in], &mut self.act_scales[..k_in / gs]);

        // gate_up matmul
        let mut gate_up = vec![0f32; n_gu];
        self.dispatch_matmul_ext(gate_up_wid, &self.act_int16[..k_in], &self.act_scales[..k_in / gs], &mut gate_up);

        // SiLU(gate) * up → hidden
        let mut se_hidden = vec![0f32; intermediate];
        for i in 0..intermediate {
            let x = gate_up[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            se_hidden[i] = x * sigmoid * gate_up[intermediate + i];
        }

        // down matmul: quantize se_hidden, then matmul
        let d_w = &self.weights[down_wid];
        let k_down = d_w.cols;
        let n_down = d_w.rows;
        let gs_down = d_w.group_size;

        // Grow scratch if needed for down proj input
        if k_down > self.scratch_k {
            self.scratch_k = k_down;
            self.act_int16 = vec![0i16; k_down];
            self.act_scales = vec![0f32; k_down / gs_down];
        }

        quantize_activation_int16_f32(
            &se_hidden, gs_down, &mut self.act_int16[..k_down], &mut self.act_scales[..k_down / gs_down]);

        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, n_down)
        };
        self.dispatch_matmul_ext(down_wid, &self.act_int16[..k_down], &self.act_scales[..k_down / gs_down], output);

        Ok(())
    }

    /// Gated DeltaNet recurrent state update + query output.
    ///
    /// state: [nv, dk, dv] f32 (modified in-place)
    /// q: [nv, dk] f32 (already L2-normalized and scaled, with heads expanded)
    /// k: [nv, dk] f32 (already L2-normalized, with heads expanded)
    /// v: [nv, dv] f32
    /// g: [nv] f32 (decay = exp(-A * softplus(a + dt_bias)), already computed)
    /// beta: [nv] f32 (sigmoid already applied)
    /// output: [nv, dv] f32 (query @ state result)
    pub fn linear_attention_recurrent(
        &self,
        state_ptr: usize,
        q_ptr: usize,
        k_ptr: usize,
        v_ptr: usize,
        g_ptr: usize,
        beta_ptr: usize,
        output_ptr: usize,
        nv: usize,
        dk: usize,
        dv: usize,
    ) -> PyResult<()> {
        let state: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(state_ptr as *mut f32, nv * dk * dv)
        };
        let q: &[f32] = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, nv * dk) };
        let k: &[f32] = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, nv * dk) };
        let v: &[f32] = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, nv * dv) };
        let g: &[f32] = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, nv) };
        let beta: &[f32] = unsafe { std::slice::from_raw_parts(beta_ptr as *const f32, nv) };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, nv * dv)
        };

        // For each value head:
        //   1. Decay: state[h, :, :] *= exp(g[h])
        //   2. kv_mem[dv] = state[h, :, :].T @ k[h, :] (sum over dk)
        //   3. delta[dv] = (v[h, :] - kv_mem) * beta[h]
        //   4. state[h, :, :] += k[h, :].outer(delta)
        //   5. output[h, dv] = state[h, :, :].T @ q[h, :] (sum over dk)

        // Scratch buffers for kv_mem, delta, and output accumulation
        let mut kv_mem = vec![0.0f32; dv];
        let mut delta = vec![0.0f32; dv];
        let mut out_buf = vec![0.0f32; dv];

        for h in 0..nv {
            let g_exp = g[h].exp();
            let beta_h = beta[h];
            let s_base = h * dk * dv;
            let q_base = h * dk;
            let k_base = h * dk;
            let v_base = h * dv;
            let o_base = h * dv;

            // Zero scratch
            for j in 0..dv { kv_mem[j] = 0.0; out_buf[j] = 0.0; }

            // Pass 1: Decay state + compute kv_mem (cache-friendly: row-major access)
            for i in 0..dk {
                let row_base = s_base + i * dv;
                let k_val = k[k_base + i];
                for j in 0..dv {
                    state[row_base + j] *= g_exp;
                    kv_mem[j] += state[row_base + j] * k_val;
                }
            }

            // Compute delta[j] = (v[j] - kv_mem[j]) * beta
            for j in 0..dv {
                delta[j] = (v[v_base + j] - kv_mem[j]) * beta_h;
            }

            // Pass 2: State update + output accumulation (cache-friendly)
            for i in 0..dk {
                let row_base = s_base + i * dv;
                let k_val = k[k_base + i];
                let q_val = q[q_base + i];
                for j in 0..dv {
                    state[row_base + j] += k_val * delta[j];
                    out_buf[j] += state[row_base + j] * q_val;
                }
            }

            // Write output
            output[o_base..o_base + dv].copy_from_slice(&out_buf[..dv]);
        }

        Ok(())
    }

    /// Gated RMSNorm + SiLU gate: out = SiLU(z) * RMSNorm(x, weight)
    ///
    /// x: [nv * dv] f32 (recurrent output)
    /// z: [nv * dv] f32 (gate signal from projection)
    /// norm_weight: [nv, dv] or [nv * dv] f32
    /// output: [nv * dv] f32
    /// eps: RMSNorm epsilon
    /// nv: number of value heads (norm is per-head)
    /// dv: value head dimension
    pub fn gated_rmsnorm_silu(
        &self,
        x_ptr: usize,
        z_ptr: usize,
        norm_weight_ptr: usize,
        output_ptr: usize,
        eps: f32,
        nv: usize,
        dv: usize,
    ) -> PyResult<()> {
        let size = nv * dv;
        let x: &[f32] = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, size) };
        let z: &[f32] = unsafe { std::slice::from_raw_parts(z_ptr as *const f32, size) };
        let norm_weight: &[f32] = unsafe {
            std::slice::from_raw_parts(norm_weight_ptr as *const f32, size)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, size)
        };

        // Per-head RMSNorm: for each head h, norm over dv dimensions
        for h in 0..nv {
            let base = h * dv;
            let mut sum_sq = 0.0f32;
            for j in 0..dv {
                sum_sq += x[base + j] * x[base + j];
            }
            let rms = (sum_sq / dv as f32 + eps).sqrt().recip();

            for j in 0..dv {
                let normed = x[base + j] * rms * norm_weight[base + j];
                // SiLU(z) * normed
                let zval = z[base + j];
                let silu_z = zval / (1.0 + (-zval).exp());
                output[base + j] = silu_z * normed;
            }
        }

        Ok(())
    }

    /// Number of stored weights.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Total bytes used by stored weights.
    pub fn total_bytes(&self) -> usize {
        self.weights.iter().map(|w| {
            w.packed.len() * 4 + w.scales.len() * 2
        }).sum()
    }

    /// Bytes used by a single weight matrix.
    pub fn weight_bytes(&self, weight_id: usize) -> usize {
        let w = &self.weights[weight_id];
        w.packed.len() * 4 + w.scales.len() * 2
    }
}

// Private helper (not exposed to Python)
impl CpuDecodeStore {
    /// Dispatch matmul to correct INT4/INT8 kernel (uses provided buffers).
    fn dispatch_matmul_ext(&self, weight_id: usize, act_int16: &[i16], act_scales: &[f32], output: &mut [f32]) {
        self.dispatch_matmul(weight_id, act_int16, act_scales, output);
    }

    /// Dispatch matmul to correct INT4/INT8 kernel.
    fn dispatch_matmul(&self, weight_id: usize, act_int16: &[i16], act_scales: &[f32], output: &mut [f32]) {
        let w = &self.weights[weight_id];
        let k = w.cols;
        let n = w.rows;
        let gs = w.group_size;

        match w.num_bits {
            4 => {
                if self.parallel && n > 64 {
                    matmul_int4_transposed_integer_parallel(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                } else {
                    matmul_int4_transposed_integer(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                }
            }
            8 => {
                if self.parallel && n > 64 {
                    matmul_int8_transposed_integer_parallel(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                } else {
                    matmul_int8_transposed_integer(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                }
            }
            _ => unreachable!(),
        }
    }
}
