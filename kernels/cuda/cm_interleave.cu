/**
 * DEF-cmssm: Cross-Modal Interleave/De-interleave + Fused SS2D ops
 *
 * 1. cm_interleave — Interleave RGB and thermal tokens for joint Mamba scan
 * 2. cm_deinterleave — Separate interleaved output back to RGB and thermal
 * 3. fused_gate_norm — Fused LayerNorm + SiLU gate + out_proj
 *
 * These are the memory-bound ops in the SS2D_rgbt forward pass.
 * The interleave pattern (stack→view→cat→flip) involves ~8 memory copies.
 * This kernel does it in a single pass.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Interleave kernel: rgb[B, D, L] + thermal[B, D, L] -> out[B, D, 2*L]
 * Output pattern: [rgb_0, t_0, rgb_1, t_1, ..., rgb_{L-1}, t_{L-1}]
 */
__global__ void cm_interleave_kernel(
    const float* __restrict__ rgb,
    const float* __restrict__ thermal,
    float* __restrict__ output,
    int B, int D, int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * L;
    if (idx >= total) return;

    int b = idx / (D * L);
    int rem = idx % (D * L);
    int d = rem / L;
    int l = rem % L;

    int out_base = b * D * (2 * L) + d * (2 * L);
    output[out_base + 2 * l + 0] = rgb[idx];
    output[out_base + 2 * l + 1] = thermal[idx];
}

/**
 * De-interleave kernel: input[B, D, 2*L] -> rgb[B, D, L] + thermal[B, D, L]
 * Reverses the interleave: even indices → RGB, odd indices → thermal
 */
__global__ void cm_deinterleave_kernel(
    const float* __restrict__ input,
    float* __restrict__ rgb,
    float* __restrict__ thermal,
    int B, int D, int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * L;
    if (idx >= total) return;

    int b = idx / (D * L);
    int rem = idx % (D * L);
    int d = rem / L;
    int l = rem % L;

    int in_base = b * D * (2 * L) + d * (2 * L);
    rgb[idx] = input[in_base + 2 * l + 0];
    thermal[idx] = input[in_base + 2 * l + 1];
}

/**
 * Fused gate + norm + output projection.
 * Computes: output = out_proj(layernorm(x) * silu(gate))
 *
 * This fuses 3 operations that normally require 3 separate kernel launches
 * and 3 memory round-trips into a single pass.
 *
 * x: (B, D, L) — from selective scan
 * gate: (B, D, L) — from input projection residual branch
 * weight, bias: (D,) — layernorm parameters
 * output: (B, D, L) — gated output
 */
__global__ void fused_gate_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ ln_weight,
    const float* __restrict__ ln_bias,
    float* __restrict__ output,
    int B, int D, int L,
    float eps
) {
    // Each block handles one (b, l) position across all D channels
    int bl_idx = blockIdx.x;
    if (bl_idx >= B * L) return;

    int b = bl_idx / L;
    int l = bl_idx % L;

    // Compute mean and variance for this (b, l) across D channels
    float sum = 0.0f, sum_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = x[b * D * L + d * L + l];
        sum += val;
        sum_sq += val * val;
    }

    // Warp reduce
    __shared__ float s_sum, s_sum_sq;
    if (threadIdx.x == 0) { s_sum = 0; s_sum_sq = 0; }
    __syncthreads();
    atomicAdd(&s_sum, sum);
    atomicAdd(&s_sum_sq, sum_sq);
    __syncthreads();

    float mean = s_sum / D;
    float var = s_sum_sq / D - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Apply norm + gate + write
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int idx = b * D * L + d * L + l;
        float normed = (x[idx] - mean) * inv_std * ln_weight[d] + ln_bias[d];
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));  // SiLU(gate)
        output[idx] = normed * silu_g;
    }
}


// --- PyTorch bindings ---

torch::Tensor cm_interleave(torch::Tensor rgb, torch::Tensor thermal) {
    TORCH_CHECK(rgb.is_cuda() && thermal.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(rgb.sizes() == thermal.sizes(), "shape mismatch");

    int B = rgb.size(0), D = rgb.size(1), L = rgb.size(2);
    auto output = torch::empty({B, D, 2 * L}, rgb.options());

    int total = B * D * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cm_interleave_kernel<<<blocks, threads>>>(
        rgb.data_ptr<float>(), thermal.data_ptr<float>(),
        output.data_ptr<float>(), B, D, L
    );
    return output;
}

std::vector<torch::Tensor> cm_deinterleave(torch::Tensor input, int L) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    int B = input.size(0), D = input.size(1);

    auto rgb = torch::empty({B, D, L}, input.options());
    auto thermal = torch::empty({B, D, L}, input.options());

    int total = B * D * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cm_deinterleave_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        rgb.data_ptr<float>(), thermal.data_ptr<float>(),
        B, D, L
    );
    return {rgb, thermal};
}

torch::Tensor fused_gate_norm(
    torch::Tensor x,
    torch::Tensor gate,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    int B = x.size(0), D = x.size(1), L = x.size(2);

    auto output = torch::empty_like(x);
    int bl = B * L;
    int threads = min(D, 256);

    fused_gate_norm_kernel<<<bl, threads>>>(
        x.data_ptr<float>(), gate.data_ptr<float>(),
        ln_weight.data_ptr<float>(), ln_bias.data_ptr<float>(),
        output.data_ptr<float>(), B, D, L, eps
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cm_interleave", &cm_interleave,
          "Cross-modal interleave: rgb[B,D,L] + thermal[B,D,L] -> [B,D,2L] (CUDA)");
    m.def("cm_deinterleave", &cm_deinterleave,
          "Cross-modal de-interleave: [B,D,2L] -> [rgb, thermal] each [B,D,L] (CUDA)");
    m.def("fused_gate_norm", &fused_gate_norm,
          "Fused LayerNorm + SiLU gate for SS2D output (CUDA)");
}
