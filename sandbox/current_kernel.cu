/*
 * Starter scaffold: INT4 → FP16 fused dequantization matmul.
 *
 * This is a naive reference implementation intentionally left for the agent
 * to replace. It is mathematically correct but not optimised.
 *
 * AWQ INT4 layout
 * ---------------
 *   weight_packed : [out_features, in_features / 8]  int32
 *     8 INT4 nibbles packed per int32, lower nibble first.
 *   scales        : [out_features, in_features / group_size]  fp16
 *   zeros         : [out_features, in_features / group_size]  fp16
 *   group_size    : 128
 *
 * Dequant formula
 * ---------------
 *   w_fp16 = (int4_val - zero_point) * scale
 *
 * Output
 * ------
 *   out : [batch_size, out_features]  fp16
 *   out = x @ W^T   where W is the dequantised weight matrix
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static const int GROUP_SIZE = 128;

__global__ void dequant_matmul_kernel(
    const __half* __restrict__ x,            // [batch, in_feat]
    const int32_t* __restrict__ weight_packed, // [out_feat, in_feat/8]
    const __half* __restrict__ scales,       // [out_feat, in_feat/group]
    const __half* __restrict__ zeros,        // [out_feat, in_feat/group]
    __half* __restrict__ out,                // [batch, out_feat]
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // out_features index

    if (row >= batch_size || col >= out_features) return;

    int n_groups = in_features / GROUP_SIZE;
    float acc = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        float scale = __half2float(scales[col * n_groups + g]);
        float zero  = __half2float(zeros [col * n_groups + g]);

        int group_start = g * GROUP_SIZE;

        // Unpack 8 INT4 values per packed int32
        for (int packed_idx = group_start / 8; packed_idx < (group_start + GROUP_SIZE) / 8; ++packed_idx) {
            int32_t packed = weight_packed[col * (in_features / 8) + packed_idx];

            #pragma unroll
            for (int bit = 0; bit < 8; ++bit) {
                int in_col = packed_idx * 8 + bit;
                if (in_col >= in_features) break;

                int   int4_val  = (packed >> (bit * 4)) & 0xF;
                float w_fp32    = ((float)int4_val - zero) * scale;
                float x_fp32    = __half2float(x[row * in_features + in_col]);
                acc += x_fp32 * w_fp32;
            }
        }
    }

    out[row * out_features + col] = __float2half(acc);
}


/* Launcher — called from binding.cpp */
void launch_dequant_matmul(
    const __half* x,
    const int32_t* weight_packed,
    const __half* scales,
    const __half* zeros,
    __half* out,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (batch_size   + block.x - 1) / block.x,
        (out_features + block.y - 1) / block.y
    );
    dequant_matmul_kernel<<<grid, block, 0, stream>>>(
        x, weight_packed, scales, zeros, out,
        batch_size, in_features, out_features
    );
}
