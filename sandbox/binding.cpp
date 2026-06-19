/*
 * pybind11 + PyTorch C++ extension wrapper.
 *
 * Exposes dequantize_matmul() to Python as:
 *   output = cuda_kernel.dequantize_matmul(x, weight_packed, scales, zeros)
 *
 * Arguments
 * ---------
 *   x             : torch.Tensor  [batch_size, in_features]  float16, CUDA
 *   weight_packed : torch.Tensor  [out_features, in_features/8]  int32, CUDA
 *   scales        : torch.Tensor  [out_features, in_features/group_size]  float16, CUDA
 *   zeros         : torch.Tensor  [out_features, in_features/group_size]  float16, CUDA
 *
 * Returns
 * -------
 *   torch.Tensor  [batch_size, out_features]  float16, CUDA
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdexcept>

/* Declared in current_kernel.cu */
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
);

torch::Tensor dequantize_matmul(
    torch::Tensor x,
    torch::Tensor weight_packed,
    torch::Tensor scales,
    torch::Tensor zeros
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(),             "x must be a CUDA tensor");
    TORCH_CHECK(weight_packed.is_cuda(), "weight_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(),        "scales must be a CUDA tensor");
    TORCH_CHECK(zeros.is_cuda(),         "zeros must be a CUDA tensor");

    TORCH_CHECK(x.dtype()             == torch::kFloat16, "x must be float16");
    TORCH_CHECK(weight_packed.dtype() == torch::kInt32,   "weight_packed must be int32");
    TORCH_CHECK(scales.dtype()        == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype()         == torch::kFloat16, "zeros must be float16");

    TORCH_CHECK(x.dim() == 2,             "x must be 2D [batch, in_features]");
    TORCH_CHECK(weight_packed.dim() == 2, "weight_packed must be 2D [out, in/8]");

    const int batch_size   = x.size(0);
    const int in_features  = x.size(1);
    const int out_features = weight_packed.size(0);

    TORCH_CHECK(weight_packed.size(1) == in_features / 8,
        "weight_packed dim1 must equal in_features/8");

    // Allocate output
    auto out = torch::empty(
        {batch_size, out_features},
        torch::TensorOptions().dtype(torch::kFloat16).device(x.device())
    );

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_dequant_matmul(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        weight_packed.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        batch_size,
        in_features,
        out_features,
        stream
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "INT4 AWQ dequantize + matmul CUDA kernel";
    m.def(
        "dequantize_matmul",
        &dequantize_matmul,
        "Fused INT4 AWQ dequantization + FP16 matrix multiplication",
        py::arg("x"),
        py::arg("weight_packed"),
        py::arg("scales"),
        py::arg("zeros")
    );
}
