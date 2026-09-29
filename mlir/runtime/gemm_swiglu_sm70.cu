// Sm70 CUTLASS SwiGLU. Two Tensor Core GEMMs write FP32 workspace, then a
// separate SiluMul kernel computes silu(gate) * up. The C ABI is one call.
// Intermediate tiles stay in device workspace owned by the call.
#include "cutlass_c_api.h"
#include "cuda_utils.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <vector>

namespace {

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAcc = float;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;

using GemmSm70 = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor, ElementB, cutlass::layout::RowMajor,
    ElementC, cutlass::layout::RowMajor, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70, ThreadblockShape,
    WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc,
        ElementAcc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

__global__ void F32ToF16(const float *src, cutlass::half_t *dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<cutlass::half_t>(src[i]);
}

__global__ void SiluMul(const float *gate, const float *up, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    float s = 1.f / (1.f + expf(-g));
    out[i] = (g * s) * up[i];
  }
}

bool RunGemm(int M, int N, int K, const cutlass::half_t *A, const cutlass::half_t *B,
             float *C, cudaStream_t stream) {
  GemmSm70 gemm;
  typename GemmSm70::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},
                                    {1.f, 0.f});
  size_t ws = GemmSm70::get_workspace_size(args);
  void *workspace = nullptr;
  if (ws) CUDA_CHECK(cudaMalloc(&workspace, ws));
  cutlass::Status st = gemm.can_implement(args);
  bool ok = st == cutlass::Status::kSuccess;
  if (ok) ok = gemm.initialize(args, workspace, stream) == cutlass::Status::kSuccess;
  if (ok) ok = gemm(stream) == cutlass::Status::kSuccess;
  if (workspace) cudaFree(workspace);
  return ok;
}

} // namespace

extern "C" int cutlass_gemm_swiglu(const float *X, const float *W1, const float *W3,
                                   float *D, int64_t M, int64_t N, int64_t K) {
  cudaStream_t stream = nullptr;
  cutlass::half_t *X16 = nullptr, *W1_16 = nullptr, *W3_16 = nullptr;
  float *gate = nullptr, *up = nullptr;
  size_t mk = size_t(M) * K, kn = size_t(K) * N, mn = size_t(M) * N;
  CUDA_CHECK(cudaMalloc(&X16, mk * sizeof(*X16)));
  CUDA_CHECK(cudaMalloc(&W1_16, kn * sizeof(*W1_16)));
  CUDA_CHECK(cudaMalloc(&W3_16, kn * sizeof(*W3_16)));
  CUDA_CHECK(cudaMalloc(&gate, mn * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&up, mn * sizeof(float)));

  int block = 256;
  F32ToF16<<<(mk + block - 1) / block, block, 0, stream>>>(X, X16, mk);
  F32ToF16<<<(kn + block - 1) / block, block, 0, stream>>>(W1, W1_16, kn);
  F32ToF16<<<(kn + block - 1) / block, block, 0, stream>>>(W3, W3_16, kn);

  bool ok = RunGemm(int(M), int(N), int(K), X16, W1_16, gate, stream);
  if (ok) ok = RunGemm(int(M), int(N), int(K), X16, W3_16, up, stream);
  if (ok) {
    SiluMul<<<(mn + block - 1) / block, block, 0, stream>>>(gate, up, D, mn);
    ok = cudaGetLastError() == cudaSuccess;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(X16); cudaFree(W1_16); cudaFree(W3_16); cudaFree(gate); cudaFree(up);
  return ok ? 0 : 1;
}
