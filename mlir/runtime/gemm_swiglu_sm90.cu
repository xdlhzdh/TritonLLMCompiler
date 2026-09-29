// Hopper SwiGLU: CUTLASS 3.x CollectiveBuilder, TMA warp-specialized mainloop.
// Gate GEMM epilogue is SiLU, up GEMM is linear, then D = gate * up.
// Compiled for sm_90a. Below sm_90 the entry calls the sm_70 implementation.
// Device tests SKIP the numeric check when compute capability < 90.
#include "cutlass_c_api.h"
#include "cuda_utils.cuh"

#include "cute/tensor.hpp"
using namespace cute;

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

namespace hopper {

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementD = float;
using LayoutD = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementScalar = float;

constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 4;
constexpr int AlignmentD = 4;

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

template <class FusionOp>
struct Sm90Gemm {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
          EpilogueSchedule, FusionOp>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA,
          AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Gate: SiLU in the epilogue. Up: linear GEMM. D = gate * up in a later kernel.
using SiluFusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::SiLu, ElementD, ElementCompute, ElementC, ElementScalar>;
using LinearFusion = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementScalar>;
using GateGemm = Sm90Gemm<SiluFusion>::Gemm;
using UpGemm = Sm90Gemm<LinearFusion>::Gemm;

static_assert(sizeof(GateGemm) > 0, "Hopper gate GEMM must be instantiated");
static_assert(sizeof(UpGemm) > 0, "Hopper up GEMM must be instantiated");

__global__ void F32ToF16(const float *src, cutlass::half_t *dst, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<cutlass::half_t>(src[i]);
}

__global__ void Mul(const float *gate, const float *up, float *out, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = gate[i] * up[i];
}

template <typename Gemm>
bool RunGemm(int M, int N, int K, const cutlass::half_t *A, const cutlass::half_t *B,
             float *D, cudaStream_t stream) {
  int const L = 1;
  auto stride_A = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideD{}, cute::make_shape(M, N, L));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},
      {A, stride_A, B, stride_B},
      {{}, D, stride_C, D, stride_D},
  };
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  Gemm gemm;
  size_t ws = Gemm::get_workspace_size(args);
  void *workspace = nullptr;
  if (ws) CUDA_CHECK(cudaMalloc(&workspace, ws));
  bool ok = gemm.can_implement(args) == cutlass::Status::kSuccess &&
            gemm.initialize(args, workspace, stream) == cutlass::Status::kSuccess &&
            gemm.run(stream) == cutlass::Status::kSuccess;
  if (workspace) cudaFree(workspace);
  return ok;
}

bool DeviceIsSm90() {
  int device = 0, major = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
      cudaSuccess)
    return false;
  return major >= 9;
}

} // namespace hopper

extern "C" int cutlass_gemm_swiglu_sm90(const float *X, const float *W1,
                                        const float *W3, float *D, int64_t M,
                                        int64_t N, int64_t K) {
  // V100 / Ampere: do not launch WGMMA. The default C ABI covers numerics.
  if (!hopper::DeviceIsSm90())
    return cutlass_gemm_swiglu(X, W1, W3, D, M, N, K);

  cudaStream_t stream = nullptr;
  cutlass::half_t *X16 = nullptr, *W1_16 = nullptr, *W3_16 = nullptr;
  float *gate = nullptr, *up = nullptr;
  size_t mk = static_cast<size_t>(M) * K;
  size_t kn = static_cast<size_t>(K) * N;
  size_t mn = static_cast<size_t>(M) * N;
  CUDA_CHECK(cudaMalloc(&X16, mk * sizeof(*X16)));
  CUDA_CHECK(cudaMalloc(&W1_16, kn * sizeof(*W1_16)));
  CUDA_CHECK(cudaMalloc(&W3_16, kn * sizeof(*W3_16)));
  CUDA_CHECK(cudaMalloc(&gate, mn * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&up, mn * sizeof(float)));
  int block = 256;
  hopper::F32ToF16<<<(static_cast<int>(mk) + block - 1) / block, block>>>(X, X16, mk);
  hopper::F32ToF16<<<(static_cast<int>(kn) + block - 1) / block, block>>>(W1, W1_16, kn);
  hopper::F32ToF16<<<(static_cast<int>(kn) + block - 1) / block, block>>>(W3, W3_16, kn);

  bool ok = hopper::RunGemm<hopper::GateGemm>(int(M), int(N), int(K), X16, W1_16, gate, stream);
  if (ok)
    ok = hopper::RunGemm<hopper::UpGemm>(int(M), int(N), int(K), X16, W3_16, up, stream);
  if (ok) {
    hopper::Mul<<<(static_cast<int>(mn) + block - 1) / block, block, 0, stream>>>(gate, up, D, mn);
    ok = cudaGetLastError() == cudaSuccess;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(X16);
  cudaFree(W1_16);
  cudaFree(W3_16);
  cudaFree(gate);
  cudaFree(up);
  return ok ? 0 : 1;
}

#else

extern "C" int cutlass_gemm_swiglu_sm90(const float *X, const float *W1,
                                        const float *W3, float *D, int64_t M,
                                        int64_t N, int64_t K) {
  return cutlass_gemm_swiglu(X, W1, W3, D, M, N, K);
}

#endif
