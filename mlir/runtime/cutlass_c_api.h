#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// SwiGLU: D = silu(X@W1) * (X@W3). Row-major FP32 I/O, FP16 tensor-core math.
/// SM70 runs two GEMMs and a separate SiLU*mul kernel. Returns 0 on success.
int cutlass_gemm_swiglu(const float *X, const float *W1, const float *W3, float *D,
                        int64_t M, int64_t N, int64_t K);

/// SM90 CollectiveBuilder SwiGLU. Gate GEMM uses a SiLU epilogue, up GEMM is
/// linear, then D = gate * up. Below sm_90 this calls cutlass_gemm_swiglu.
int cutlass_gemm_swiglu_sm90(const float *X, const float *W1, const float *W3,
                             float *D, int64_t M, int64_t N, int64_t K);

/// Fused RMSNorm + RoPE. X/D: [B,S,H,D] row-major; weight [D]; cos/sin [S,D].
int cutlass_rmsnorm_rope(const float *X, const float *weight, const float *cos,
                         const float *sin, float *D, float eps, int64_t B,
                         int64_t S, int64_t H, int64_t Dim);

#ifdef __cplusplus
}
#endif
