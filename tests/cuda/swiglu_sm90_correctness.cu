#include "cutlass_c_api.h"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

static bool DeviceSupportsSm90() {
  int device = 0, major = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
      cudaSuccess)
    return false;
  return major >= 9;
}

int main() {
  if (!DeviceSupportsSm90()) {
    std::printf("SKIP: device compute capability < 90 (Hopper reference only)\n");
    return 0;
  }

  const int M = 64, N = 64, K = 64;
  std::vector<float> hX(M * K), hW1(K * N), hW3(K * N), hD(M * N), hRef(M * N);
  for (int i = 0; i < M * K; ++i) hX[i] = ((i % 7) - 3) * 0.05f;
  for (int i = 0; i < K * N; ++i) {
    hW1[i] = ((i % 5) - 2) * 0.05f;
    hW3[i] = ((i % 3) - 1) * 0.05f;
  }
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float gate = 0.f, up = 0.f;
      for (int k = 0; k < K; ++k) {
        gate += hX[m * K + k] * hW1[k * N + n];
        up += hX[m * K + k] * hW3[k * N + n];
      }
      float silu = gate / (1.f + std::exp(-gate));
      hRef[m * N + n] = silu * up;
    }
  }

  float *dX = nullptr, *dW1 = nullptr, *dW3 = nullptr, *dD = nullptr;
  cudaMalloc(&dX, hX.size() * sizeof(float));
  cudaMalloc(&dW1, hW1.size() * sizeof(float));
  cudaMalloc(&dW3, hW3.size() * sizeof(float));
  cudaMalloc(&dD, hD.size() * sizeof(float));
  cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dW1, hW1.data(), hW1.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dW3, hW3.data(), hW3.size() * sizeof(float), cudaMemcpyHostToDevice);

  int rc = cutlass_gemm_swiglu_sm90(dX, dW1, dW3, dD, M, N, K);
  cudaMemcpy(hD.data(), dD, hD.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dX);
  cudaFree(dW1);
  cudaFree(dW3);
  cudaFree(dD);
  if (rc != 0) {
    std::printf("sm90 swiglu rc=%d\n", rc);
    return rc;
  }

  float max_abs = 0.f;
  for (size_t i = 0; i < hD.size(); ++i) {
    float err = std::fabs(hD[i] - hRef[i]);
    if (err > max_abs) max_abs = err;
  }
  // FP16 tensor-core math, compared against an FP32 host product.
  const float atol = 2e-2f;
  if (max_abs > atol) {
    std::printf("FAIL: max abs err %g > %g\n", max_abs, atol);
    return 1;
  }
  std::printf("sm90 swiglu ok max_abs=%g\n", max_abs);
  return 0;
}
