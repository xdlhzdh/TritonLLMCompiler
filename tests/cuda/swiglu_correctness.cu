#include <algorithm>
#include "cutlass_c_api.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

static float silu(float x) { return x / (1.f + std::exp(-x)); }

int main() {
  const int M = 64, N = 64, K = 64;
  std::vector<float> X(M * K), W1(K * N), W3(K * N), D(M * N), Ref(M * N);
  for (int i = 0; i < M * K; ++i) X[i] = float((i % 7) - 3) * 0.1f;
  for (int i = 0; i < K * N; ++i) {
    W1[i] = float((i % 5) - 2) * 0.05f;
    W3[i] = float((i % 3) - 1) * 0.05f;
  }
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float g = 0.f, u = 0.f;
      for (int k = 0; k < K; ++k) {
        g += X[m * K + k] * W1[k * N + n];
        u += X[m * K + k] * W3[k * N + n];
      }
      Ref[m * N + n] = silu(g) * u;
    }
  }
  float *dX, *dW1, *dW3, *dD;
  cudaMalloc(&dX, X.size() * 4);
  cudaMalloc(&dW1, W1.size() * 4);
  cudaMalloc(&dW3, W3.size() * 4);
  cudaMalloc(&dD, D.size() * 4);
  cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dW1, W1.data(), W1.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dW3, W3.data(), W3.size() * 4, cudaMemcpyHostToDevice);
  int rc = cutlass_gemm_swiglu(dX, dW1, dW3, dD, M, N, K);
  if (rc) { std::printf("FAIL: kernel returned %d\n", rc); return 1; }
  cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
  float max_abs = 0.f, max_rel = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float a = std::fabs(D[i] - Ref[i]);
    float r = a / (std::fabs(Ref[i]) + 1e-5f);
    max_abs = std::max(max_abs, a);
    max_rel = std::max(max_rel, r);
  }
  std::printf("swiglu max_abs=%.6f max_rel=%.6f\n", max_abs, max_rel);
  cudaFree(dX); cudaFree(dW1); cudaFree(dW3); cudaFree(dD);
  if (max_abs > 1e-2f || max_rel > 1e-1f) return 1;
  return 0;
}
