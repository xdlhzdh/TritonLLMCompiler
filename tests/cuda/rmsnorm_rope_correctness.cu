#include <algorithm>
#include "cutlass_c_api.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

int main() {
  const int B = 1, S = 8, H = 2, D = 64;
  std::vector<float> X(B * S * H * D), W(D), Cos(S * D), Sin(S * D), Out(B * S * H * D), Ref(B * S * H * D);
  for (size_t i = 0; i < X.size(); ++i) X[i] = float((int(i) % 9) - 4) * 0.1f;
  for (int i = 0; i < D; ++i) W[i] = 1.f;
  for (int s = 0; s < S; ++s)
    for (int i = 0; i < D; ++i) {
      Cos[s * D + i] = std::cos(0.01f * (s + i));
      Sin[s * D + i] = std::sin(0.01f * (s + i));
    }
  float eps = 1e-6f;
  int half = D / 2;
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s)
      for (int h = 0; h < H; ++h) {
        float var = 0.f;
        size_t base = (((size_t)b * S + s) * H + h) * D;
        for (int i = 0; i < D; ++i) var += X[base + i] * X[base + i];
        float rstd = 1.f / std::sqrt(var / D + eps);
        for (int i = 0; i < half; ++i) {
          float x1 = X[base + i] * rstd * W[i];
          float x2 = X[base + i + half] * rstd * W[i + half];
          Ref[base + i] = x1 * Cos[s * D + i] - x2 * Sin[s * D + i];
          Ref[base + i + half] =
              x2 * Cos[s * D + i + half] + x1 * Sin[s * D + i + half];
        }
      }
  float *dX, *dW, *dC, *dS, *dO;
  cudaMalloc(&dX, X.size() * 4);
  cudaMalloc(&dW, W.size() * 4);
  cudaMalloc(&dC, Cos.size() * 4);
  cudaMalloc(&dS, Sin.size() * 4);
  cudaMalloc(&dO, Out.size() * 4);
  cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dW, W.data(), W.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, Cos.data(), Cos.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dS, Sin.data(), Sin.size() * 4, cudaMemcpyHostToDevice);
  int rc = cutlass_rmsnorm_rope(dX, dW, dC, dS, dO, eps, B, S, H, D);
  if (rc) { std::printf("FAIL rc=%d\n", rc); return 1; }
  cudaMemcpy(Out.data(), dO, Out.size() * 4, cudaMemcpyDeviceToHost);
  float max_abs = 0.f;
  for (size_t i = 0; i < Out.size(); ++i)
    max_abs = std::max(max_abs, std::fabs(Out[i] - Ref[i]));
  std::printf("rmsnorm_rope max_abs=%.6f\n", max_abs);
  cudaFree(dX); cudaFree(dW); cudaFree(dC); cudaFree(dS); cudaFree(dO);
  return max_abs < 1e-3f ? 0 : 1;
}
