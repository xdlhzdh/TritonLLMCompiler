#include "cutlass_c_api.h"
#include "cuda_utils.cuh"
#include <cmath>

__global__ void RmsNormRopeKernel(const float *X, const float *weight,
                                  const float *cos, const float *sin, float *D,
                                  float eps, int S, int H, int Dim) {
  int b = blockIdx.z;
  int s = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  int half = Dim / 2;
  extern __shared__ float sh[];
  float *x = sh;
  // load
  for (int i = tid; i < Dim; i += blockDim.x) {
    size_t idx = (((size_t)b * S + s) * H + h) * Dim + i;
    x[i] = X[idx];
  }
  __syncthreads();
  // rms
  float var = 0.f;
  for (int i = tid; i < Dim; i += blockDim.x)
    var += x[i] * x[i];
  // block reduce
  __shared__ float red[256];
  red[tid] = var;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) red[tid] += red[tid + stride];
    __syncthreads();
  }
  float rstd = rsqrtf(red[0] / float(Dim) + eps);
  for (int i = tid; i < half; i += blockDim.x) {
    float x1 = x[i] * rstd * weight[i];
    float x2 = x[i + half] * rstd * weight[i + half];
    float c1 = cos[s * Dim + i];
    float s1 = sin[s * Dim + i];
    float c2 = cos[s * Dim + i + half];
    float s2 = sin[s * Dim + i + half];
    float o1 = x1 * c1 - x2 * s1;
    float o2 = x2 * c2 + x1 * s2;
    size_t base = (((size_t)b * S + s) * H + h) * Dim;
    D[base + i] = o1;
    D[base + i + half] = o2;
  }
}

extern "C" int cutlass_rmsnorm_rope(const float *X, const float *weight,
                                    const float *cos, const float *sin, float *D,
                                    float eps, int64_t B, int64_t S, int64_t H,
                                    int64_t Dim) {
  dim3 launchGrid(static_cast<unsigned>(H), static_cast<unsigned>(S),
                   static_cast<unsigned>(B));
  unsigned threads = 256;
  size_t smem = size_t(Dim) * sizeof(float);
  RmsNormRopeKernel<<<launchGrid, threads, smem>>>(X, weight, cos, sin, D, eps, int(S),
                                             int(H), int(Dim));
  return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess
             ? 0
             : 1;
}
