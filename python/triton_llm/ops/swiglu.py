"""Fused SwiGLU: silu(x @ W1) * (x @ W3) with register-friendly tiling."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_llm.arch.tiling import choose_gemm_tile


@triton.jit
def _swiglu_kernel(
    X, W1, W3, Out,
    M, N, K,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask_k = k < K
        x_ptrs = X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)

        w1_ptrs = W1 + k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n
        w3_ptrs = W3 + k[:, None] * stride_w3k + offs_n[None, :] * stride_w3n
        w1 = tl.load(w1_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        w3 = tl.load(w3_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)

        acc_g += tl.dot(x, w1)
        acc_u += tl.dot(x, w3)

    # silu(g) = g * sigmoid(g)
    sig = 1.0 / (1.0 + tl.exp(-acc_g))
    out = (acc_g * sig) * acc_u

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def swiglu(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """x: [M,K], w1/w3: [K,N] -> [M,N]."""
    assert x.is_cuda and w1.is_cuda and w3.is_cuda
    assert x.dim() == 2 and w1.dim() == 2 and w3.dim() == 2
    m, k = x.shape
    k2, n = w1.shape
    assert k == k2 and w3.shape == (k, n)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    major, minor = torch.cuda.get_device_capability(x.device)
    tile = choose_gemm_tile(major * 10 + minor)
    block_m, block_n, block_k = tile["BLOCK_M"], tile["BLOCK_N"], tile["BLOCK_K"]
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _swiglu_kernel[grid](
        x, w1, w3, out,
        m, n, k,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=tile["num_stages"],
    )
    return out
