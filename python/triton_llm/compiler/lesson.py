"""One kernel that exercises the frontend constructs a Triton middle-end consumes.

``M``, ``N``, and ``K`` are ``do_not_specialize`` so they stay function
arguments. ``BLOCK_*`` are ``tl.constexpr`` and are baked into tensor shapes.
Pointer arguments that Triton proves 16-byte aligned carry
``tt.divisibility = 16``. ``tl.make_block_ptr`` is still ``tt.make_tensor_ptr``
in this IR; ``make_ttir`` rewrites it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "N", "K"])
def _frontend_lesson(
    A, B, C, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    c = tl.load(c_ptr, boundary_check=(0, 1), padding_option="zero")
    acc = acc + c
    if pid == 0:
        acc = acc * 2.0
    scale = tl.sum(acc, axis=1)
    acc = acc * scale[:, None]
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


def launch_frontend_lesson(
    m: int = 16,
    n: int = 16,
    k: int = 32,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
) -> torch.Tensor:
    """Launch one tile. ``tl.dot`` requires both non-batch dimensions to be >= 16."""
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.randn(m, n, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    _frontend_lesson[(1, )](
        a, b, c, out,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return out
