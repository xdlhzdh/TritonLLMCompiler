"""One kernel compiled by ``FakeGPUBackend``.

The ``for`` / ``tl.dot`` tile is written in the DSL. ``tl.chip_rcp`` is the
DSL op from ``0001-frontend-llvm19.patch``. This backend stops after TTIR
and prints ``fake.rcp.f32``. It does not call ``CUDABackend.make_ttgir``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_llm.backend.fakegpu import FAKE_DEVICE, activate


@triton.jit
def _fake_gpu_lesson(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * N + offs_n[None, :],
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    acc = tl.chip_rcp(acc)
    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def compile_fake_gpu_lesson(block: int = 16):
    """Compile and "launch" under ``activate``. The launch does not touch the GPU."""
    a = torch.ones((block, block), dtype=torch.float16)
    b = torch.ones((block, block), dtype=torch.float16)
    c = torch.empty((block, block), dtype=torch.float32)
    _fake_gpu_lesson.cache[FAKE_DEVICE].clear()
    with activate():
        return _fake_gpu_lesson[(1, )](
            a, b, c,
            block, block, block,
            BLOCK_M=block, BLOCK_N=block, BLOCK_K=block,
        )
