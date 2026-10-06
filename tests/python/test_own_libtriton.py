"""The kernel compiler is this repo's libtriton.so, built by scripts/build_libtriton.sh.

Tile sizes come from choose_gemm_tile and lower through TTGIR to mma.sync.
tl.chip_rcp lowers through the same so: TTIR, TTGIR, LLVM IR, PTX, cubin.
"""
from __future__ import annotations

import os

import pytest
import torch
import triton
import triton.language as tl

from triton_llm.arch.tiling import choose_gemm_tile
from triton_llm.compiler.lesson import _frontend_lesson, launch_frontend_lesson


def _require_own_triton():
    so = os.path.realpath(triton._C.libtriton.__file__)
    src = os.path.realpath(triton.__file__)
    if "TritonQAttn" in so or "TritonQAttn" in src:
        pytest.fail(f"import triton loaded {src} / {so}; run scripts/build_libtriton.sh")
    if not hasattr(tl, "chip_rcp"):
        pytest.fail("built triton.language has no chip_rcp")


@triton.jit
def _chip_rcp_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    y = tl.chip_rcp(x)
    tl.store(y_ptr + offs, y, mask=mask)


@pytest.mark.cuda
def test_own_libtriton_tiles_dot_through_ttgir_to_llvm(cuda_or_skip):
    _require_own_triton()
    from triton.runtime import driver

    tile = choose_gemm_tile(70)
    block_m = tile["BLOCK_M"]
    block_n = tile["BLOCK_N"]
    block_k = tile["BLOCK_K"]
    device = driver.active.get_current_device()
    _frontend_lesson.cache[device].clear()
    launch_frontend_lesson(
        m=block_m,
        n=block_n,
        k=block_k * 2,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_stages=tile["num_stages"],
    )
    asm = list(_frontend_lesson.cache[device].values())[0].asm
    assert "scf.for" in asm["ttir"]
    assert "tt.dot" in asm["ttir"]
    assert "#triton_gpu.nvidia_mma<{versionMajor = 1" in asm["ttgir"]
    assert "tt.dot" not in asm["llir"]
    assert "triton_gpu" not in asm["llir"]
    assert 'asm sideeffect "mma.sync.aligned.m8n8k4' in asm["llir"]
    assert "mma.sync.aligned.m8n8k4" in asm["ptx"]
    assert len(asm["cubin"]) > 0


@pytest.mark.cuda
def test_chip_rcp_lowers_through_ttgir_to_cubin(cuda_or_skip):
    _require_own_triton()
    from triton.runtime import driver

    n = 128
    x = torch.linspace(0.5, 4.0, n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    device = driver.active.get_current_device()
    _chip_rcp_kernel.cache[device].clear()
    _chip_rcp_kernel[(1,)](x, y, n, BLOCK=128)
    torch.testing.assert_close(y, torch.reciprocal(x), rtol=2**-11, atol=1e-5)
    asm = list(_chip_rcp_kernel.cache[device].values())[0].asm
    assert "tt.chip_rcp" in asm["ttir"]
    assert "tt.chip_rcp" in asm["ttgir"]
    assert "#triton_gpu.blocked" in asm["ttgir"] or "blocked" in asm["ttgir"]
    assert "tt.chip_rcp" not in asm["llir"]
    assert "rcp.approx.ftz.f32" in asm["llir"]
    assert "rcp.approx.ftz.f32" in asm["ptx"]
    assert len(asm["cubin"]) > 0
