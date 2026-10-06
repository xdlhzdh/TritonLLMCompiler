"""Frontend AST mapping and the first TTGIR layout passes.

The raw module is ``ast_to_ttir``. ``asm['ttir']`` is that module after
``make_ttir``, so it cannot show ``tt.make_tensor_ptr``.
"""
from pathlib import Path

import pytest

from triton_llm.compiler.frontend import capture_frontend_ttir
from triton_llm.compiler.lesson import _frontend_lesson, launch_frontend_lesson
from triton_llm.compiler.pipeline import PassOptions
from triton_llm.tt_opt import run_file


@pytest.mark.cuda
def test_frontend_ops_and_middle_end_layout(cuda_or_skip, tmp_path):
    raw = capture_frontend_ttir(_frontend_lesson, launch_frontend_lesson)
    path = Path(tmp_path) / "lesson.mlir"
    path.write_text(raw)

    # constexpr BLOCK_* become the tensor shape. M/N/K stay arguments.
    assert "tensor<16x16x" in raw
    assert "BLOCK_M" not in raw
    for op in (
        "tt.get_program_id",
        "tt.make_range",
        "tt.addptr",
        "tt.load",
        "tt.dot",
        "tt.store",
        "tt.make_tensor_ptr",
        "tt.reduce",
        "tt.splat",
        "scf.for",
        "scf.if",
        "tt.divisibility",
    ):
        assert op in raw, op

    options = PassOptions(sm=70, num_warps=4)
    rewritten = run_file(path, ["--triton-rewrite-tensor-pointer"], options)
    assert "tt.make_tensor_ptr" not in rewritten
    assert "tt.load" in rewritten

    lowered = run_file(
        path,
        ["--make-ttir", "--convert-triton-to-tritongpu", "--tritongpu-accelerate-matmul"],
        options,
    )
    assert "tt.make_tensor_ptr" not in lowered
    assert "#triton_gpu.blocked" in lowered
    # sm_70 selects MMA v1 (versionMajor = 1); sm_80 would print versionMajor = 2.
    assert "#triton_gpu.nvidia_mma<{versionMajor = 1" in lowered
    assert "#triton_gpu.dot_op<{opIdx = 0, parent = #mma}>" in lowered
    assert "triton_gpu.convert_layout" in lowered


@pytest.mark.cuda
def test_tiled_dot_lowers_through_ttgir_to_mma(cuda_or_skip):
    """Tile sizes come from ``choose_gemm_tile``. The JIT runs ``make_ttir``,
    ``make_ttgir``, ``make_llir``, and ``make_ptx``. ``K`` is two ``BLOCK_K``
    steps so the K loop stays in ``asm['ttir']``.
    """
    from triton.runtime import driver

    from triton_llm.arch.tiling import choose_gemm_tile

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
    kernels = list(_frontend_lesson.cache[device].values())
    assert len(kernels) == 1
    asm = kernels[0].asm
    assert "scf.for" in asm["ttir"]
    assert "tt.dot" in asm["ttir"]
    assert f"tensor<{block_m}x{block_k}x" in asm["ttir"]
    assert "#triton_gpu.nvidia_mma<{versionMajor = 1" in asm["ttgir"]
    assert "#triton_gpu.dot_op<{opIdx = 0, parent = #mma}>" in asm["ttgir"]
    # make_llir then llvm.to_module. This text is LLVM IR, after tt.dot is gone.
    llir = asm["llir"]
    assert 'source_filename = "LLVMDialectModule"' in llir
    assert "define void @_frontend_lesson(" in llir
    assert "tt.dot" not in llir
    assert "triton_gpu" not in llir
    assert 'asm sideeffect "mma.sync.aligned.m8n8k4' in llir
    assert "mma.sync.aligned.m8n8k4" in asm["ptx"]
