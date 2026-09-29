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
