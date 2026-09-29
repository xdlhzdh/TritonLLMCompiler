"""Python front end -> textual TTIR -> the C++ ``build/bin/tt-opt`` binary.

Two different MLIR builds meet here. ``libtriton.so`` (installed Triton 3.1)
carries the LLVM that Triton pinned. ``build/bin/tt-opt`` links LLVM 23.
Only the IR text crosses the boundary, so this test checks that the text the
front end emits is accepted by the binary and that a handwritten pass acts on it.
"""
import subprocess
from pathlib import Path

import pytest

from triton_llm.compiler.frontend import capture_frontend_ttir
from triton_llm.compiler.lesson import _frontend_lesson, launch_frontend_lesson

ROOT = Path(__file__).resolve().parents[2]
TT_OPT = ROOT / "build" / "bin" / "tt-opt"


@pytest.mark.cuda
def test_front_end_ttir_runs_through_cpp_tt_opt(cuda_or_skip, tmp_path):
    if not TT_OPT.exists():
        pytest.skip("build/bin/tt-opt is missing; run cmake --build build --target tt-opt")
    raw = capture_frontend_ttir(_frontend_lesson, launch_frontend_lesson)
    src = tmp_path / "lesson.mlir"
    src.write_text(raw)

    # CodeGenerator.visit_For calls builder.create_undef for the loop induction
    # variable placeholder. That op is llvm.mlir.undef, so the binary must
    # register the llvm dialect to parse front-end output.
    assert "llvm.mlir.undef" in raw

    proc = subprocess.run(
        [str(TT_OPT), str(src), "--triton-annotate-dot-stages"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "tt.dot" in proc.stdout
    assert "triton_llm.num_stages = 2 : i64" in proc.stdout

    proc = subprocess.run(
        [str(TT_OPT), str(src), "--triton-annotate-dot-stages=sm=90"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "triton_llm.num_stages = 3 : i64" in proc.stdout
