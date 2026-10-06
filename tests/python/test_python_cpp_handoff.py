"""Python front end -> textual TTIR -> ``build/bin/triton-opt``.

Both sides are the LLVM 19 build from ``scripts/build_libtriton.sh``.
The front end emits ``llvm.mlir.undef``, and ``triton-opt`` must parse it.
"""
import subprocess
from pathlib import Path

import pytest

from triton_llm.compiler.frontend import capture_frontend_ttir
from triton_llm.compiler.lesson import _frontend_lesson, launch_frontend_lesson

ROOT = Path(__file__).resolve().parents[2]
TT_OPT = ROOT / "build" / "bin" / "triton-opt"


@pytest.mark.cuda
def test_front_end_ttir_runs_through_cpp_tt_opt(cuda_or_skip, tmp_path):
    if not TT_OPT.exists():
        pytest.skip("build/bin/triton-opt is missing; run scripts/build_libtriton.sh")
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
