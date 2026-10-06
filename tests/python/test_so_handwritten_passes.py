"""The four handwritten TTIR passes run inside libtriton.so and triton-opt."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from triton_llm.tt_opt import run_file
from triton_llm.compiler.pipeline import PassOptions

ROOT = Path(__file__).resolve().parents[2]
TT = ROOT / "tests" / "tt"
TRITON_OPT = ROOT / "build" / "bin" / "triton-opt"


def _so(path: Path, flag: str, sm: int = 70) -> str:
    return run_file(path, [flag], PassOptions(sm=sm))


def _opt(path: Path, flag: str) -> str:
    if not TRITON_OPT.is_file():
        pytest.skip("build/bin/triton-opt missing")
    proc = subprocess.run(
        [str(TRITON_OPT), str(path), flag],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_annotate_dot_stages_from_so_and_triton_opt():
    src = TT / "annotate_dot_stages.mlir"
    for text in (_so(src, "--triton-annotate-dot-stages"), _opt(src, "--triton-annotate-dot-stages")):
        assert "triton_llm.num_stages = 2" in text


def test_annotate_dot_stages_sm90():
    src = TT / "annotate_dot_stages_sm90.mlir"
    text = _so(src, "--triton-annotate-dot-stages=sm=90")
    assert "triton_llm.num_stages = 3" in text


def test_fuse_and_lower_round_trip():
    src = TT / "fuse_dot_epilogue.mlir"
    fused = _so(src, "--triton-fuse-dot-epilogue")
    assert "tt.fused_dot_mul" in fused
    assert "tt.dot" not in fused
    lowered = run_file_on_text(fused, "--triton-lower-fused-dot-mul")
    assert "tt.dot" in lowered
    assert "arith.mulf" in lowered
    assert "tt.fused_dot_mul" not in lowered


def test_tile_dot_stays_in_ttir():
    src = TT / "tile_dot.mlir"
    text = _so(src, "--triton-tile-dot")
    assert "scf.for" in text
    assert "tensor.extract_slice" in text
    assert "tt.dot" in text
    assert "triton_llm.block_k = 64" in text
    assert "triton_llm.num_stages = 2" in text


def run_file_on_text(text: str, flag: str) -> str:
    path = ROOT / "build" / "fused-roundtrip.mlir"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return _so(path, flag)
