"""The C++ tile table and ``arch/tiling.py`` must not drift apart.

``gemmTileForSm`` in ``FuseAndTileDot.cpp`` and ``choose_gemm_tile`` are two
copies of one strategy. This test runs the real binary for several sm values and
compares the annotations on the rewritten ``scf.for`` with the Python answer.
"""
import re
import subprocess
from pathlib import Path

import pytest

from triton_llm.arch.tiling import choose_gemm_tile

ROOT = Path(__file__).resolve().parents[2]
TT_OPT = ROOT / "build" / "bin" / "triton-opt"
SRC = ROOT / "tests" / "tt" / "tile_dot.mlir"


@pytest.mark.parametrize("sm", [70, 75, 80, 86, 90])
def test_cpp_tile_matches_python_table(sm):
    if not TT_OPT.exists():
        pytest.skip("build/bin/triton-opt is missing; run scripts/build_libtriton.sh")
    proc = subprocess.run(
        [str(TT_OPT), str(SRC), f"--triton-tile-dot=sm={sm}"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    block_k = int(re.search(r"triton_llm.block_k = (\d+)", proc.stdout).group(1))
    stages = int(re.search(r"triton_llm.num_stages = (\d+)", proc.stdout).group(1))
    expected = choose_gemm_tile(sm)
    assert block_k == expected["BLOCK_K"]
    assert stages == expected["num_stages"]
