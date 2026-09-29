"""Tile legality. The picker in ``arch.tiling`` is what ``ops.swiglu`` launches."""
from __future__ import annotations

import pytest

from triton_llm.arch.sm70 import SMEM_BYTES as SM70_SMEM
from triton_llm.arch.tiling import choose_gemm_tile, choose_tile, max_pipeline_stages

_DRY_RUN_BYTES = 512 * 1024 * 1024


@pytest.mark.parametrize("batch", [1, 8, 32, 128])
@pytest.mark.parametrize("seqlen", [512, 2048, 8192, 32768])
def test_tiling_config_scan(batch, seqlen):
    cfg = choose_tile(batch, seqlen)
    assert cfg["BLOCK_M"] > 0 and cfg["BLOCK_N"] > 0
    assert cfg["num_stages"] == max_pipeline_stages(70)
    assert cfg["num_stages"] == 2
    assert cfg["smem_bytes"] <= SM70_SMEM
    head_dim = 64
    assert cfg["dry_run"] == (batch * seqlen * head_dim * 2 >= _DRY_RUN_BYTES)
    assert cfg == {**choose_gemm_tile(70), "dry_run": cfg["dry_run"]}


def test_hopper_tile_uses_three_stages():
    cfg = choose_gemm_tile(90)
    assert cfg["num_stages"] == 3
    assert cfg["smem_bytes"] <= 228 * 1024
