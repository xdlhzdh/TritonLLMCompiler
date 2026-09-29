"""Tile sizes that fit the architecture shared-memory and pipeline limits.

``choose_gemm_tile`` is what ``ops.swiglu`` launches. ``flash_attention_tile``
is the SM70 online-softmax tile: Triton's software pipeliner does not run
when ``sm // 10 < 8``, so that kernel stays at 2 stages.
"""
from __future__ import annotations

from triton_llm.arch.hopper import PIPELINE_STAGES as SM90_STAGES
from triton_llm.arch.hopper import SMEM_BYTES as SM90_SMEM
from triton_llm.arch.sm70 import MAX_PIPELINE_STAGES as SM70_STAGES
from triton_llm.arch.sm70 import SMEM_BYTES as SM70_SMEM

# Preference order: the last candidate that fits is the one we launch.
_GEMM_CANDIDATES = (
    (64, 64, 32, 2),
    (64, 128, 32, 2),
    (128, 128, 32, 2),
    (128, 128, 64, 2),
    (128, 128, 64, 3),
)

_DRY_RUN_BYTES = 512 * 1024 * 1024


def max_pipeline_stages(sm: int) -> int:
    """Volta has no cp.async pipeline. Ampere can use 3. Hopper uses 3."""
    if sm >= 90:
        return SM90_STAGES
    if sm >= 80:
        return 3
    return SM70_STAGES


def smem_capacity(sm: int) -> int:
    if sm >= 90:
        return SM90_SMEM
    if sm >= 80:
        return 164 * 1024
    return SM70_SMEM


def _staged_bytes(block_m: int, block_n: int, block_k: int, stages: int) -> int:
    return stages * (block_m * block_k + block_n * block_k) * 2


def choose_gemm_tile(sm: int = 70) -> dict:
    """Largest FP16 A/B tile whose staged bytes fit this architecture."""
    limit = smem_capacity(sm)
    stage_cap = max_pipeline_stages(sm)
    chosen = None
    for block_m, block_n, block_k, stages in _GEMM_CANDIDATES:
        if stages > stage_cap:
            continue
        nbytes = _staged_bytes(block_m, block_n, block_k, stages)
        if nbytes <= limit:
            chosen = {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "num_stages": stages,
                "smem_bytes": nbytes,
            }
    if chosen is None:
        raise RuntimeError(f"no legal GEMM tile for sm_{sm}")
    return chosen


def choose_tile(batch: int, seqlen: int, head_dim: int = 64, sm: int = 70) -> dict:
    """GEMM tile plus whether materializing ``[batch, seqlen, head_dim]`` is a dry run."""
    chosen = choose_gemm_tile(sm)
    chosen["dry_run"] = batch * seqlen * head_dim * 2 >= _DRY_RUN_BYTES
    return chosen


def flash_attention_tile() -> dict:
    """Fixed SM70 attention tile. ``num_stages`` matches ``arch.sm70``."""
    return {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": SM70_STAGES}
