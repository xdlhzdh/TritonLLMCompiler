"""Unfused PagedAttention reference. Same arguments as ``triton_llm.ops.paged_attention``.

Gather KV pages through ``block_tables``, then run the dense score-matrix
attention. The Triton kernel indexes the block table inside the loop and
never concatenates the KV cache into a contiguous tensor.
"""
from __future__ import annotations

import torch

from triton_llm.reference.flash_attention import flash_attention_ref


def paged_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """q: [B, H, 1, D]; caches: [num_blocks, block_size, H, D]; block_tables: [B, max_blocks]."""
    if q.shape[2] != 1:
        raise ValueError("paged_attention_ref matches the decode kernel: seqlen_q must be 1")
    b = q.shape[0]
    outs = []
    for bi in range(b):
        clen = int(context_lens[bi].item())
        nblocks = (clen + block_size - 1) // block_size
        ks, vs = [], []
        for blk in range(nblocks):
            phys = int(block_tables[bi, blk].item())
            ks.append(k_cache[phys])
            vs.append(v_cache[phys])
        k = torch.cat(ks, dim=0)[:clen].permute(1, 0, 2).unsqueeze(0)
        v = torch.cat(vs, dim=0)[:clen].permute(1, 0, 2).unsqueeze(0)
        outs.append(flash_attention_ref(q[bi : bi + 1], k, v, causal=False, sm_scale=sm_scale))
    return torch.cat(outs, dim=0)
