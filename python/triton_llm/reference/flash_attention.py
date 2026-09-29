"""Unfused FlashAttention reference. Same arguments as ``triton_llm.ops.flash_attention``.

This materializes the full score matrix in PyTorch. The Triton op does not:
it keeps the online softmax state in registers and never writes SxS scores to HBM.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """q/k/v: [B, H, S, D] -> [B, H, Sq, D]. Matches ``ops.flash_attention``."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        m, n = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(m, n, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)
