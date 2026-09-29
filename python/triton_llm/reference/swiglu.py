"""Unfused SwiGLU reference. Same arguments as ``triton_llm.ops.swiglu``.

``silu(x @ W1) * (x @ W3)`` as two matmuls plus SiLU. The Triton kernel
accumulates both GEMMs in one launch and applies SiLU in registers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu_ref(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    # x: [M, K], w1/w3: [K, N]. Result dtype matches the Triton kernel.
    out = F.silu(x.float() @ w1.float()) * (x.float() @ w3.float())
    return out.to(x.dtype)
