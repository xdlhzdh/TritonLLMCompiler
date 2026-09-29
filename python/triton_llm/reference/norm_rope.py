"""Unfused RMSNorm + RoPE reference. Same arguments as ``triton_llm.ops.rmsnorm_rope``.

The Triton kernel does both steps in one launch and does not store the
normalized tensor. This reference is two PyTorch ops.
"""
from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [..., D]
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(variance + eps)
    return (x_norm * weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D]; cos/sin: [S, D] or [1, S, 1, D]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    if cos.dim() == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    cos1, cos2 = cos[..., : d // 2], cos[..., d // 2 :]
    sin1, sin2 = sin[..., : d // 2], sin[..., d // 2 :]
    # Standard rotate half: (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
    o1 = x1.float() * cos1.float() - x2.float() * sin1.float()
    o2 = x2.float() * cos2.float() + x1.float() * sin2.float()
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


def rmsnorm_rope_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return apply_rope(rmsnorm(x, weight, eps), cos, sin)
