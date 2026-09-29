"""Fused RMSNorm + RoPE in a single Triton kernel (no intermediate HBM write)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_rope_kernel(
    X, W, Cos, Sin, Out,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_cos_s, stride_cos_d,
    eps,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_s = tl.program_id(1)
    off_h = tl.program_id(2)

    offs1 = tl.arange(0, HALF)
    x1 = tl.load(
        X + off_b * stride_xb + off_s * stride_xs + off_h * stride_xh + offs1 * stride_xd
    ).to(tl.float32)
    x2 = tl.load(
        X + off_b * stride_xb + off_s * stride_xs + off_h * stride_xh + (offs1 + HALF) * stride_xd
    ).to(tl.float32)
    w1 = tl.load(W + offs1).to(tl.float32)
    w2 = tl.load(W + offs1 + HALF).to(tl.float32)
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / float(HEAD_DIM)
    rstd = tl.rsqrt(var + eps)
    x1 = x1 * rstd * w1
    x2 = x2 * rstd * w2

    cos1 = tl.load(Cos + off_s * stride_cos_s + offs1 * stride_cos_d).to(tl.float32)
    sin1 = tl.load(Sin + off_s * stride_cos_s + offs1 * stride_cos_d).to(tl.float32)
    cos2 = tl.load(Cos + off_s * stride_cos_s + (offs1 + HALF) * stride_cos_d).to(tl.float32)
    sin2 = tl.load(Sin + off_s * stride_cos_s + (offs1 + HALF) * stride_cos_d).to(tl.float32)

    o1 = x1 * cos1 - x2 * sin1
    o2 = x2 * cos2 + x1 * sin2

    out1 = Out + off_b * stride_ob + off_s * stride_os + off_h * stride_oh + offs1 * stride_od
    out2 = Out + off_b * stride_ob + off_s * stride_os + off_h * stride_oh + (offs1 + HALF) * stride_od
    tl.store(out1, o1.to(Out.dtype.element_ty))
    tl.store(out2, o2.to(Out.dtype.element_ty))


def rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """x: [B,S,H,D]; weight: [D]; cos/sin: [S,D]. Single fused kernel."""
    assert x.is_cuda and x.dim() == 4
    b, s, h, d = x.shape
    assert d % 2 == 0 and d in (16, 32, 64, 128)
    assert weight.shape == (d,)
    assert cos.shape == (s, d) and sin.shape == (s, d)
    out = torch.empty_like(x)
    grid = (b, s, h)
    _rmsnorm_rope_kernel[grid](
        x, weight, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos.stride(0), cos.stride(1),
        float(eps),
        HEAD_DIM=d, HALF=d // 2,
        num_warps=4, num_stages=2,
    )
    return out
