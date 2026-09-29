"""FlashAttention-2 on SM70: block pointers and online softmax.

The launch uses ``num_stages=2`` from ``arch.sm70``. Triton's ``make_ttgir``
does not run the software pipeliner when ``sm // 10 < 8``, so this is a
launch hint, not a cp.async pipeline. This kernel has no Hopper TMA/WGMMA
path. The SM90 CUTLASS file is SwiGLU, not attention.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_llm.arch.tiling import flash_attention_tile


@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    stride_z, stride_m, stride_d,
    seqlen_q, seqlen_k,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_base = Q + off_hz * stride_z
    k_base = K + off_hz * stride_z
    v_base = V + off_hz * stride_z
    o_base = Out + off_hz * stride_z

    q_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(seqlen_q, HEAD_DIM),
        strides=(stride_m, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(HEAD_DIM, seqlen_k),
        strides=(stride_d, stride_m),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(seqlen_k, HEAD_DIM),
        strides=(stride_m, stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * sm_scale).to(q.dtype)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen_q
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen_k
        k = tl.load(k_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k)
        if CAUSAL:
            qk = tl.where(
                (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :],
                qk,
                float("-inf"),
            )
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    o_ptr = tl.make_block_ptr(
        base=o_base,
        shape=(seqlen_q, HEAD_DIM),
        strides=(stride_m, stride_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


def flash_attention(q, k, v, *, causal: bool = False, sm_scale: float | None = None):
    """q/k/v: [B, H, S, D] FP16. Online softmax, no SxS score matrix in HBM."""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    if q.dtype == torch.bfloat16:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            raise ValueError("BF16 flash_attention requires sm_80+; use FP16 on V100")
    b, h, seqlen_q, d = q.shape
    seqlen_k = k.shape[2]
    assert d in (16, 32, 64, 128)
    if sm_scale is None:
        sm_scale = d ** -0.5
    q2 = q.reshape(b * h, seqlen_q, d).contiguous()
    k2 = k.reshape(b * h, seqlen_k, d).contiguous()
    v2 = v.reshape(b * h, seqlen_k, d).contiguous()
    out2 = torch.empty_like(q2)
    tile = flash_attention_tile()
    block_m, block_n = tile["BLOCK_M"], tile["BLOCK_N"]
    grid = (triton.cdiv(seqlen_q, block_m), b * h)
    _flash_attn_fwd[grid](
        q2, k2, v2, out2,
        q2.stride(0), q2.stride(1), q2.stride(2),
        seqlen_q, seqlen_k, float(sm_scale),
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d, CAUSAL=causal,
        num_warps=4, num_stages=tile["num_stages"],
    )
    return out2.reshape(b, h, seqlen_q, d)
