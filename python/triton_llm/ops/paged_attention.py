"""PagedAttention with vLLM-style block table indirect addressing (sm_70)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_llm.arch.sm70 import MAX_PIPELINE_STAGES


@triton.jit
def _paged_attn_kernel(
    Q, K_cache, V_cache, Out,
    block_tables, context_lens,
    stride_qb, stride_qh, stride_qd,
    stride_kblock, stride_kbs, stride_kh, stride_kd,
    stride_vblock, stride_vbs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_bt_b, stride_bt_n,
    sm_scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Decode-style: Q is [B,H,1,D]; one query token per sequence."""
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    context_len = tl.load(context_lens + off_b)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)
    q = (q * sm_scale).to(q.dtype)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # ``break`` is illegal in Triton. The host sets MAX_BLOCKS to the longest
    # sequence in this launch; shorter rows mask the remaining blocks.
    for blk_idx in range(0, MAX_BLOCKS):
        start_tok = blk_idx * BLOCK_SIZE
        block_valid = start_tok < context_len
        phys = tl.load(block_tables + off_b * stride_bt_b + blk_idx * stride_bt_n)
        offs_n = tl.arange(0, BLOCK_N)
        tok = start_tok + offs_n
        mask_n = (tok < context_len) & block_valid

        k_ptrs = (
            K_cache
            + phys * stride_kblock
            + offs_n[None, :] * stride_kbs
            + off_h * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        # tl.dot requires both non-batch dimensions >= 16. Pad the single
        # query row to 16 and keep row 0.
        rows = tl.arange(0, 16)[:, None]
        q_mat = tl.where(rows == 0, q[None, :], tl.zeros([16, HEAD_DIM], q.dtype))
        qk = tl.sum(tl.where(rows == 0, tl.dot(q_mat, k), 0), 0)
        qk = tl.where(mask_n, qk, float("-inf"))

        m_ij = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        p = tl.where(mask_n, p, 0.0)
        # An invalid block is all -inf. Keep the running softmax state unchanged.
        alpha = tl.where(block_valid, alpha, 1.0)
        p_sum = tl.sum(p, 0)
        l_i = tl.where(block_valid, l_i * alpha + p_sum, l_i)
        acc = tl.where(block_valid, acc * alpha, acc)

        v_ptrs = (
            V_cache
            + phys * stride_vblock
            + offs_n[:, None] * stride_vbs
            + off_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = tl.where(block_valid, acc + tl.sum(p[:, None] * v.to(tl.float32), 0), acc)
        m_i = tl.where(block_valid, m_new, m_i)

    acc = acc / l_i
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty))


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """q: [B,H,1,D]; caches: [num_blocks, block_size, H, D]; block_tables: [B, max_blocks]."""
    assert q.is_cuda
    b, h, m, d = q.shape
    assert m == 1, "paged_attention decode kernel expects seqlen_q == 1"
    assert d in (16, 32, 64, 128)
    assert block_size in (16, 32, 64)
    if sm_scale is None:
        sm_scale = d ** -0.5

    out = torch.empty((b, h, 1, d), device=q.device, dtype=q.dtype)
    max_blocks = block_tables.shape[1]
    max_context = int(context_lens.max().item()) if b else 0
    n_blocks = max(1, min(max_blocks, triton.cdiv(max_context, block_size)))
    grid = (b, h)
    _paged_attn_kernel[grid](
        q, k_cache, v_cache, out,
        block_tables, context_lens,
        q.stride(0), q.stride(1), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        float(sm_scale),
        BLOCK_SIZE=block_size, HEAD_DIM=d, BLOCK_N=block_size, MAX_BLOCKS=n_blocks,
        num_warps=4, num_stages=MAX_PIPELINE_STAGES,
    )
    return out
