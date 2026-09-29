import pytest
import torch

from triton_llm.ops.paged_attention import paged_attention
from triton_llm.reference.paged_attention import paged_attention_ref

ATOL, RTOL = 1e-3, 1e-2


@pytest.mark.cuda
def test_paged_attention_fp16(cuda_or_skip):
    torch.manual_seed(1)
    b, h, d, block_size = 2, 2, 64, 16
    context_lens = torch.tensor([40, 24], device="cuda", dtype=torch.int32)
    max_blocks = 4
    num_blocks = b * max_blocks
    k_cache = torch.randn(num_blocks, block_size, h, d, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(num_blocks, block_size, h, d, device="cuda", dtype=torch.float16)
    block_tables = torch.arange(num_blocks, device="cuda", dtype=torch.int32).reshape(b, max_blocks)
    q = torch.randn(b, h, 1, d, device="cuda", dtype=torch.float16)

    out = paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    ref = paged_attention_ref(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    assert torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
