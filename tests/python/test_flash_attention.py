import pytest
import torch

from triton_llm.ops.flash_attention import flash_attention
from triton_llm.reference.flash_attention import flash_attention_ref

ATOL, RTOL = 1e-3, 1e-2


@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", [64, 128])
def test_flash_attention_fp16(cuda_or_skip, causal, seqlen):
    torch.manual_seed(0)
    b, h, d = 1, 2, 64
    q = torch.randn(b, h, seqlen, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, seqlen, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, seqlen, d, device="cuda", dtype=torch.float16)
    out = flash_attention(q, k, v, causal=causal)
    ref = flash_attention_ref(q, k, v, causal=causal)
    assert torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.cuda
def test_flash_attention_bf16_rejected_on_volta(cuda_or_skip):
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        pytest.skip("device supports BF16")
    q = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        flash_attention(q, q, q)
