import pytest
import torch

from triton_llm.ops.swiglu import swiglu
from triton_llm.reference.swiglu import swiglu_ref

ATOL, RTOL = 1e-3, 1e-2


@pytest.mark.cuda
@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (128, 256, 64)])
def test_swiglu_fp16(cuda_or_skip, m, n, k):
    torch.manual_seed(3)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w1 = torch.randn(k, n, device="cuda", dtype=torch.float16)
    w3 = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = swiglu(x, w1, w3)
    ref = swiglu_ref(x, w1, w3)
    assert torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
