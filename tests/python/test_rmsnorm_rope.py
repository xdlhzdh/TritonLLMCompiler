import math
import pytest
import torch

from triton_llm.ops.rmsnorm_rope import rmsnorm_rope
from triton_llm.reference.norm_rope import rmsnorm_rope_ref

ATOL, RTOL = 1e-3, 1e-2


def _rope_cos_sin(seqlen: int, dim: int, device):
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(torch.float16), emb.sin().to(torch.float16)


@pytest.mark.cuda
def test_rmsnorm_rope_fp16(cuda_or_skip):
    torch.manual_seed(2)
    b, s, h, d = 2, 16, 4, 64
    x = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
    w = torch.randn(d, device="cuda", dtype=torch.float16)
    cos, sin = _rope_cos_sin(s, d, "cuda")
    out = rmsnorm_rope(x, w, cos, sin)
    ref = rmsnorm_rope_ref(x, w, cos, sin)
    assert torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
