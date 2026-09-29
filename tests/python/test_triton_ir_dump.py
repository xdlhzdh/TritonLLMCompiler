"""The Triton JIT path must actually emit TTIR, TTGIR, and LLVM IR."""
import pytest
import torch

from triton_llm.compiler_dump import dump_kernel
from triton_llm.ops.flash_attention import _flash_attn_fwd, flash_attention


@pytest.mark.cuda
def test_flash_attention_emits_ttir_ttgir_llir(cuda_or_skip, tmp_path):
    q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    flash_attention(q, q, q, causal=False)
    dump_kernel(_flash_attn_fwd, tmp_path)
    ttir = (tmp_path / "_flash_attn_fwd.ttir").read_text()
    ttgir = (tmp_path / "_flash_attn_fwd.ttgir").read_text()
    llir = (tmp_path / "_flash_attn_fwd.llir").read_text()
    ptx = (tmp_path / "_flash_attn_fwd.ptx").read_text()
    assert "tt.func" in ttir
    assert "tt.dot" in ttir
    assert "ttg." in ttgir or "triton_gpu" in ttgir
    assert "define " in llir
    assert ".target" in ptx or ".version" in ptx
