"""One PyTorch reference per Triton op. Names and signatures match ``triton_llm.ops``.

| op | reference | what the reference does that the kernel does not |
|----|-----------|--------------------------------------------------|
| ``ops.flash_attention`` | ``flash_attention_ref`` | writes the full QK score matrix |
| ``ops.paged_attention`` | ``paged_attention_ref`` | gathers pages into a dense K/V, then calls the score-matrix reference |
| ``ops.rmsnorm_rope`` | ``rmsnorm_rope_ref`` | RMSNorm and RoPE are two separate PyTorch ops |
| ``ops.swiglu`` | ``swiglu_ref`` | two GEMMs and SiLU are separate PyTorch ops |
"""
from triton_llm.reference.flash_attention import flash_attention_ref
from triton_llm.reference.paged_attention import paged_attention_ref
from triton_llm.reference.norm_rope import rmsnorm_rope_ref
from triton_llm.reference.swiglu import swiglu_ref

__all__ = [
    "flash_attention_ref",
    "paged_attention_ref",
    "rmsnorm_rope_ref",
    "swiglu_ref",
]
