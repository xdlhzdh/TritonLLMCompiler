"""Device kernels. Import the op you need; the package root re-exports them."""
from triton_llm.ops.flash_attention import flash_attention
from triton_llm.ops.paged_attention import paged_attention
from triton_llm.ops.rmsnorm_rope import rmsnorm_rope
from triton_llm.ops.swiglu import swiglu

__all__ = ["flash_attention", "paged_attention", "rmsnorm_rope", "swiglu"]
