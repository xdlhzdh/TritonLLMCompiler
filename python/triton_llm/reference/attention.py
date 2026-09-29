"""Deprecated barrel. Import ``flash_attention_ref`` or ``paged_attention_ref`` instead."""
from triton_llm.reference.flash_attention import flash_attention_ref as attention_ref
from triton_llm.reference.paged_attention import paged_attention_ref

__all__ = ["attention_ref", "paged_attention_ref"]
