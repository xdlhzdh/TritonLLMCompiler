"""Teaching backends that sit beside NVIDIA.

``register`` puts them in ``triton.backends.backends``. ``is_active`` stays
false, so ``import triton`` still selects ``CudaDriver``.
"""
from __future__ import annotations


def register() -> None:
    """Insert ``fakegpu`` into Triton's backend table."""
    from triton.backends import Backend, backends

    from triton_llm.backend.fakegpu.compiler import FakeGPUBackend
    from triton_llm.backend.fakegpu.driver import FakeGPUDriver

    backends["fakegpu"] = Backend(FakeGPUBackend, FakeGPUDriver)
