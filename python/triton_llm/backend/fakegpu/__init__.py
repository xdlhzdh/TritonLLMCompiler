"""Opt-in teaching backend. Call ``activate`` around one kernel launch."""
from __future__ import annotations

from contextlib import contextmanager

from triton_llm.backend.fakegpu.driver import FAKE_DEVICE, LAUNCHES, FakeGPUDriver


def register() -> None:
    from triton_llm.backend import register as register_backends

    register_backends()


@contextmanager
def activate():
    """Make ``@triton.jit`` compile with ``FakeGPUBackend`` for this block."""
    register()
    from triton.runtime.driver import driver

    previous = driver.active
    driver.set_active(FakeGPUDriver())
    try:
        yield
    finally:
        driver.set_active(previous)


__all__ = ["FAKE_DEVICE", "LAUNCHES", "activate", "register"]
