"""FakeGPUBackend replaces CUDABackend.add_stages for one launch.

The kernel still enters libtriton.so through ASTSource.make_ir. The stages
after TTIR are FakeGPUBackend.make_fakeasm / make_fakebin, so the asm dict
has ttir, fakeasm, and fakegpu, and no ttgir, ptx, or cubin.
"""
from __future__ import annotations

import os

import pytest

from triton_llm.backend.fakegpu import FAKE_DEVICE, LAUNCHES
from triton_llm.backend.fakegpu.lesson import compile_fake_gpu_lesson


def test_fake_gpu_backend_stops_after_ttir(tmp_path, monkeypatch):
    import triton
    import triton.language as tl
    from triton.runtime.driver import driver

    so = os.path.realpath(triton._C.libtriton.__file__)
    if not hasattr(tl, "chip_rcp"):
        pytest.fail(f"tl.chip_rcp missing; libtriton.so is {so}")

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    LAUNCHES.clear()
    kernel = compile_fake_gpu_lesson()

    assert kernel.metadata.target.backend == "fakegpu"
    assert kernel.metadata.name == "_fake_gpu_lesson"
    assert "tt.chip_rcp" in kernel.asm["ttir"]
    assert "scf.for" in kernel.asm["ttir"]
    assert "tt.dot" in kernel.asm["ttir"]
    fakeasm = kernel.asm["fakeasm"]
    positions = [fakeasm.index(op) for op in (
        "fake.loop", "fake.load", "fake.dot", "fake.yield", "fake.rcp.f32", "fake.store",
    )]
    assert positions == sorted(positions)
    assert b"fake.rcp.f32" in kernel.asm["fakegpu"]
    assert "ttgir" not in kernel.asm
    assert "ptx" not in kernel.asm
    assert "cubin" not in kernel.asm
    assert LAUNCHES[-1][0] == "_fake_gpu_lesson"
    assert LAUNCHES[-1][1] == (1, 1, 1)

    # activate() puts the previous driver back. CUDA stays the default.
    assert type(driver.active).__name__ != "FakeGPUDriver"
    assert FAKE_DEVICE == 99
