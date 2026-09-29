"""Dump the IR Triton actually emits for a ``@triton.jit`` kernel.

Triton 3.1 stores every stage on ``CompiledKernel.asm`` after the first
launch. Stages, in order, are the NVIDIA backend in
``triton/backends/nvidia/compiler.py``::

    ttir -> ttgir -> llir -> ptx -> cubin

See ``docs/triton_mlir_path.md``.
"""
from __future__ import annotations

from pathlib import Path

STAGES = ("ttir", "ttgir", "llir", "ptx", "cubin")


def compiled_kernels(jit_fn):
    kernels = []
    for device_cache in jit_fn.cache.values():
        kernels.extend(device_cache.values())
    return kernels


def dump_kernel(jit_fn, out_dir: Path) -> list[Path]:
    """Write ``<jit_fn>.{ttir,ttgir,llir,ptx,cubin}`` for the latest specialization."""
    kernels = compiled_kernels(jit_fn)
    if not kernels:
        raise RuntimeError(f"{jit_fn.__name__} has not been launched; no IR to dump")
    kernel = kernels[-1]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in STAGES:
        payload = kernel.asm.get(ext)
        if payload is None:
            continue
        path = out_dir / f"{jit_fn.__name__}.{ext}"
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload)
        written.append(path)
    return written
