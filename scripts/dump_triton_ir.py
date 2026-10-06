#!/usr/bin/env python3
"""Launch each Triton op once and write TTIR / TTGIR / LLVM IR / PTX.

Usage:
  python scripts/dump_triton_ir.py
  # files land in artifacts/triton_ir/<op>/
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from triton_llm.compiler_dump import dump_kernel  # noqa: E402
from triton_llm.ops.flash_attention import _flash_attn_fwd, flash_attention
from triton_llm.ops.paged_attention import _paged_attn_kernel, paged_attention
from triton_llm.ops.rmsnorm_rope import _rmsnorm_rope_kernel, rmsnorm_rope
from triton_llm.ops.swiglu import _swiglu_kernel, swiglu


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to compile Triton kernels")
    out = ROOT / "artifacts" / "triton_ir"
    b, h, s, d = 1, 1, 64, 64
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    flash_attention(q, q, q)
    x = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    w = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    swiglu(x, w, w)
    hidden = torch.randn(1, 8, 2, d, device="cuda", dtype=torch.float16)
    weight = torch.ones(d, device="cuda", dtype=torch.float16)
    cos = torch.ones(8, d, device="cuda", dtype=torch.float16)
    sin = torch.zeros(8, d, device="cuda", dtype=torch.float16)
    rmsnorm_rope(hidden, weight, cos, sin)
    block = 16
    q1 = torch.randn(1, 1, 1, d, device="cuda", dtype=torch.float16)
    cache = torch.randn(4, block, 1, d, device="cuda", dtype=torch.float16)
    tables = torch.arange(4, device="cuda", dtype=torch.int32).view(1, 4)
    lens = torch.tensor([32], device="cuda", dtype=torch.int32)
    paged_attention(q1, cache, cache, tables, lens, block_size=block)

    pairs = [
        ("flash_attention", _flash_attn_fwd),
        ("paged_attention", _paged_attn_kernel),
        ("rmsnorm_rope", _rmsnorm_rope_kernel),
        ("swiglu", _swiglu_kernel),
    ]
    for name, fn in pairs:
        paths = dump_kernel(fn, out / name)
        print(name)
        for p in paths:
            print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
