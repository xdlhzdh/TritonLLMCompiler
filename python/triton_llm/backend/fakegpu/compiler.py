"""A stand-in GPU backend. It compiles TTIR with ``libtriton.so`` and then
emits a text listing. It does not build a GPU dialect, PTX, or a cubin.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from triton.backends.compiler import BaseBackend, GPUTarget

_FUNC_NAME = re.compile(r"tt\.func\s+(?:public\s+)?@([A-Za-z0-9_]+)")

# TTIR op → fake opcode. This table is the stand-in for
# ElementwiseOpToLLVM.cpp / DotOpToLLVM: add a row to lower one more op.
# Order of rows does not matter; emission follows the order ops appear in TTIR.
FAKE_OPCODES = (
    ("scf.for", "fake.loop"),
    ("scf.yield", "fake.yield"),
    ("tt.load", "fake.load"),
    ("tt.store", "fake.store"),
    ("tt.dot", "fake.dot"),
    ("tt.chip_rcp", "fake.rcp.f32"),
)
_OP_RE = re.compile(r"\b(" + "|".join(re.escape(op) for op, _ in FAKE_OPCODES) + r")\b")
_OPCODE = dict(FAKE_OPCODES)


@dataclass
class FakeOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    enable_fp_fusion: bool = True
    debug: bool = False
    shared: int = 0
    cluster_dims: tuple = (1, 1, 1)
    backend_name: str = "fakegpu"
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    max_num_imprecise_acc_default: int = 0

    def hash(self) -> str:
        key = f"{self.num_warps}-{self.num_ctas}-{self.num_stages}-{self.debug}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class FakeGPUBackend(BaseBackend):
    binary_ext = "fakegpu"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "fakegpu"

    def hash(self) -> str:
        return f"fakegpu-{self.target.arch}"

    def parse_options(self, opts) -> FakeOptions:
        fields = FakeOptions.__dataclass_fields__
        args = {key: opts[key] for key in fields if key in opts}
        return FakeOptions(**args)

    def pack_metadata(self, metadata):
        return (metadata.num_warps,)

    def get_codegen_implementation(self):
        return {}

    def load_dialects(self, context):
        return None

    @staticmethod
    def make_ttir(mod, metadata, opt):
        """Same TTIR cleanup as ``CUDABackend.make_ttir``. No handwritten passes."""
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        text = str(mod)
        found = _FUNC_NAME.search(text)
        if found:
            metadata["name"] = found.group(1)
        return mod

    @staticmethod
    def lower_ttir(text: str) -> str:
        """Emit one fake instruction per TTIR op, in the order the IR prints them."""
        lines = [
            "// fakegpu isa",
            "// FakeGPUBackend.lower_ttir; opcodes live in FAKE_OPCODES",
        ]
        for match in _OP_RE.finditer(text):
            lines.append(_OPCODE[match.group(1)])
        return "\n".join(lines) + "\n"

    @staticmethod
    def make_fakeasm(mod, metadata):
        return FakeGPUBackend.lower_ttir(str(mod))

    @staticmethod
    def make_fakebin(src, metadata):
        metadata["shared"] = 0
        metadata["cluster_dims"] = (1, 1, 1)
        metadata.setdefault("name", "fakegpu_kernel")
        return src.encode("utf-8")

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["fakeasm"] = lambda src, metadata: self.make_fakeasm(src, metadata)
        stages["fakegpu"] = lambda src, metadata: self.make_fakebin(src, metadata)
