"""Pass list for ``tt-opt``.

Flags that call ``passes.ttir`` / ``passes.ttgpuir`` execute the C++ pass
inside the installed Triton 3.1 ``libtriton.so``. The source of each pass
is the file in ``third_party/triton/``. The handwritten
``--triton-annotate-dot-stages`` pass is ``build/bin/tt-opt``, not a flag
on this driver.

``--make-ttir`` and ``--make-ttgir`` expand to the order in
``CUDABackend.make_ttir`` / ``make_ttgir``. That order does not include
``add_annotate_dot_stages``. On ``sm_70`` the TTGIR expansion
omits ``add_f32_dot_tc``, ``add_pipeline``, ``add_fence_insertion``, and
``add_tma_lowering``, because ``capability // 10`` is 7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

@dataclass
class PassOptions:
    sm: int = 70
    num_warps: int = 4
    num_stages: int = 3
    num_ctas: int = 1


@dataclass
class PassStep:
    flag: str
    summary: str
    source: str
    run: Callable


def _cpp(add) -> Callable:
    def run(module, context, options: PassOptions) -> None:
        from triton._C.libtriton import ir

        pm = ir.pass_manager(context)
        add(pm, options)
        pm.run(module)

    return run


def _bind_ttir(method_name: str) -> Callable:
    def add(pm, options: PassOptions) -> None:
        from triton._C.libtriton import passes

        getattr(passes.ttir, method_name)(pm)

    return add


def _bind_common(method_name: str) -> Callable:
    def add(pm, options: PassOptions) -> None:
        from triton._C.libtriton import passes

        getattr(passes.common, method_name)(pm)

    return add


def _bind_ttgpu(method_name: str) -> Callable:
    def add(pm, options: PassOptions) -> None:
        from triton._C.libtriton import passes

        getattr(passes.ttgpuir, method_name)(pm)

    return add


def _add_convert(pm, options: PassOptions) -> None:
    from triton._C.libtriton import passes

    passes.ttir.add_convert_to_ttgpuir(
        pm, f"cuda:{options.sm}", options.num_warps, 32, options.num_ctas
    )


def _add_f32_dot_tc(pm, options: PassOptions) -> None:
    from triton._C.libtriton import passes

    passes.ttgpuir.add_f32_dot_tc(pm)


def _add_plan_cta(pm, options: PassOptions) -> None:
    from triton._C.libtriton import nvidia

    cluster = nvidia.ClusterInfo()
    nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster)


def _add_optimize_dot_operands(pm, options: PassOptions) -> None:
    from triton._C.libtriton import passes

    passes.ttgpuir.add_optimize_dot_operands(pm, options.sm >= 80)


def _add_pipeline(pm, options: PassOptions) -> None:
    from triton._C.libtriton import passes

    passes.ttgpuir.add_pipeline(pm, options.num_stages)


def _add_fence(pm, options: PassOptions) -> None:
    from triton._C.libtriton import nvidia

    nvidia.passes.ttnvgpuir.add_fence_insertion(pm)


def _add_tma(pm, options: PassOptions) -> None:
    from triton._C.libtriton import nvidia

    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)


def step(flag: str, summary: str, source: str, add) -> PassStep:
    return PassStep(flag, summary, source, _cpp(add))


# Individual passes. ``--make-ttir`` / ``--make-ttgir`` reference these flags.
STEPS = [
    step("--inline", "MLIR inliner", "upstream MLIR", _bind_common("add_inliner")),
    step(
        "--triton-rewrite-tensor-pointer",
        "tt.make_tensor_ptr / tt.advance -> tt.load of tensor<!tt.ptr>",
        "third_party/triton/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp",
        _bind_ttir("add_rewrite_tensor_pointer"),
    ),
    step(
        "--triton-combine",
        "fold addptr(addptr) when the offsets are i64 or constant",
        "third_party/triton/lib/Dialect/Triton/Transforms/Combine.cpp",
        _bind_ttir("add_combine"),
    ),
    step("--canonicalize", "MLIR canonicalizer", "upstream MLIR", _bind_common("add_canonicalizer")),
    step(
        "--triton-reorder-broadcast",
        "move broadcast/splat below elementwise ops",
        "third_party/triton/lib/Dialect/Triton/Transforms/ReorderBroadcast.cpp",
        _bind_ttir("add_reorder_broadcast"),
    ),
    step("--cse", "MLIR CSE", "upstream MLIR", _bind_common("add_cse")),
    step("--licm", "MLIR LICM", "upstream MLIR", _bind_common("add_licm")),
    step("--symbol-dce", "MLIR symbol DCE", "upstream MLIR", _bind_common("add_symbol_dce")),
    step(
        "--convert-triton-to-tritongpu",
        "tt -> triton_gpu, attach blocked layouts",
        "third_party/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp",
        _add_convert,
    ),
    step(
        "--tritongpu-coalesce",
        "coalesce loads/stores",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp",
        _bind_ttgpu("add_coalesce"),
    ),
    step(
        "--tritongpu-f32-dot-tc",
        "tf32 dot lowering; make_ttgir calls this only when sm//10 >= 8",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp",
        _add_f32_dot_tc,
    ),
    step(
        "--tritongpu-plan-cta",
        "CTA planning",
        "third_party/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp",
        _add_plan_cta,
    ),
    step(
        "--tritongpu-remove-layout-conversions",
        "delete redundant convert_layout",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp",
        _bind_ttgpu("add_remove_layout_conversions"),
    ),
    step(
        "--tritongpu-optimize-thread-locality",
        "thread-locality",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp",
        _bind_ttgpu("add_optimize_thread_locality"),
    ),
    step(
        "--tritongpu-accelerate-matmul",
        "pick MMA layout for tt.dot",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp",
        _bind_ttgpu("add_accelerate_matmul"),
    ),
    step(
        "--tritongpu-optimize-dot-operands",
        "dot operand layout",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp",
        _add_optimize_dot_operands,
    ),
    step(
        "--tritongpu-combine-tensor-select-and-if",
        "combine select and if",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/CombineTensorSelectAndIf.cpp",
        _bind_ttgpu("add_combine_tensor_select_and_if"),
    ),
    step(
        "--tritongpu-pipeline",
        "software pipeline; make_ttgir calls this only when sm//10 >= 8",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp",
        _add_pipeline,
    ),
    step(
        "--tritongpu-prefetch",
        "prefetch",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/Prefetch.cpp",
        _bind_ttgpu("add_prefetch"),
    ),
    step(
        "--tritongpu-reduce-data-duplication",
        "reduce data duplication",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/ReduceDataDuplication.cpp",
        _bind_ttgpu("add_reduce_data_duplication"),
    ),
    step(
        "--tritongpu-reorder-instructions",
        "reorder instructions",
        "third_party/triton/lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp",
        _bind_ttgpu("add_reorder_instructions"),
    ),
    step(
        "--tritongpu-fence-insertion",
        "fence; make_ttgir calls this only when sm//10 >= 9",
        "third_party/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp",
        _add_fence,
    ),
    step(
        "--tritongpu-tma-lowering",
        "TMA; make_ttgir calls this only when sm//10 >= 9",
        "third_party/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp",
        _add_tma,
    ),
]

BY_FLAG = {s.flag: s for s in STEPS}

MAKE_TTIR = [
    "--inline",
    "--triton-rewrite-tensor-pointer",
    "--triton-combine",
    "--canonicalize",
    "--triton-reorder-broadcast",
    "--cse",
    "--licm",
    "--symbol-dce",
]


def make_ttgir(sm: int) -> list[str]:
    """Same guards as ``CUDABackend.make_ttgir``."""
    flags = [
        "--convert-triton-to-tritongpu",
        "--tritongpu-coalesce",
    ]
    if sm // 10 >= 8:
        flags.append("--tritongpu-f32-dot-tc")
    flags.extend(
        [
            "--tritongpu-plan-cta",
            "--tritongpu-remove-layout-conversions",
            "--tritongpu-optimize-thread-locality",
            "--tritongpu-accelerate-matmul",
            "--tritongpu-remove-layout-conversions",
            "--tritongpu-optimize-dot-operands",
            "--cse",
        ]
    )
    if sm // 10 >= 8:
        flags.append("--tritongpu-combine-tensor-select-and-if")
        flags.append("--tritongpu-pipeline")
    flags.extend(
        [
            "--tritongpu-prefetch",
            "--tritongpu-optimize-dot-operands",
            "--tritongpu-remove-layout-conversions",
            "--tritongpu-reduce-data-duplication",
            "--tritongpu-reorder-instructions",
            "--cse",
            "--symbol-dce",
        ]
    )
    if sm // 10 >= 9:
        flags.append("--tritongpu-fence-insertion")
        flags.append("--tritongpu-tma-lowering")
    flags.append("--canonicalize")
    return flags


def format_catalog(sm: int) -> str:
    lines = ["passes:"]
    for item in STEPS:
        lines.append(f"  {item.flag}")
        lines.append(f"    {item.summary}")
        lines.append(f"    {item.source}")
    lines.append("")
    lines.append("make-ttir:")
    lines.extend(f"  {flag}" for flag in MAKE_TTIR)
    lines.append("")
    lines.append(f"make-ttgir (sm_{sm}):")
    lines.extend(f"  {flag}" for flag in make_ttgir(sm))
    return "\n".join(lines)


def expand_flags(argv_flags: list[str], sm: int) -> list[str]:
    expanded: list[str] = []
    for flag in argv_flags:
        if flag == "--make-ttir":
            expanded.extend(MAKE_TTIR)
        elif flag == "--make-ttgir":
            expanded.extend(make_ttgir(sm))
        else:
            expanded.append(flag)
    return expanded
