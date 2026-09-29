"""Hopper SM90 schedule constants.

``mlir/runtime/gemm_swiglu_sm90.cu`` is the CUTLASS 3.x CollectiveBuilder
SwiGLU for sm_90a: a gate GEMM whose epilogue is SiLU, an up GEMM, then
``D = silu(X @ W1) * (X @ W3)``. Below sm_90 that entry calls the sm_70
implementation in ``gemm_swiglu_sm70.cu``. This is not an attention kernel.
"""
SMEM_BYTES = 228 * 1024
CLUSTER_SHAPE = (2, 1, 1)
PIPELINE_STAGES = 3  # GMEM(TMA) -> SMEM -> WGMMA registers

HOPPER = {
    "arch": 90,
    "smem_bytes": SMEM_BYTES,
    "cluster_shape": CLUSTER_SHAPE,
    "pipeline_stages": PIPELINE_STAGES,
    "mainloop": "KernelTmaWarpSpecializedCooperative",
    "mma": "wgmma.mma_async",
    "copy": "cp.async.bulk.tensor",
}


def require_sm90() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            "Hopper TMA/WGMMA SwiGLU needs sm_90. "
            "On this GPU use triton_llm.ops.swiglu or cutlass_gemm_swiglu."
        )
