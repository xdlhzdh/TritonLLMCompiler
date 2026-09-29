"""Volta SM70 limits used to prune Triton launch configs."""
SMEM_BYTES = 96 * 1024
MAX_REGS_PER_THREAD = 255
# cp.async / TMA are not available. Pipeline depth stays at 2.
MAX_PIPELINE_STAGES = 2

SM70 = {
    "arch": 70,
    "smem_bytes": SMEM_BYTES,
    "max_regs_per_thread": MAX_REGS_PER_THREAD,
    "max_pipeline_stages": MAX_PIPELINE_STAGES,
}
