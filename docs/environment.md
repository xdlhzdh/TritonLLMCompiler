# Environment

- GPUs: Tesla V100 (sm_70)
- CUDA Toolkit: **12.x** required for Volta (CUDA 13 drops sm_70)
- FileCheck: `/opt/torch-mlir/externals/llvm-project/build/bin/FileCheck`
- MLIR for `tt-opt`: `/opt/torch-mlir/externals/llvm-project/build` (LLVM 23). CMake looks for `lib/cmake/mlir` under that prefix.
- Python: Torch 2.5.1 + Triton 3.1.0 in `.venv`. `TRITON_LLM_ENABLE_TRITON_TESTS` defaults ON, so `ctest` runs pytest with `.venv/bin/python` when that interpreter exists.
