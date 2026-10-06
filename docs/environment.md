# Environment

- GPUs: Tesla V100 (sm_70)
- CUDA Toolkit: **12.x** required for Volta (CUDA 13 drops sm_70)
- Compiler LLVM: Triton's pin in `cmake/llvm-hash.txt`, `10dc3a8e` (LLVM 19). `scripts/build_libtriton.sh` builds `libtriton.so` and `build/bin/triton-opt` against it.
- FileCheck: `/opt/torch-mlir/externals/llvm-project/build/bin/FileCheck`. Text comparison only; it is not linked into the compiler.
- Python: Torch 2.5.1 + Triton 3.1.0 in `.venv`. `TRITON_LLM_ENABLE_TRITON_TESTS` defaults ON, so `ctest` runs pytest with `.venv/bin/python` when that interpreter exists.
