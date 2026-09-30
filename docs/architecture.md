# Architecture

Pass 开发在 `tt` / `triton_gpu` 上。上游已有的 pass 用 `python -m triton_llm.tt_opt` 调 venv 里的 `libtriton.so`。`--make-ttir` / `--make-ttgir` 的顺序与引进的 `CUDABackend.make_ttir` / `make_ttgir` 相同，里面没有手写 pass。手写 pass 编进 `build/bin/tt-opt`：`AnnotateDotStages.cpp`（`--triton-annotate-dot-stages`），以及 `FuseAndTileDot.cpp`（`--triton-fuse-dot-epilogue`、`--triton-lower-fused-dot-mul`、`--triton-tile-dot`）。上游源码是 submodule `third_party/triton`，不进本仓库的提交。本地改动是 `third_party/patches/*.patch`，编译用的是 `build/triton-patched/`。哪些文件参与编译、哪些只供阅读，见 `third_party/patches/README.md`。`build/bin/tt-opt` 链接 LLVM 23，注册 `tt`、`arith`、`scf`、`tensor`、`llvm` 等方言，不含 `triton_gpu`，可以直接读 Python 抓到的前端 TTIR。它与 `libtriton.so` 之间只共享 IR 文本。JIT 怎么调用 `libtriton.so`、`triton-opt` 和它是什么关系，以及和工业界差在哪，见 `docs/triton_mlir_path.md` 第 1 节、第 2 节和第 2.2 节。

`@triton.jit` kernel 在 `python/triton_llm/ops/`。`compiler/lesson.py` 把 `program_id`、`arange`、指针、`dot`、`for`、`if`、`reduce`、`splat`、`make_block_ptr` 放进同一个 kernel；`compiler/frontend.py` 取出还没跑 `make_ttir` 的 TTIR。`scripts/dump_triton_ir.py` 写的是 `make_ttir` 之后的各 stage，目录是 `artifacts/triton_ir/`。

`mlir/runtime/` 是 CUTLASS 设备 kernel，结果都是 `silu(X@W1)*(X@W3)`。sm_70 是两次 GEMM 再加一个独立的 SiLU*mul kernel。sm_90a 是 gate GEMM（SiLU epilogue）加 up GEMM 再相乘；设备 compute capability 低于 9.0 时测试打印 SKIP 后退出 0，高于等于 9.0 时对这个公式做数值比较。
