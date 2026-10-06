# TritonLLMCompiler

Triton 3.1 路线的 LLM 算子与编译器开发仓库。默认目标是 NVIDIA Volta sm_70（Tesla V100，CUDA 12.x）。IR 使用 Triton 自己的 `tt`（TTIR）和 `triton_gpu`（TTGIR），不经过 StableHLO、XLA 或 Linalg。

能直接运行的有三条。Triton 的 C++ 源码不在本仓库里，通过 submodule 引进，本地改动只留补丁。说明见 [`third_party/patches/README.md`](third_party/patches/README.md)。

| 路径 | 命令 | 作用 |
|---|---|---|
| 本仓库编译的 JIT | `scripts/build_libtriton.sh`，`.venv` 指向它 | `@triton.jit`：DSL 建 op、TTIR、TTGIR、LLVM、PTX、cubin |
| Python pass 驱动 | `python -m triton_llm.tt_opt` | 用上一行那份 `libtriton.so` 单步跑上游 pass，包括 `triton_gpu` |
| 同一次 cmake 的 opt | `./build/bin/triton-opt` | 和 `libtriton.so` 同一份 LLVM 19、同一套 pass，读 `.mlir` 做 FileCheck |
| Triton 源码 | submodule [`third_party/triton`](third_party/patches/README.md) | 不进本仓库的 git 历史。v3.1.0（`cf34004b`）。补丁在 [`third_party/patches/`](third_party/patches/README.md)，由 `scripts/build_libtriton.sh` 打进 `build/triton-src` |

`@triton.jit` 和 `python -m triton_llm.tt_opt` 调用同一份 `libtriton.so`。`build/bin/triton-opt` 是同一次 cmake 编出的命令行。改一个文件之后哪条路径会重新编译，见 [`docs/triton_mlir_path.md`](docs/triton_mlir_path.md) 第 5 节。

## 从这里开始

```bash
git submodule update --init third_party/triton
scripts/setup_venv.sh
source .venv/bin/activate
cmake -B build -DTRITON_LLM_CUDA_ARCH=70
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

`scripts/build_all.sh` 依次做 cmake（带 `-DTRITON_LLM_ENABLE_SM90=ON`）、编译、ctest，再跑一次 pytest。

`setup_venv.sh` 往 `.venv` 里装两个包。第一步调用 `scripts/build_libtriton.sh`：打 `0001-frontend-llvm19.patch` 和 `0002-ttir-passes-llvm19.patch`，用 Triton 固定的 LLVM 19 编出 `libtriton.so`，用 `pip install -e build/triton-src/python` 把编译器 `triton` 装进 `.py-triton`，把 `triton-opt` 复制到 `build/bin/triton-opt`，再让 `.venv` 指向 `.py-triton`。第二步用 `pip install --no-deps -e .` 把本仓库的 `triton_llm`（`python/triton_llm`）装进同一个 `.venv`。两个包的区别见 [`docs/triton_mlir_path.md`](docs/triton_mlir_path.md) 第 1 节。

没有 `build/bin/triton-opt` 时，`test_python_cpp_handoff.py` 和 `test_tile_table_sync.py` 会 skip；`./tests/shell/run_tt_opt_tests.sh` 会失败。先跑 `scripts/build_libtriton.sh`。

## 环境

| 项 | 值 |
|---|---|
| GPU | Tesla V100，compute capability 7.0 |
| CUDA | 12.x。CUDA 13 去掉 sm_70 |
| Python | 3.10+ |
| LLVM | Triton 3.1.0 在 `cmake/llvm-hash.txt` 里固定的 `10dc3a8e`（LLVM 19）。`scripts/build_libtriton.sh` 用这份 LLVM 编 `libtriton.so` 和 `build/bin/triton-opt` |
| FileCheck | `/opt/torch-mlir/externals/llvm-project/build/bin/FileCheck`。只比对文本，不参与编译。`FILECHECK=` 可覆盖 |

确认解释器和 GPU：

```bash
.venv/bin/python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```

| CMake 变量 | 默认 | 作用 |
|---|---|---|
| `TRITON_LLM_CUDA_ARCH` | `70` | `CMAKE_CUDA_ARCHITECTURES` |
| `TRITON_LLM_ENABLE_CUDA` | `ON` | 编 CUTLASS 运行时。找不到 nvcc 时关掉 |
| `TRITON_LLM_ENABLE_SM90` | `ON` | 额外编 `triton_llm_cutlass_sm90`，`CUDA_ARCHITECTURES` 为 `90a` |
| `TRITON_LLM_BUILD_TESTS` | `ON` | 注册 ctest |
| `TRITON_LLM_ENABLE_TRITON_TESTS` | `ON` | 把 `pytest tests/python` 注册为 `triton_ops_pytest` |

configure 且 `TRITON_LLM_ENABLE_CUDA=ON` 时，[`cmake/FindOrFetchCutlass.cmake`](cmake/FindOrFetchCutlass.cmake) 把 CUTLASS v3.5.1 头文件拉进构建树，不放进 `third_party/`。

## Triton 算子

包名 `triton-llm`，版本 0.2.0，需要 GPU。下面四个函数可以从 `triton_llm` 或 `triton_llm.ops` 导入。

```python
import torch
from triton_llm import flash_attention, paged_attention, rmsnorm_rope, swiglu

q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
flash_attention(q, q, q, causal=False)

x = torch.randn(16, 32, device="cuda", dtype=torch.float16)
w = torch.randn(32, 32, device="cuda", dtype=torch.float16)
swiglu(x, w, w)
```

| 函数 | 形状 | 实现 |
|---|---|---|
| `flash_attention(q, k, v, *, causal=False, sm_scale=None)` | `[B,H,S,D]` | [`ops/flash_attention.py`](python/triton_llm/ops/flash_attention.py)。tile 固定 64×64、2 stage（`flash_attention_tile`） |
| `paged_attention(q, k_cache, v_cache, block_tables, context_lens, *, block_size, sm_scale=None)` | `q` 为 `[B,H,1,D]` | [`ops/paged_attention.py`](python/triton_llm/ops/paged_attention.py)。只做 decode。`D` 为 16/32/64/128，`block_size` 为 16/32/64。query 补成 16 行后再做 `tl.dot` |
| `rmsnorm_rope(hidden, weight, cos, sin)` | hidden `[B,S,H,D]` | [`ops/rmsnorm_rope.py`](python/triton_llm/ops/rmsnorm_rope.py)。一个 kernel 做完 RMSNorm 和 RoPE。最后一维分成两半分别加载，方差用两半一起算 |
| `swiglu(x, w1, w3)` | `x` `[M,K]`，`w1`/`w3` `[K,N]` | [`ops/swiglu.py`](python/triton_llm/ops/swiglu.py)。`silu(x @ w1) * (x @ w3)`。`BLOCK_*` 和 `num_stages` 来自当前设备的 `choose_gemm_tile` |

数值参考在 [`python/triton_llm/reference/`](python/triton_llm/reference/)，不参与编译。FP16 比较用 atol 1e-3、rtol 1e-2。

Shared memory 和 stage 上限：sm_70 见 [`arch/sm70.py`](python/triton_llm/arch/sm70.py)，96 KiB、最多 2 stage。sm 80–89 见 [`arch/tiling.py`](python/triton_llm/arch/tiling.py)，164 KiB、3 stage。sm≥90 见 [`arch/hopper.py`](python/triton_llm/arch/hopper.py)，228 KiB、3 stage。GEMM 候选表与 C++ 的 `gemmTileForSm` 相同。

## 上游 pass：`python -m triton_llm.tt_opt`

实现在 [`python/triton_llm/tt_opt.py`](python/triton_llm/tt_opt.py) 和 [`compiler/pipeline.py`](python/triton_llm/compiler/pipeline.py)。它读 `.mlir`，调用已安装 `libtriton.so` 里的 `passes.*.add_*`。`--make-ttir` 和 `--make-ttgir` 的顺序与附带源码里的 `CUDABackend` 相同，不含本仓库的手写 pass。sm_70 上 `--make-ttgir` 不会跑 software pipeline、f32-dot-tc、fence 和 TMA。

```bash
python -m triton_llm.tt_opt --list --sm 70
python -m triton_llm.tt_opt tests/tt/combine_addptr.mlir --triton-combine
python -m triton_llm.tt_opt tests/tt/reorder_broadcast.mlir --triton-reorder-broadcast
python -m triton_llm.tt_opt --sm 70 --num-warps 4 --num-stages 2 --num-ctas 1 \
  artifacts/triton_ir/swiglu/_swiglu_kernel.ttir --make-ttgir
```

默认 `--sm 70`、`--num-warps 4`、`--num-stages 3`、`--num-ctas 1`。`--list` 打印全部 flag，包括下面四个手写 pass。

## 手写 pass：`./build/bin/triton-opt`

入口是上游 `bin/triton-opt.cpp`。`0002` 把四个 pass 写进 `Passes.td`，`RegisterTritonDialects.h` 注册方言之后调用 `MlirOptMain`。`scripts/build_libtriton.sh` 把编出的二进制复制到 `build/bin/triton-opt`。它和 `libtriton.so` 是同一份 C++、同一份 LLVM 19。

```bash
./build/bin/triton-opt --help
./build/bin/triton-opt tests/tt/annotate_dot_stages.mlir --triton-annotate-dot-stages
./build/bin/triton-opt tests/tt/annotate_dot_stages_sm90.mlir --triton-annotate-dot-stages="sm=90"
./build/bin/triton-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue
./build/bin/triton-opt tests/tt/fuse_then_tile.mlir \
  --triton-fuse-dot-epilogue --triton-lower-fused-dot-mul --triton-tile-dot
./build/bin/triton-opt tests/tt/tile_dot.mlir --triton-tile-dot
./build/bin/triton-opt tests/tt/tile_dot_sm90.mlir --triton-tile-dot="sm=90"
./build/bin/triton-opt tests/tt/fused_dot_mul_verify.mlir -split-input-file -verify-diagnostics
./build/bin/triton-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue --mlir-print-ir-after-all
```

| Flag | 源码 | 作用 |
|---|---|---|
| `--canonicalize` | 上游 MLIR | 与下面四个一起注册 |
| `--triton-annotate-dot-stages` | `AnnotateDotStages.cpp` | 每个 `tt.dot` 写 `triton_llm.num_stages`：`sm<80` 为 2，否则 3 |
| `--triton-fuse-dot-epilogue` | `FuseAndTileDot.cpp` | 单次使用的 `tt.dot` 与同类型 `arith.mulf` 收成 `tt.fused_dot_mul` |
| `--triton-lower-fused-dot-mul` | 同上 | 展开回 `tt.dot` 和循环外的 `arith.mulf` |
| `--triton-tile-dot` | 同上 | 静态二维 `tt.dot` 在 K 能被 `BLOCK_K` 整除时改写成 `scf.for` + `tensor.extract_slice`。`triton_llm.block_k` 和 `triton_llm.num_stages` 写在 `scf.for` 上。这份 IR 停在 TTIR |

四个手写 TTIR pass 在 `libtriton.so` 和 `build/bin/triton-opt` 里都能用同名 flag 调用。`CUDABackend.make_ttir` 不跑它们。lesson kernel 用 `choose_gemm_tile` 的 `BLOCK_*`，由这份 `.so` 经 TTGIR 到 `mma.sync.aligned.m8n8k4`，见 [`docs/triton_mlir_path.md`](docs/triton_mlir_path.md) 第 3.3 节。`tl.chip_rcp` 见第 3.5 节。假的自有 GPU backend 在 `python/triton_llm/backend/fakegpu/`，见第 4 节。

[`tests/shell/run_tt_opt_tests.sh`](tests/shell/run_tt_opt_tests.sh) 执行 `tests/tt/*.mlir` 里的 `// RUN:`。它先把单词 `tt-opt` 换成 `python -m triton_llm.tt_opt`，再把 `TT_OPT_CPP` 换成 `build/bin/triton-opt`。手写 pass 的测试要写 `TT_OPT_CPP`。新增 pass 怎样同时进 `.so` 和 `triton-opt`，见 [`docs/triton_mlir_path.md`](docs/triton_mlir_path.md) 第 4 节。

## IR 样例

下面两份 TTIR 所处的阶段不同。

| 阶段 | 怎么得到 | 里面有什么 |
|---|---|---|
| `make_ttir` 之后 | `python scripts/dump_triton_ir.py` | `artifacts/triton_ir/<op>/*.{ttir,ttgir,llir,ptx,cubin}`。`tt.make_tensor_ptr` 已经不在 |
| `make_ttir` 之前 | [`compiler/frontend.py`](python/triton_llm/compiler/frontend.py) 的 `capture_frontend_ttir` | [`compiler/lesson.py`](python/triton_llm/compiler/lesson.py) 生成的模块，仍有 `tt.make_tensor_ptr` 和 `llvm.mlir.undef` |

`dump_triton_ir.py` 需要 CUDA，会编译 flash attention、paged attention、RMSNorm+RoPE 和 SwiGLU。`artifacts/` 不进版本库。把 `make_ttir` 之后的文本送进 TTGIR pass：

```bash
python -m triton_llm.tt_opt artifacts/triton_ir/swiglu/_swiglu_kernel.ttir \
  --convert-triton-to-tritongpu --tritongpu-accelerate-matmul
```

打印已安装 Triton 每个 pass 前后的 IR。命中编译缓存时 pass 不会重跑，所以要加上 `TRITON_ALWAYS_COMPILE`：

```bash
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python scripts/dump_triton_ir.py
```

缓存目录用 `TRITON_CACHE_DIR` 指定。

## CUTLASS 运行时

C API 在 [`mlir/runtime/cutlass_c_api.h`](mlir/runtime/cutlass_c_api.h)。两个 GEMM 函数的输入输出是行主序 FP32，计算用 FP16 tensor core，结果是 `silu(X@W1)*(X@W3)`。

| C API | 实现 | 行为 |
|---|---|---|
| `cutlass_gemm_swiglu` | [`gemm_swiglu_sm70.cu`](mlir/runtime/gemm_swiglu_sm70.cu) | 两次 GEMM，再加一个独立的 SiLU*mul kernel |
| `cutlass_gemm_swiglu_sm90` | [`gemm_swiglu_sm90.cu`](mlir/runtime/gemm_swiglu_sm90.cu) | gate GEMM 在 epilogue 里做 SiLU，up GEMM 不做激活，然后两者相乘。设备 cc&lt;90 时调用 `cutlass_gemm_swiglu` |
| `cutlass_rmsnorm_rope` | [`rmsnorm_rope.cu`](mlir/runtime/rmsnorm_rope.cu) | 融合 RMSNorm+RoPE，不是 GEMM |

CMake 目标 `triton_llm_cutlass_runtime` 生成 `libcutlass_runtime.so`。sm90 的目标是 `triton_llm_cutlass_sm90`。正确性程序在 `build/mlir/runtime/`，由 ctest 调用。

```bash
ctest --test-dir build -R cutlass --output-on-failure
scripts/verify_sm90_static.sh
```

`cutlass_swiglu_sm90_correctness_test` 在设备 cc&lt;90 时打印 `SKIP` 并返回 0。cc≥90 时与 `silu(X@W1)*(X@W3)` 比较，atol 为 2e-2。`verify_sm90_static.sh` 只检查构建目录里有没有文件名包含 `cutlass_sm90` 的文件。

## 测试

```bash
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
./tests/shell/run_tt_opt_tests.sh
pytest tests/python -q
scripts/profile_ncu.sh
```

ctest 名称：`tt_opt_ir_tests`、`triton_ops_pytest`、`cutlass_swiglu_correctness_test`、`cutlass_rmsnorm_rope_correctness_test`、`cutlass_swiglu_sm90_correctness_test`。无 GPU 时，标了 `@pytest.mark.cuda` 的用例 skip（[`tests/python/conftest.py`](tests/python/conftest.py)）。

| 检查 | 位置 | 结果 |
|---|---|---|
| 四个算子的数值 | `tests/python/test_flash_attention.py`、`test_paged_attention.py`、`test_rmsnorm_rope.py`、`test_swiglu.py` | atol 1e-3、rtol 1e-2 |
| tile 合法性 | `tests/python/test_tiling_config.py` | sm70 为 2 stage |
| C++ 与 Python 的 tile 表 | `tests/python/test_tile_table_sync.py` | sm 70、75、80、86、90 一致。没有 `triton-opt` 则 skip |
| 前端 op、blocked、MMA v1、`dot_op` | `tests/python/test_frontend_middle.py` | 需要 GPU |
| 前端 TTIR 交给 `triton-opt` | `tests/python/test_python_cpp_handoff.py` | 需要 GPU 和 `triton-opt` |
| JIT 打出 ttir/ttgir/llir/ptx | `tests/python/test_triton_ir_dump.py` | 需要 GPU |
| FileCheck 与 verifier 负例 | `tests/tt/*.mlir`，经 `run_tt_opt_tests.sh` | 需要 `triton-opt` 和 FileCheck |
| CUTLASS 数值 | 上面三个 `cutlass_*` ctest | sm90 在 cc&lt;90 时 SKIP |
| ncu | `scripts/profile_ncu.sh` | 只检查 `ncu` 是否在 PATH，不采集指标 |

完整标准见 [`docs/acceptance.md`](docs/acceptance.md)。

## 源码位置

| 路径 | 内容 |
|---|---|
| [`CMakeLists.txt`](CMakeLists.txt) | 选项、`common`、`mlir/runtime`、两个 ctest |
| [`scripts/build_libtriton.sh`](scripts/build_libtriton.sh) | 打 `0001`、`0002`，编 `libtriton.so` 和 `build/bin/triton-opt` |
| [`common/cuda_utils.cuh`](common/cuda_utils.cuh) | CUDA 错误检查 |
| [`python/triton_llm/ops/`](python/triton_llm/ops/) | 四个 `@triton.jit` kernel |
| [`python/triton_llm/reference/`](python/triton_llm/reference/) | 数值参考 |
| [`python/triton_llm/arch/`](python/triton_llm/arch/) | sm70 / sm90 常数和 tile 表 |
| [`python/triton_llm/compiler/`](python/triton_llm/compiler/) | pass 列表、`make_ttir` 之前的 TTIR、lesson kernel |
| [`python/triton_llm/tt_opt.py`](python/triton_llm/tt_opt.py) | `python -m triton_llm.tt_opt` |
| [`python/triton_llm/compiler_dump.py`](python/triton_llm/compiler_dump.py) | 把 `CompiledKernel.asm` 写成 `.ttir`、`.ttgir`、`.llir`、`.ptx`、`.cubin` |
| [`mlir/runtime/`](mlir/runtime/) | CUTLASS 设备 kernel 与 C API |
| [`tests/tt/`](tests/tt/) | MLIR FileCheck |
| [`tests/python/`](tests/python/) | pytest |
| [`tests/cuda/`](tests/cuda/) | CUTLASS 正确性 |
| [`scripts/setup_venv.sh`](scripts/setup_venv.sh)、[`scripts/build_all.sh`](scripts/build_all.sh) | 环境与全量构建 |
| [`scripts/dump_triton_ir.py`](scripts/dump_triton_ir.py) | 编译四个 kernel 并写出各阶段 IR |
| [`scripts/verify_sm90_static.sh`](scripts/verify_sm90_static.sh)、[`scripts/profile_ncu.sh`](scripts/profile_ncu.sh) | sm90 产物是否存在；ncu 是否在 PATH |
| [`third_party/patches/`](third_party/patches/README.md) | 为什么用 submodule 和补丁，哪些源码参与编译，哪些只供阅读 |
| [`docs/`](docs/) | 见下表 |

不提交到 git：`.venv/`、`build/`（含 `bin/triton-opt`、`triton-src/`、`mlir/runtime/` 下的测试程序，以及 cmake 下载的 CUTLASS）、`artifacts/`。

| 文档 | 内容 |
|---|---|
| [`docs/triton_mlir_path.md`](docs/triton_mlir_path.md) | `libtriton.so` 的编译、加载、流水线和命令 |
| [`docs/architecture.md`](docs/architecture.md) | 两个 pass 驱动和 CUTLASS 各做什么 |
| [`docs/environment.md`](docs/environment.md) | 本机路径 |
| [`docs/acceptance.md`](docs/acceptance.md) | 验收门槛 |
| [`docs/spec.md`](docs/spec.md) | 默认目标 sm_70 |
