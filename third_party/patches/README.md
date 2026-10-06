# Triton 源码怎么引进

本仓库不保存 Triton 的 C++ / Python 源码。`third_party/triton` 是一个 git submodule，指向 [triton-lang/triton](https://github.com/triton-lang/triton) 的 v3.1.0，commit `cf34004b8a67d290a962da166f5aa2fc66751326`。本仓库自己只提交 `third_party/patches/*.patch`。

```bash
git submodule update --init third_party/triton
```

不要加 `--recursive`。Triton 自己的 LLVM submodule 不需要。`scripts/build_libtriton.sh` 按 `cmake/llvm-hash.txt` 下载 Triton 固定的 LLVM 19（`10dc3a8e`）。

## 为什么用补丁

submodule 里的树必须和上游那个 commit 一致。本地改动如果直接写进 submodule，下一次 `git submodule update` 会丢掉，也分不清哪些行是上游的、哪些是我们的。

补丁把这两件事分开：

- submodule 里 `git diff` 是空的，说明源码没有被改过。
- 本仓库相对上游的全部改动都在 `*.patch` 里。看改了哪些文件：

```bash
grep '^diff' third_party/patches/*.patch
```

升级 Triton 时，把 submodule 指到新的 commit，再在新树上重放补丁。重放失败的地方就是要手工合并的冲突，不会混在几千个上游文件里。

## 构建时怎么用这棵源码

`scripts/build_libtriton.sh` 做这些事：

1. 检查 `third_party/triton` 已经初始化。
2. 把它复制到 `build/triton-src`。这个目录在 `build/` 里，不进 git。
3. 按顺序打 `0001-frontend-llvm19.patch`、`0002-ttir-passes-llvm19.patch`。
4. 用 Triton 固定的 LLVM 19 编出 `libtriton.so` 和 `build/bin/triton-opt`。

submodule 的工作区保持上游原样。改 pass 时改 `build/triton-src` 里的文件，确认行为之后，用这份副本相对 `third_party/triton` 重新生成补丁，替换 `third_party/patches/` 里的文件。`0002` 是打在已经打过 `0001` 的树上的，重新生成时要先打 `0001`。

`@triton.jit` 和 `python -m triton_llm.tt_opt` 读的是这一次编出的 `libtriton.so`。`build/bin/triton-opt` 是同一次 cmake 的命令行。

## 哪些参与本仓库的编译

参与编译的是 `build/triton-src` 上 Triton 自己的 cmake 编进 `libtriton.so` 和 `triton-opt` 的那些文件，再加上两份补丁改过的文件。

`0001-frontend-llvm19.patch` 只加 `tl.chip_rcp`。改了 `python/triton/language/math.py`、`python/triton/language/__init__.py`、`python/src/ir.cc`、`lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` 和 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`。`@triton.jit` 调用 `tl.chip_rcp` 时走 `create_chip_rcp`，再经 TTGIR 到 PTX `rcp.approx.ftz.f32`。

`0002-ttir-passes-llvm19.patch` 把四个 TTIR pass 和 `tt.fused_dot_mul` 编进同一份 `.so` 和 `build/bin/triton-opt`。源文件是 `AnnotateDotStages.cpp`、`FuseAndTileDot.cpp`，flag 在 `Passes.td`，Python 入口在 `passes.cc`。

## 哪些只供阅读

submodule 里其余文件不参与这次编译，用来对照上游 Triton 3.1.0 的实现。阅读时以 submodule 里的路径为准；被补丁改过的文件以 `build/triton-src` 里的副本为准。

| 路径 | 读它是为了看什么 |
|---|---|
| `python/triton/compiler/`、`python/triton/language/`、`python/triton/runtime/` | AST 怎么变成 TTIR。`tl.chip_rcp` 的改动在补丁打进 `build/triton-src` 之后的 `language/math.py` 和 `__init__.py` |
| `third_party/nvidia/backend/compiler.py` | `make_ttir` / `make_ttgir` / `make_llir` 的 pass 顺序。补丁不改这个文件 |
| `python/src/passes.cc`、`ir.cc`、`llvm.cc` | Python 怎么通过 pybind 调用 C++。`create_chip_rcp` 和 `add_annotate_dot_stages` 在补丁里 |
| `lib/Conversion/`、`lib/Dialect/TritonGPU/Transforms/`、`lib/Dialect/TritonNvidiaGPU/` | TTIR 降到 TTGIR，以及布局、流水、TMA。`tt.chip_rcp` 的 `GenericOpPattern` 在补丁打进 `build/triton-src` 之后的 `TritonToTritonGPUPass.cpp` |
| `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/` | TTGIR 降到 LLVM。Volta 的 `mma.sync` 在 `DotOpToLLVM/MMAv1.cpp`。`tt.chip_rcp` 的 `ChipRcpOpConversion` 在补丁打进之后的 `ElementwiseOpToLLVM.cpp` |
| `bin/triton-opt.cpp`、`test/` | 上游的 opt 工具和 lit。`scripts/build_libtriton.sh` 会编出 `build/bin/triton-opt`。本仓库不运行 lit |
| `third_party/amd/`、`third_party/proton/` | AMD 后端和 Proton。本仓库的目标是 NVIDIA sm_70，不使用 |

正在跑的 kernel 编译器是 `.venv` 加载的 `libtriton.so`，由 `build/triton-src` 编出。submodule 工作区本身不参与这次编译。
