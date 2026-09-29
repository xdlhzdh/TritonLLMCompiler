# Triton 源码怎么引进

本仓库不保存 Triton 的 C++ / Python 源码。`third_party/triton` 是一个 git submodule，指向 [triton-lang/triton](https://github.com/triton-lang/triton) 的 v3.1.0，commit `cf34004b8a67d290a962da166f5aa2fc66751326`。本仓库自己只提交 `third_party/patches/*.patch`。

```bash
git submodule update --init third_party/triton
```

不要加 `--recursive`。Triton 自己的 LLVM submodule 不需要，本仓库用的是 `/opt/torch-mlir` 里的 LLVM 23。

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

`compiler/CMakeLists.txt` 在配置时做三件事：

1. 检查 `third_party/triton` 已经初始化。
2. 把它复制到 `build/triton-patched`。这个目录在 `build/` 里，不进 git。
3. 按文件名顺序对副本打上 `third_party/patches/*.patch`，然后只编译副本。

submodule 的工作区保持上游原样。改 pass 时改 `build/triton-patched` 里的文件，确认行为之后，用这份副本相对 `third_party/triton` 重新生成补丁，替换 `third_party/patches/` 里的文件。

`tt-opt` 链接的是这份打过补丁的副本和 LLVM 23，不链接 `.venv` 里的 `libtriton.so`。已安装的 JIT 也不读 submodule，也不读补丁。

## 哪些参与本仓库的编译

只有 `tt` 方言上编 `build/bin/tt-opt` 需要的部分。TableGen 输入是 `include/triton/Dialect/Triton/IR/`、`include/triton/Dialect/Triton/Transforms/`、`include/triton/Dialect/TritonGPU/IR/`。

编进静态库 `TritonIR` 的是：

- `lib/Dialect/Triton/IR/` 的 `Dialect.cpp`、`Ops.cpp`、`Traits.cpp`、`Types.cpp`
- `lib/Dialect/TritonGPU/IR/` 的 `Dialect.cpp`、`Types.cpp`、`LinearLayoutConversions.cpp`
- 补丁新增的 `lib/Analysis/ReshapeDecomposition.cpp`

`Traits.cpp` 会调用 `TritonGPUDialect::getNumWarps`，所以要带上那三份 GPU 的 `.cpp`。`tt-opt` 的解析器没有注册 `triton_gpu`，仍然不能读 TTGIR。

直接编进 `tt-opt` 的是补丁新增的 `AnnotateDotStages.cpp`、`FuseAndTileDot.cpp`，加上本仓库的 `compiler/tt-opt.cpp`。

## 哪些只供阅读

submodule 里其余文件不参与这次编译，用来对照上游 Triton 3.1.0 的实现。阅读时以 submodule 里的路径为准；被补丁改过的文件以 `build/triton-patched` 里的副本为准。

| 路径 | 读它是为了看什么 |
|---|---|
| `python/triton/compiler/`、`python/triton/language/`、`python/triton/runtime/` | AST 怎么变成 TTIR，`@triton.jit` 怎么启动编译 |
| `third_party/nvidia/backend/compiler.py` | `make_ttir` / `make_ttgir` 的 pass 顺序 |
| `python/src/passes.cc`、`ir.cc`、`llvm.cc` | Python 怎么通过 pybind 调用 C++ |
| `lib/Conversion/`、`lib/Dialect/TritonGPU/Transforms/`、`lib/Dialect/TritonNvidiaGPU/` | TTIR 降到 TTGIR，以及布局、流水、TMA |
| `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/` | TTGIR 降到 LLVM。Volta 的 `mma.sync` 在 `DotOpToLLVM/MMAv1.cpp` |
| `bin/triton-opt.cpp`、`test/` | 上游的 opt 工具和 lit。本仓库不编译、不运行它们 |
| `third_party/amd/`、`third_party/proton/` | AMD 后端和 Proton。本仓库的目标是 NVIDIA sm_70，不使用 |

正在跑的 kernel 编译器是 `.venv` 里的 `libtriton.so`。上表这些文件和它不是同一份二进制。
