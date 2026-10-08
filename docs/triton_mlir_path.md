# Triton 编译器是怎么跑起来的

TTIR 是 `tt` 方言，TTGIR 是 `triton_gpu` 方言。本仓库走 Triton 自己的 MLIR，不经过 StableHLO、XLA 或 Linalg。

源码是 submodule `third_party/triton`（v3.1.0，`cf34004b`）。本地改动是两份补丁，由 [`scripts/build_libtriton.sh`](../scripts/build_libtriton.sh) 打到 `build/triton-src`，再用 Triton 固定的 LLVM 19 编译。清单见 [`third_party/patches/README.md`](../third_party/patches/README.md)。

| 节 | 内容 |
|---|---|
| 1 | `triton` 和 `triton_llm` 两个包的区别和安装；`libtriton.so` 怎么编出来 |
| 2 | 这个 `.so` 何时加载，`@triton.jit` 的哪几行调用它 |
| 3 | 从 Python 函数到 cubin：建 op、TTIR pass、tile、`tl.chip_rcp` |
| 4 | `triton-opt` 与 `libtriton.so` 共用同一份 C++ |
| 5 | 按这个顺序运行 |
| 6 | 主要入口函数 |

## 1. 两个 Python 包：`triton` 和 `triton_llm`

`.venv` 里有两个包，各有一条 `pip install -e`。两条都由 [`scripts/setup_venv.sh`](../scripts/setup_venv.sh) 按顺序执行：

```bash
git submodule update --init third_party/triton
scripts/setup_venv.sh
source .venv/bin/activate
```

| | `triton` | `triton_llm` |
|---|---|---|
| 是什么 | Triton 编译器：DSL（`triton.language`）、编译流水线（`triton.compiler`）、运行时（`triton.runtime`）、C++ 扩展 `triton._C.libtriton` | 本仓库的代码：kernel、`tt_opt` 驱动 |
| 源码目录 | `build/triton-src/python`，即 submodule `third_party/triton` 打过补丁后的副本 | 本仓库的 `python/triton_llm` |
| 安装命令 | `pip install -e build/triton-src/python --no-build-isolation --no-deps` | `pip install --no-deps -e .`（在仓库根目录） |
| 谁执行 | [`scripts/build_libtriton.sh`](../scripts/build_libtriton.sh)，由 `setup_venv.sh` 第一步调用 | `setup_venv.sh` 第二步 |
| 编译 | 跑 cmake 和 ninja，产物是 `libtriton.so` | 纯 Python |

依赖只有一个方向：`triton_llm` 的代码 `import triton`。改 Triton 的 C++ 之后重跑 `scripts/build_libtriton.sh`，它重新编 `libtriton.so` 并重新安装 `triton`。

`triton_llm` 的 [`pyproject.toml`](../pyproject.toml) 把依赖写成 `triton==3.1.0`。安装 `triton_llm` 的那条命令因此必须带 `--no-deps`，否则 pip 会从 PyPI 下载官方 `triton` 包，替换刚编出来的这一份。

仓库根目录的 CMake 编 CUTLASS 运行时并注册 ctest，不编 `libtriton.so`。

### 1.1 `build_libtriton.sh`：编 `libtriton.so`，安装 `triton`

LLVM 用 [`third_party/triton/cmake/llvm-hash.txt`](../third_party/triton/cmake/llvm-hash.txt) 里的 `10dc3a8e`（LLVM 19）。脚本按以下顺序执行：

1. 把 submodule 复制到 `build/triton-src`（这份副本不进 git），先打 `0001-frontend-llvm19.patch`，再打 `0002-ttir-passes-llvm19.patch`。同时改这份副本的 `python/setup.py`：编译时不把警告当错误，不编 Triton 的单元测试，并且不编 proton。补丁内容一变，脚本就删掉 `build/triton-src`，重新复制。submodule 本身保持上游原样。
2. 没有 `.py-triton` 时，用 Python 3.12 创建这个 venv，并写入 `torch-from-tritonqattn.pth`。这个文件把提供 PyTorch 的那个环境的 `site-packages` 加进模块搜索路径，torch、numpy、pytest 都从那里来。那个环境本身不被修改。
3. 在 `.py-triton` 的 site-packages 里写入 `zz-prefer-built-triton.pth`，每次运行脚本都覆盖。文件名以 `zz-` 开头，Python 启动时会排在 pip 生成的 editable `.pth` 后面执行。它只有一行 `import`，把自编 `triton` 的 finder 移到 `sys.meta_path` 最前，因此 `import triton` 加载的是 `build/triton-src` 里的包。没有这个文件时，会加载到 PyTorch 那个环境里的官方 `triton`。
4. 用 `.py-triton/bin/pip install -e build/triton-src/python --no-build-isolation --no-deps` 编译并安装 `triton`。这一条命令同时跑 cmake/ninja 和 editable 安装，细节在 1.2 节。
5. 把同一次 cmake 编出的 `triton-opt` 复制到 `build/bin/triton-opt`。`triton-opt` 是命令行程序，不是 Python 模块。
6. `ln -sfn .py-triton .venv`，并确认 `triton.language.chip_rcp` 存在。

两份补丁：

| 补丁 | 作用 |
|---|---|
| `0001-frontend-llvm19.patch` | DSL 增加 `tl.chip_rcp`，经 TTGIR 降到 PTX `rcp.approx.ftz.f32`。见第 3.5 节 |
| `0002-ttir-passes-llvm19.patch` | 增加 `tt.fused_dot_mul`，以及四条只在命令行 flag 上运行的 TTIR pass。flag 和变换见第 3.2 节 |

### 1.2 一条 `pip install -e` 同时完成编译和安装

```bash
.py-triton/bin/pip install -e build/triton-src/python --no-build-isolation --no-deps
```

编译：pip 调用 `build/triton-src/python/setup.py` 的 `CMakeBuild.build_extension`，由它跑 cmake 和 ninja。cmake 的源码目录是 `build/triton-src`。`-DCMAKE_LIBRARY_OUTPUT_DIRECTORY` 指向包内的 `triton/_C/`，ninja 把 `.so` 写到 `build/triton-src/python/triton/_C/libtriton.so`。`build/triton-src/python/src/main.cc` 的 `PYBIND11_MODULE(libtriton, m)` 注册子模块 `ir`、`passes`、`llvm`、`interpreter` 和 `nvidia`。两份补丁里的 C++ 都链进这一个 `.so`。

安装：`-e` 是 editable 安装，源码留在 `build/triton-src/python`。pip 在 `.py-triton` 的 site-packages 里放 finder `__editable___triton_3_1_0_finder`。`import triton` 经这个 finder 找到 `build/triton-src/python/triton/__init__.py`，再加载同目录的 `_C/libtriton.so`。进程里的 `triton._C.libtriton` 就是 ninja 写出的那个文件。

两个参数：

- `--no-build-isolation`：在 `.py-triton` 里构建，pip 不另建临时环境。cmake 的构建目录留在 `build/triton-src/python/build`，重跑时只编改过的文件。
- `--no-deps`：不按 `setup.py` 的依赖列表再装包。torch 由 `torch-from-tritonqattn.pth` 指向的环境提供。这和安装 `triton_llm` 时的 `--no-deps` 是两回事，后者是为了避开 PyPI 上的 `triton==3.1.0`。

提供 PyTorch 的环境里另有一份官方 `triton`。`torch-from-tritonqattn.pth` 把那个 `site-packages` 加进模块搜索路径之后，按普通路径查找会先找到这份官方包。`zz-prefer-built-triton.pth` 把自编 finder 提前，`import triton` 才会用 `build/triton-src` 里的包。

### 1.3 确认两个包的来源

```bash
.venv/bin/python -c "import os, triton, triton_llm; print(triton.__file__); print(os.path.realpath(triton._C.libtriton.__file__)); print(triton_llm.__file__)"
```

前两行应在 `build/triton-src/python/triton/` 下，第三行应在本仓库的 `python/triton_llm/` 下。

## 2. `@triton.jit` 对 `libtriton.so` 的调用

`@triton.jit` 把函数包装成 `JITFunction`（`triton/runtime/jit.py`）。装饰器执行时不生成 IR。第一次 `kernel[grid](...)` 进入 `JITFunction.run`；`self.cache[device]` 里没有这份结果时，`run` 调用 `triton.compiler.compile`。

这次编译分成前端、中端、后端。`CUDABackend`（`nvidia/backend/compiler.py`）是 NVIDIA 目标的插件，`add_stages` 登记从 `make_ttir` 到 `make_cubin` 的阶段：中端是 `make_ttir` 和 `make_ttgir`，后端是 `make_llir`、`make_ptx`、`make_cubin`。

| 段 | 输入 → 输出 | 函数 |
|---|---|---|
| 前端 | Python AST → TTIR | `ASTSource.make_ir`、`ast_to_ttir` |
| 中端 | TTIR → 优化后的 TTIR → TTGIR | `CUDABackend.make_ttir`、`make_ttgir` |
| 后端 | TTGIR → LLVM IR → PTX → cubin | `CUDABackend.make_llir`、`make_ptx`、`make_cubin` |

### 2.1 编译前的准备

1. `JITFunction.run` 按实参的类型和 constexpr 值生成 cache key（不同的类型或 constexpr 各编一份 kernel），查 `self.cache[device]`。命中则直接 `kernel.run`。
2. 未命中时，`make_backend(driver.active.get_current_target())` 得到 `CUDABackend`，`parse_options` 得到编译选项。然后构造 `signature`、`constants` 和 `ASTSource`。
3. `compile()` 用同一个 `target` 再调用一次 `make_backend`，然后查磁盘缓存。命中则读回 `CompiledKernel`，不登记、不执行 stage。
4. 磁盘缓存未命中时，`CUDABackend.add_stages` 把 `make_ttir`、`make_ttgir`、`make_llir`、`make_ptx`、`make_cubin` 放进 `stages`。这一步只登记，不执行。跑完后写入磁盘缓存，`run` 再把 `CompiledKernel` 放进 `self.cache[device]`。

### 2.2 前端：Python AST → TTIR

`compile()` 调用 `ASTSource.make_ir`，再调用 `ast_to_ttir`。`CodeGenerator.visit` 遍历 Python AST，通过 `builder.create_*`（`.so` 中的 `ir.cc`）建 op。结果是还没有执行任何 pass 的 TTIR。见第 3.1 节。

### 2.3 中端：TTIR 优化，降到 TTGIR

`compile()` 按 `stages` 的顺序先执行这两段，每段都是 `passes.*.add_*` 加上 `pm.run`：

- `make_ttir`：TTIR 上与目标无关的优化，例如 inline、`triton-combine`、CSE。见第 3.2 节。
- `make_ttgir`：TTIR 转成 TTGIR，按 sm 给张量选 layout，把 `tt.dot` 标成 MMA layout。从这一段开始依赖目标。见第 3.3 节。

### 2.4 后端：TTGIR → cubin

- `make_llir`：TTGIR 降到 LLVM dialect，再转成 LLVM IR 并做 O3。
- `make_ptx`：LLVM 的 NVPTX 后端生成 PTX。
- `make_cubin`：`ptxas` 把 PTX 汇编成 cubin。

见第 3.4 节。

### 2.5 `libtriton.so` 何时加载

加载发生在 `import triton`，早于 `@triton.jit`。`triton/backends/__init__.py` 的 `_discover_backends()` 导入 `nvidia/backend/compiler.py`，该文件开头是：

```python
from triton._C.libtriton import ir, passes, llvm, nvidia
```

执行这一行时，动态链接器加载 `build/triton-src/python/triton/_C/libtriton.so`。前端、中端、后端调用的都是这个已经加载的模块。

### 2.6 调用图

```text
import triton                                          加载 .so
  _discover_backends()
    import nvidia/backend/compiler.py
      from triton._C.libtriton import ir, passes, llvm, nvidia
        加载 build/triton-src/python/triton/_C/libtriton.so

@triton.jit                                            构造 JITFunction
kernel[grid](...)
  JITFunction.run                                      triton/runtime/jit.py
    按实参类型和 constexpr 生成 cache key
    查 self.cache[device]                              进程内缓存
    命中
      kernel.run                                       CudaLauncher，CUDA driver API
    未命中
      make_backend → CUDABackend，parse_options        Python
      构造 signature、constants、ASTSource             Python，这一步还没有 IR
      triton.compiler.compile                          triton/compiler/compiler.py
        make_backend → CUDABackend                     Python
        查 TRITON_CACHE_DIR/<hash>/                    磁盘缓存，FileCacheManager
        命中
          读回 CompiledKernel，不执行 stages
        未命中
          CUDABackend.add_stages                       Python，登记五个 stage
          ir.context()、ir.load_dialects               .so  ir.cc
          CUDABackend.load_dialects                    .so  nvidia
          前端
            ASTSource.make_ir
              ast_to_ttir                              triton/compiler/code_generator.py
                builder.create_*                       .so  ir.cc
          中端
            make_ttir
              passes.common / passes.ttir 的 add_*     .so  passes.cc
              pm.run                                   .so  ir.cc
            make_ttgir
              passes.ttgpuir / nvidia.passes 的 add_*  .so  passes.cc、nvidia
              pm.run                                   .so  ir.cc
          后端
            make_llir
              nvidia.passes.ttgpuir.add_to_llvmir      .so  nvidia
              llvm.to_module、llvm.optimize_module     .so  llvm.cc
            make_ptx
              llvm.translate_to_asm                    .so  llvm.cc
            make_cubin
              ptxas                                    不经过 libtriton.so
          把各 stage 写入 TRITON_CACHE_DIR/<hash>/
      self.cache[device][key] = kernel                 写入进程内缓存
      kernel.run                                       CudaLauncher，CUDA driver API
```

`pm.run` 在 C++ 里跑完才回到 Python。`ptxas` 和 `CudaLauncher` 不进入 `libtriton.so`。

调用图里有两次查找，这就是两层缓存。先查进程内，进程内没有再进 `compile()` 查磁盘。

**进程内缓存**由 `JITFunction.run` 查。`JITFunction.__init__` 把 `self.cache` 设成 `defaultdict(dict)`：外层 key 是 device，内层 key 是实参类型和 constexpr 组成的签名。命中则直接 `kernel.run`，不调用 `compile()`。未命中时，`compile()` 返回之后，`run` 执行 `self.cache[device][key] = kernel`。这张表只在当前进程里。

**磁盘缓存**由 `compile()` 查，在第二次 `make_backend` 之后、`add_stages` 之前。`get_cache_manager` 返回 `FileCacheManager`，目录是 `TRITON_CACHE_DIR/<hash>/`。`hash` 是这一串的 sha256：`triton_key()`、`src.hash()`、`backend.hash()`、`options.hash()`、`get_cache_invalidating_env_vars()`。`triton_key()` 覆盖 `libtriton.so` 的字节，以及 `compiler/`、`backends/`、`language/` 下的 Python 文件。目录里已有 `<kernel 名>.json` 时，`compile()` 用这些文件构造 `CompiledKernel` 并返回，不执行 `stages`。否则跑完 stage，把 `ttir`、`ttgir`、`llir`、`ptx`、`cubin` 和这份 json 写进同一目录。`TRITON_ALWAYS_COMPILE=1` 时跳过这次查找。

Python 侧用到的接口就是图里标了 `.so` 的那些：

| 调用 | 子模块 | 作用 |
|---|---|---|
| `ir.context`、`ir.load_dialects`、`builder.create_*`、`ir.pass_manager`、`pm.run`、`module.str`、`ir.parse_mlir_module` | `ir` | 建 context、建 op、跑 pass、把模块换成文本或读回来 |
| `passes.ttir.add_*`、`passes.ttgpuir.add_*`、`passes.common.add_*` | `passes` | 往 pass manager 里追加一个 C++ pass |
| `nvidia.load_dialects`、`nvidia.passes.ttgpuir.add_to_llvmir` | `nvidia` | NVIDIA 方言，以及 TTGIR 降到 LLVM dialect |
| `llvm.to_module`、`llvm.optimize_module`、`llvm.translate_to_asm` | `llvm` | LLVM dialect 变成 LLVM IR，再变成 PTX 文本 |

`python -m triton_llm.tt_opt` 对一份 `.mlir` 调用同一张表里的 `passes.*.add_*` 和 `pm.run`。`build/bin/triton-opt` 不走 pybind：`bin/triton-opt.cpp` 经 `RegisterTritonDialects.h` 注册方言和 pass，然后 `MlirOptMain`，调用的是同一批 C++ 构造函数。

## 3. 从 Python 函数到 cubin

默认顺序在 `CUDABackend.add_stages`：`ttir` → `ttgir` → `llir` → `ptx` → `cubin`。这些 stage 里的 pass 是官方 v3.1.0 的。`tl.chip_rcp` 是补丁加的 op，走同一条 stage 顺序，见第 3.5 节。

### 3.1 `ASTSource.make_ir`：建 op

`compile()` 调用 `ASTSource.make_ir`，再调用 `ast_to_ttir`。`CodeGenerator.visit` 通过 `builder.create_*` 把 op 写进模块。这一步还没有 pass。`CompiledKernel.asm["ttir"]` 是后面 `CUDABackend.make_ttir` 跑完之后的文本。

[`lesson.py`](../python/triton_llm/compiler/lesson.py) 把常用 op 放在同一个 kernel 里。[`frontend.py`](../python/triton_llm/compiler/frontend.py) 的 `capture_frontend_ttir` 在 `make_ttir` 之前取出这个模块。`tl.dot` 写成 `tt.dot`，`for` 写成 `scf.for`，`tl.make_block_ptr` 写成 `tt.make_tensor_ptr`。`asm["ttir"]` 里已经没有 `tt.make_tensor_ptr`，因为 `make_ttir` 里的 `--triton-rewrite-tensor-pointer` 把它改成了普通指针上的 load / store。对照在 [`tests/python/test_frontend_middle.py`](../tests/python/test_frontend_middle.py)。

`tl.chip_rcp` 也在这一步建 op：`language/math.py` 调用 `_builder.create_chip_rcp`，`ir.cc` 按 `TT_ChipRcpOp` 写成 `tt.chip_rcp`。之后怎么降到 PTX 见第 3.5 节。`tt.fused_dot_mul` 不是 DSL 里的 `tl.*`，由 `--triton-fuse-dot-epilogue` 从 `tt.dot` 改写出来。

### 3.2 `CUDABackend.make_ttir`

`python -m triton_llm.tt_opt --make-ttir` 执行同一组 pass。跑完仍是 `tt` 方言，张量上没有 layout。

| pass | 作用 |
|---|---|
| `--inline` | 内联 `tt.call` |
| `--triton-rewrite-tensor-pointer` | `tt.make_tensor_ptr` 改成普通指针上的 load / store |
| `--triton-combine` | 折叠连续的 `addptr`，以及 dot 的累加 |
| `--canonicalize` | 各 op 自带的常量折叠 |
| `--triton-reorder-broadcast` | 把 `tt.splat` 挪到逐元素运算之后 |
| `--cse`、`--licm`、`--symbol-dce` | 公共子表达式、循环不变量、无用符号 |

`0002-ttir-passes-llvm19.patch` 另外把四条 TTIR 变换链进了同一个 `.so` 和 `build/bin/triton-opt`。带上对应 flag 时，`build/bin/triton-opt` 和 `python -m triton_llm.tt_opt` 都会跑这些变换。`CUDABackend.make_ttir` 的默认顺序里没有它们，所以 `@triton.jit` 不会跑到。

| flag | 变换 |
|---|---|
| `--triton-annotate-dot-stages` | 每个 `tt.dot` 写上 `triton_llm.num_stages`：`sm < 80` 为 2，否则 3 |
| `--triton-fuse-dot-epilogue` | 单次使用的 `tt.dot` 乘一个同类型标量，改写成 `tt.fused_dot_mul` |
| `--triton-lower-fused-dot-mul` | 展开回 `tt.dot` 和循环外的乘法 |
| `--triton-tile-dot` | K 能被 `BLOCK_K` 整除时，把整张量 `tt.dot` 改成 `scf.for` + `tensor.extract_slice` + 内层 `tt.dot`。输出仍是 TTIR |

```bash
./build/bin/triton-opt tests/tt/annotate_dot_stages.mlir --triton-annotate-dot-stages
python -m triton_llm.tt_opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue
python -m triton_llm.tt_opt tests/tt/annotate_dot_stages_sm90.mlir --triton-annotate-dot-stages=sm=90
```

`--triton-tile-dot` 的输出仍是 TTIR，不进入 `make_ttgir`。kernel 里进入 MMA 的切块是 DSL 自己的循环，见第 3.3 节。

要让一条新 pass 跟着 `@triton.jit` 跑，先按第 4 节把它链进 `.so`，再在 `CUDABackend.make_ttir` 或 `make_ttgir` 里调用它的 `add_*`。上面四条只由 flag 调用。

### 3.3 Tile，以及 `CUDABackend.make_ttgir`

切块写在 DSL 里，发生在 pass 之前。[`lesson.py`](../python/triton_llm/compiler/lesson.py) 的 `for k0 in range(0, K, BLOCK_K)` 和循环内的 `tl.dot`，由 `ASTSource.make_ir` 写成 `scf.for` 和一块 `tt.dot`。`BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`num_stages` 来自 [`arch/tiling.py`](../python/triton_llm/arch/tiling.py) 的 `choose_gemm_tile`。sm 70 上是 `BLOCK_M=128`、`BLOCK_N=128`、`BLOCK_K=64`、`num_stages=2`。C++ 里的同一张表是 `gemmTileForSm`，`tests/python/test_tile_table_sync.py` 核对两边一致。

`CUDABackend.make_ttir` 不改变这块 tile 的形状。`CUDABackend.make_ttgir` 给已经切好的 `tt.dot` 选 layout。V100（sm 70）上和这块 tile 有关的 pass：

| pass | 作用 |
|---|---|
| `--convert-triton-to-tritongpu` | 按块形状加上 `#blocked` |
| `--tritongpu-coalesce` | 合并访存 |
| `--tritongpu-accelerate-matmul` | 把 `tt.dot` 的 layout 换成 `#triton_gpu.nvidia_mma` |
| `--tritongpu-optimize-dot-operands` | 调整 dot 两个操作数的 layout。sm 70 不使用 sm ≥ 80 那条操作数路径 |

`make_ttgir` 在 sm ≥ 80 时加入 `tritongpu-f32-dot-tc` 和 `tritongpu-pipeline`，在 sm ≥ 90 时再加入 fence 与 TMA。同一函数里还有 `tritongpu-plan-cta`、`tritongpu-remove-layout-conversions`、`tritongpu-optimize-thread-locality`、`tritongpu-prefetch`、`tritongpu-reduce-data-duplication`、`tritongpu-reorder-instructions`，以及 `cse`、`symbol-dce`、`canonicalize`。sm 70 不跑 `tritongpu-pipeline`，所以 `num_stages` 不参与这次 lowering。`choose_gemm_tile` 仍返回 `num_stages`，它作为编译选项传给 `CUDABackend`，sm ≥ 80 时 `add_pipeline` 才读取。

sm 70 上 `tritongpu-accelerate-matmul` 把 `tt.dot` 标成 `#triton_gpu.nvidia_mma<{versionMajor = 1, ...}>`。`AccelerateMatmul.cpp` 的 `getMMAVersionSafe` 按 sm 取值：低于 75 为 1（Volta 的 `mma.sync`），75 到 89 为 2，90 及以上为 3（`wgmma`）。第 3.4 节把 version 1 写成 `mma.sync.aligned.m8n8k4`。`test_own_libtriton_tiles_dot_through_ttgir_to_llvm` 用 `choose_gemm_tile(70)` 检查 `versionMajor = 1` 和这条 PTX。

同一阶段里，`TritonToTritonGPUPass.cpp` 的 `GenericOpPattern<ChipRcpOp>` 给 `tt.chip_rcp` 的结果张量加上 layout，op 名字不变。第 3.4 节读的是这份带 layout 的 TTGIR。

### 3.4 `CUDABackend.make_llir`、`make_ptx`、`make_cubin`

这三个方法也在 `nvidia/backend/compiler.py`。

- `make_llir`：先跑 `scf-to-cf`、`allocate-shared-memory`，再 `nvidia.passes.ttgpuir.add_to_llvmir`（`TritonGPUToLLVM.cpp`）、`nvgpu-to-llvm`、`arith-to-llvm`。然后 `llvm.to_module` 和 `llvm.optimize_module(..., OPTIMIZE_O3)`。Volta 的 f16 `tt.dot` 在 `DotOpToLLVM/MMAv1.cpp` 里写成 `mma.sync.aligned.m8n8k4`。
- `make_ptx`：`llvm.translate_to_asm`，triple `nvptx64-nvidia-cuda`。
- `make_cubin`：Python 用子进程调用 `triton/backends/nvidia/bin/ptxas`。这一步不进入 `libtriton.so`。

### 3.5 `tl.chip_rcp`：从 DSL 降到 `rcp.approx.ftz.f32`

`0001-frontend-llvm19.patch` 增加 `tl.chip_rcp`，只接受 fp32。DSL 里的名字是 `chip_rcp`，PTX opcode 是 `rcp`。它走过第 3.1 节到第 3.4 节，中间没有单独的 flag：

| 阶段 | `.so` 里发生的事 | 这一步之后 |
|---|---|---|
| `ASTSource.make_ir` | `ir.cc` 的 `create_chip_rcp` 建 `TT_ChipRcpOp` | `tt.chip_rcp`，张量上没有 layout |
| `CUDABackend.make_ttir` | 第 3.2 节默认的那组 pass 不改这个 op | 仍是 `tt.chip_rcp` |
| `CUDABackend.make_ttgir` | `GenericOpPattern<ChipRcpOp>` | 张量带上 layout，op 名字不变 |
| `CUDABackend.make_llir` | `ElementwiseOpToLLVM.cpp` 的 `ChipRcpOpConversion`，由 `add_to_llvmir` 执行 | f32 元素写成 `rcp.approx.ftz.f32` |
| `make_ptx`、`make_cubin` | `llvm.translate_to_asm`，然后 `ptxas` | PTX 里是同一条指令，cubin 非空 |

CodeGen 写的是：

```cpp
ptxBuilder.create<PTXInstr>("rcp")->o("approx").o("ftz").o("f32");
```

`PTXInstr` 的第一个参数是 opcode，后面每个 `.o(...)` 是一个点号修饰符，拼出来是 `rcp.approx.ftz.f32`。`rcp` 是 PTX 的倒数，`.approx` 是近似，`.ftz` 把次正规数刷成 0，`.f32` 和这个函数只接受 fp32 一致。f32 的 `rcp.approx` 带 `.ftz`，`ptxas` 才接受这条指令。

`ChipRcpOpConversion` 写在 `ElementwiseOpToLLVM.cpp`，由 `add_to_llvmir` 调用，接法和 `tl.exp` 相同。opcode 用 PTX 已有的 `rcp`。

`test_chip_rcp_lowers_through_ttgir_to_cubin` 检查：`ttir` 和 `ttgir` 里有 `tt.chip_rcp`，`llir` 和 `ptx` 里有 `rcp.approx.ftz.f32`，cubin 非空。

## 4. `triton-opt`

1.2 节那次 pip 安装里的 cmake，把同一份 pass 的 `.cpp` 链进 `libtriton.so` 和 `build/bin/triton-opt`。官方 wheel 不带 `triton-opt`。仓库根目录的 cmake 不参与。

一条新 pass 要同时进这两个产物，改的是 `build/triton-src` 里的这些文件：

1. 在 `include/triton/Dialect/Triton/Transforms/Passes.td` 声明，在 `lib/Dialect/Triton/Transforms/` 实现。类放在 `namespace mlir::triton`，构造函数写成 `mlir::triton::create*Pass()`，pattern 用 `applyPatternsAndFoldGreedily`。
2. 把 `.cpp` 加进该目录的 `CMakeLists.txt`，并在 `python/src/passes.cc` 增加 `add_*`。
3. `bin/RegisterTritonDialects.h` 已经注册 Triton 的 pass，flag 随 `Passes.td` 出现在 `triton-opt --help` 里。
4. 把改动写成补丁，再跑 `scripts/build_libtriton.sh`。`@triton.jit` 要跑到它，还得在 `make_ttir` 或 `make_ttgir` 里调用这个 `add_*`，见第 3.2 节。

`tests/tt/` 里的 `// RUN:` 指定用哪个程序跑这条测试。`tests/shell/run_tt_opt_tests.sh` 把其中的 `tt-opt` 换成 `python -m triton_llm.tt_opt`，把 `TT_OPT_CPP` 换成 `build/bin/triton-opt`。

## 5. 命令

先安装两个包（第 1 节），再跑 IR 和 pytest。`ctest` 使用仓库根目录的 CMake：它再跑一遍 shell 测试和 pytest，并加上 CUTLASS。`libtriton.so` 不由这次 cmake 编译。

```bash
git submodule update --init third_party/triton
scripts/setup_venv.sh
source .venv/bin/activate

./build/bin/triton-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue
python -m triton_llm.tt_opt tests/tt/annotate_dot_stages.mlir --triton-annotate-dot-stages
./tests/shell/run_tt_opt_tests.sh

pytest tests/python/test_frontend_middle.py
pytest tests/python/test_own_libtriton.py
pytest tests/python/test_python_cpp_handoff.py
pytest tests/python/test_so_handwritten_passes.py
pytest tests/python/test_tile_table_sync.py

cmake -B build -DTRITON_LLM_CUDA_ARCH=70
cmake --build build -j"$(nproc)"
ctest --test-dir build
```

看 `@triton.jit` 每个 stage 的 IR：

```bash
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python scripts/dump_triton_ir.py
```

改了 `build/triton-src` 里的 C++（`create_chip_rcp`、`ChipRcpOpConversion`、`gemmTileForSm`、四个 TTIR pass）之后，重新生成对应补丁，再编一次 `.so`，然后从 `triton-opt` 往下复测：

```bash
scripts/build_libtriton.sh
./build/bin/triton-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue
./tests/shell/run_tt_opt_tests.sh
pytest tests/python/test_own_libtriton.py
```

块大小写在两处。kernel 用 `choose_gemm_tile`，`--triton-tile-dot` 用 `gemmTileForSm`。两处一起改时，编完再跑 `pytest tests/python/test_tile_table_sync.py`。

`build_libtriton.sh` 编译的是 `build/triton-src`，不是 `third_party/triton` 的工作区。工作区上的修改要先写成补丁，打进 `build/triton-src`，再执行上面的命令。

## 6. 主要入口函数

下表里 Triton 的路径相对于 `third_party/triton/`。补丁改过的文件以 `build/triton-src/` 里的同名路径为准。本仓库的路径相对于仓库根目录。

| 作用 | 函数 | 文件 |
|---|---|---|
| `@triton.jit` 何时编译、何时启动 | `JITFunction.run` | `python/triton/runtime/jit.py` |
| 选中哪一个 backend、按什么顺序跑 stage | `compile`、`make_backend` | `python/triton/compiler/compiler.py` |
| DSL 函数生成 TTIR 模块 | `ASTSource.make_ir`、`ast_to_ttir` | `python/triton/compiler/compiler.py`、`python/triton/compiler/code_generator.py` |
| 默认 pass 顺序（TTIR → TTGIR → LLVM → PTX → cubin） | `CUDABackend.add_stages`、`make_ttir`、`make_ttgir`、`make_llir`、`make_ptx`、`make_cubin` | `third_party/nvidia/backend/compiler.py` |
| 增加一个 `tl.*`，并在 `.so` 里建 op | `chip_rcp`、`create_chip_rcp` | `python/triton/language/math.py`、`python/src/ir.cc`（`0001`） |
| 这个 op 在 TTGIR 上拿到 layout | `GenericOpPattern<ChipRcpOp>` | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`（`0001`） |
| 这个 op 写成 PTX 指令 | `ChipRcpOpConversion` | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`（`0001`） |
| 一块 `tt.dot` 写成 Volta MMA | `MMAv1` 的 lowering | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv1.cpp` |
| kernel 里的切块从哪来 | `_frontend_lesson` 的 `for` / `tl.dot`；`choose_gemm_tile` | `python/triton_llm/compiler/lesson.py`、`python/triton_llm/arch/tiling.py` |
| 命令行上的四条 TTIR 变换 | `createAnnotateDotStagesPass`、`gemmTileForSm` | `lib/Dialect/Triton/Transforms/AnnotateDotStages.cpp`、`FuseAndTileDot.cpp`（`0002`） |
| 这四条变换怎么从 Python 调进 `.so` | `add_annotate_dot_stages` 等 | `python/src/passes.cc`（`0002`） |
