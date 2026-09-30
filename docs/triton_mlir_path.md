# Triton 编译器是怎么跑起来的

TTIR 是 `tt` 方言，TTGIR 是 `triton_gpu` 方言。本仓库走 Triton 自己的 MLIR，不经过 StableHLO、XLA 或 Linalg。

## 1. `@triton.jit` 编译 kernel 时调用 `libtriton.so`

用户在 Python 里调用一个 `@triton.jit` 函数。第一次调用时，当前这个 Python 进程加载 `.venv` 里的 `triton/_C/libtriton.so`，把函数编译成 cubin，再通过 CUDA driver API 启动。执行这次编译的是这个 `.so`，不是 `triton-opt` 这个可执行文件。

调用链：

```text
kernel[grid](...)
  JITFunction.run                         runtime/jit.py
    内存里已有这份编译结果 → 直接启动
    否则 JITFunction.compile
      triton.compiler.compile             compiler/compiler.py
        from triton._C.libtriton import ir
        backend.add_stages                nvidia/backend/compiler.py
        ir.context() / ir.load_dialects   → libtriton.so
        src.make_ir                       → ast_to_ttir → builder.create_*
        按 stage 调用 make_ttir、make_ttgir、make_llir、make_ptx、make_cubin
          每个 stage：pass_manager + passes.*.add_* + pm.run
        写出 ~/.triton/cache
      kernel.run                          launcher 调 CUDA driver API。launcher 是 gcc 或 clang 编译出来的 C 扩展
```

`import triton` 加载的文件是安装目录里的 `triton/_C/libtriton.so`。它是 pybind11 模块，Python 里的名字是 `triton._C.libtriton`。`python/src/main.cc` 的 `PYBIND11_MODULE(libtriton, m)` 注册五个子模块：

| 子模块 | 源码 | Python 用它做什么 |
|---|---|---|
| `ir` | `python/src/ir.cc` | 建 `context`、`builder`、`module`、`pass_manager`；`parse_mlir_module` 读文本 |
| `passes` | `python/src/passes.cc` | `add_combine` 这类函数，往 pass manager 里追加一个 C++ pass |
| `llvm` | `python/src/llvm.cc` | 把 MLIR 的 LLVM dialect 转成 LLVM IR，再转成 PTX 文本 |
| `nvidia` | `third_party/nvidia/triton_nvidia.cc` | 注册 NVIDIA 方言，以及 TTGIR 上的 NVIDIA pass |
| `interpreter` | `python/src/interpreter.cc` | 只在 `TRITON_INTERPRET=1` 时处理 host 上的原子操作，不跑 MLIR pass |

Python 和这份 `.so` 之间只有三种调用：

1. **建 op。** `tl.dot` 写在 `language/core.py`，检查在 `semantic.py`，最后是 `builder.create_dot(...)`。`create_dot` 是 `ir.cc` 里绑出来的 C++ 函数，返回的 `Value` 由 Python 拿着。
2. **跑 pass。** `passes.ttir.add_combine(pm)` 只做一件事：`pm.addPass(createCombineOpsPass())`。接着 `pm.run(module)` 在 C++ 里跑完才返回。跑的过程中不再回到 Python。
3. **交换文本。** `module.str()` 生成 IR 文本，`ir.parse_mlir_module` 再读回去。缓存里每个 stage 的文件就是这些文本。

`make_ttir` 这些函数本身是 Python（`nvidia/backend/compiler.py`）。它们决定 pass 的先后，真正改 IR 的是 `.so` 里的 C++。

磁盘缓存的 key 含有 `libtriton.so` 的全部字节，以及已安装的 `compiler/`、`backends/`、`language/` 里的 Python 源码。换掉这个 `.so`，`~/.triton/cache` 里已有的编译结果全部失效。这个目录可以用 `TRITON_CACHE_DIR` 改到别处。`MLIR_ENABLE_DUMP=1` 会在每个 pass 前后打印 IR。同时还要设 `TRITON_ALWAYS_COMPILE=1`，否则命中缓存之后，这些 pass 不会再执行。

`TRITON_INTERPRET=1` 时，`jit()` 返回 `InterpretedFunction`，不会进入上面的 `compile()`。解释器按调用时给出的 grid，在 Python 里用 numpy 做计算；原子操作调用 `libtriton.interpreter`。本仓库没有测这条路径。

## 2. `libtriton.so` 和 `triton-opt` 各做什么

上游 Triton 的一次构建同时产出这两个文件。它们链接同一批 pass 的 `.cpp`，入口不同。本仓库没有做这次完整构建，见第 2.2 节。

`@triton.jit` 编译并启动 kernel 时，Python 只调用 `libtriton.so`。`triton-opt` 是命令行程序：读一份 `.mlir`，执行 pass，把改过的 IR 写到标准输出。上游的 lit 用它做 FileCheck。它不生成 cubin，`@triton.jit` 也不会启动它。

| | `libtriton.so` | `triton-opt` |
|---|---|---|
| 谁调用 | 调用了 `@triton.jit` 函数的那个 Python 进程 | 开发者在终端里执行，或 lit 测试脚本执行 |
| 输入 | Python 函数 | 一份 `.mlir` 文件 |
| 输出 | cubin，随后由 launcher 启动 kernel | 改过的 IR 文本，写到标准输出 |
| 入口 | `python/src/main.cc` | `bin/triton-opt.cpp`，大约十行：注册方言，然后 `MlirOptMain` |

`bin/triton-opt.cpp` 通过同目录的 `RegisterTritonDialects.h` 完成注册：Triton、TritonGPU、NVIDIA 的 pass，再加上 TTIR→TTGIR、TTGIR→LLVM。注册的方言包括 `tt`、`triton_gpu`、`triton_nvidia_gpu`、`nvgpu`、`arith`、`math`、`scf`、`cf`、`gpu`、`llvm`、`nvvm`、`rocdl`。同目录还有 `triton-llvm-opt`、`triton-lsp`、`triton-reduce`。这三个也是命令行程序，`@triton.jit` 不调用它们。

`test/lit.cfg.py` 用这些二进制跑 `.mlir` / `.ll` 的 FileCheck。pip 装上的 wheel 不带 `triton-opt`，本机 `.venv/bin` 里也没有。

### 2.1 工业界：同一次构建产出这两个文件

工业界不是在「重编 `libtriton.so`」和「编译 `triton-opt` 专门测 pass」里选一条。一次 `cmake` 构建同时产出这两个文件。两个文件链接同一份 pass 的 `.cpp`，也链接同一份 LLVM。

| 文件 | 谁用它 | 少了它会怎样 |
|---|---|---|
| `libtriton.so` | 用户调用 `@triton.jit` 的那个 Python 进程 | 用户启动的 kernel 里没有这个 pass。FileCheck 通过之后也仍然没有 |
| `triton-opt` | 写 pass 的人，以及 lit。输入是一份 `.mlir`，输出是改过的 IR | 没有单独的 `.mlir` 测试。要看 pass 的效果，只能把 kernel 启动起来。它不生成 cubin |

加一个 pass 时，上游仓库里改下面这些地方，然后重新构建，两个文件一起更新：

1. `Passes.td` 和 pass 的 `.cpp`。
2. 在 `bin/RegisterTritonDialects.h` 里注册，让 `triton-opt` 能通过该 pass 的 flag 运行它。`test/` 里加 lit，对 `.mlir` 做 FileCheck。
3. `python/src/passes.cc` 增加 `add_*`。
4. `make_ttir` 或 `make_ttgir` 调用这个 `add_*`。
5. 把新的 `libtriton.so` 装进 Python 环境。此后 `@triton.jit` 编译 kernel 时会执行这个 pass。

第 2 步只让命令行和 lit 能跑这个 pass。第 3 到第 5 步才让 `@triton.jit` 执行它。

芯片厂商如果沿用 Triton 的 Python 前端，`@triton.jit` 的写法可以保留。要替换的是 backend：在 `BaseBackend.add_stages` 里换成自己的 pass 顺序，用 `DriverBase` 在自己的设备上启动 kernel。NVIDIA 在 `TritonToTritonGPUPass.cpp` 和 `TritonGPUToLLVM.cpp` 的前面和后面插入 pass。换成另一种 GPU 时，才需要自己的 GPU dialect，以及降到自己指令集的 lowering。

### 2.2 本仓库停在测试这一半

`build/bin/tt-opt` 对应上表里的 `triton-opt`，但只包含四个 TTIR pass。这四个 pass 的 C++、`Passes.td`、对 `.mlir` 的 FileCheck，以及固定的 tile 表，写法和上游一致。用户调用 `@triton.jit` 时，这四个 pass 不会执行。

| 工业界一次 Triton 构建 | 本仓库现在 |
|---|---|
| `triton-opt` 注册全部方言和 pass，能从 TTIR 走到 LLVM | `build/bin/tt-opt` 只注册四个 TTIR pass。不能读 `triton_gpu`，不生成 PTX |
| lit 调用上游的 `bin/triton-opt` | `tests/shell/run_tt_opt_tests.sh` 调用 `build/bin/tt-opt`。本仓库不编译 `bin/triton-opt.cpp`，也不运行 lit |
| 同一份 `.cpp` 链进 `libtriton.so` | 四个 pass 只在 `build/bin/tt-opt` 里。`.venv` 里的 `libtriton.so` 是 pip 的 Triton 3.1.0，不含它们 |
| `passes.cc` 的 `add_*` 被 `make_ttir` / `make_ttgir` 调用，并且这份代码在已安装的 `.so` 里 | 补丁里的 `passes.cc` 只有 `add_annotate_dot_stages`。融合和切分没有 `add_*`。`make_ttir` 不调用它们。已安装的 `.so` 没有用这份补丁重编 |
| `libtriton.so` 和 `triton-opt` 链接同一份 LLVM | `.venv` 的 `.so` 用 Triton 3.1.0 在 `cmake/llvm-hash.txt` 里固定的 LLVM。`tt-opt` 用 `/opt/torch-mlir` 的 LLVM 23。两边只能传递 IR 文本 |

因此，`@triton.jit` 编译出的 cubin 里没有这四个 pass 改过的 IR。要让 `@triton.jit` 执行它们：把 pass 的 `.cpp` 编进 Triton，在 `passes.cc` 里为四个 pass 都加上 `add_*`，在 `make_ttir` 里调用，再用这次构建产出的 `libtriton.so` 替换 `.venv` 里的那一份。`build/bin/tt-opt` 仍然可以对 `.mlir` 做 FileCheck。替换 `.so` 这一步不能省。

## 3. 本仓库里的三个程序

| 程序 | 是什么 | 会不会把 `@triton.jit` 函数编成 cubin |
|---|---|---|
| `.venv` 里的 Triton 3.1.0 | 安装包：Python 文件、`libtriton.so`、`ptxas` | 会。`@triton.jit` 编译 kernel 时用的就是这一份 |
| `python -m triton_llm.tt_opt` | 本仓库的 Python 脚本。它调用上一行那个 `.so` 里已经有的 pass | 不会。输入是一份 `.mlir`，用来执行 `.so` 里已经有的 pass |
| `build/bin/tt-opt` | 本仓库编译出的可执行文件，只包含手写的 TTIR pass | 不会。不能解析 TTGIR，也不生成 PTX |

Triton 源码是 submodule `third_party/triton`（v3.1.0，`cf34004`），不进本仓库的提交。本地改动在 `third_party/patches/`。CMake 把 submodule 复制到 `build/triton-patched`，打上补丁，再编 `tt-opt`。哪些目录参与这次编译、哪些只供阅读，见 [`third_party/patches/README.md`](../third_party/patches/README.md)。

这三个程序不会互相调用。`build/bin/tt-opt` 链接 LLVM 23。`.venv` 里的 `libtriton.so` 链接的是 Triton 3.1.0 在 `cmake/llvm-hash.txt` 里固定的那一版 LLVM。这两个二进制不能传递 C++ 对象，只能传递 IR 文本。`tt.fused_dot_mul` 是补丁增加的 op，已安装的 `libtriton.so` 解析不了它。要把 `build/bin/tt-opt` 的输出再交给 `python -m triton_llm.tt_opt`，先加上 `--triton-lower-fused-dot-mul`，把这个 op 展开回 `tt.dot`。

## 4. 从 Python 函数到 cubin

### 4.1 第一份 TTIR

`compile()` 里 `src.make_ir` 调用 `ast_to_ttir`（`python/triton/compiler/code_generator.py`）。`CodeGenerator.visit` 遍历 AST，通过 `builder.create_*` 写下 `tt.func`、`tt.load`、`tt.dot`、`scf.for`。这时还没有跑 `make_ttir`。

[`lesson.py`](../python/triton_llm/compiler/lesson.py) 把下表这些写法放在同一个 kernel 里。[`frontend.py`](../python/triton_llm/compiler/frontend.py) 的 `capture_frontend_ttir(jit_fn, launch)` 在 `make_ttir` 之前取出模块。`CompiledKernel.asm["ttir"]` 是 `make_ttir` 跑完之后的文本。其中的 `triton-rewrite-tensor-pointer` 已经把 `tt.make_tensor_ptr` 改掉，所以这份文本里看不到它。

| Python | 第一份 TTIR |
|---|---|
| `tl.program_id` | `tt.get_program_id` |
| `tl.arange` | `tt.make_range` |
| 指针加偏移 | `tt.addptr` |
| `tl.load` / `tl.store` | `tt.load` / `tt.store` |
| `for` | `scf.for`。归纳变量的占位是 `llvm.mlir.undef : i32` |
| `if` | `scf.if` |
| `tl.dot` | `tt.dot` |
| `tl.sum` | `tt.reduce` |
| 标量铺到 tile | `tt.splat` |
| `tl.make_block_ptr` | `tt.make_tensor_ptr` |
| `tl.constexpr` 的 `BLOCK_*` | 张量类型的形状，例如 `tensor<16x16xf16>`，不出现在 `tt.func` 的参数列表里 |
| 16 字节对齐的指针 | 参数属性 `tt.divisibility = 16` |
| `do_not_specialize=["M","N","K"]` | `M`、`N`、`K` 留在 `tt.func` 参数里 |

`for` 的 IR 里有 `llvm.mlir.undef`，所以读取这份 IR 的工具必须能解析 `llvm` 方言。`build/bin/tt-opt` 因此注册了 `LLVMDialect`。

[`tests/python/test_frontend_middle.py`](../tests/python/test_frontend_middle.py) 检查这张表。

要增加一个用户可以调用的 `tl.xxx`：在 `TritonOps.td` 加 op，在 `ir.cc` 加 `create_xxx`，在 `language/core.py` 和 `semantic.py` 加 `@builtin`，然后重新编译 Triton 并安装新的 `libtriton.so`。用已有 op 组成新函数时，写一个 `@triton.jit` 辅助函数即可，不用重新编译。`tt.fused_dot_mul` 由 pass 生成，前端没有对应的 `tl.*`。

### 4.2 `make_ttir`

顺序写在 submodule 的 `third_party/nvidia/backend/compiler.py`，函数是 `CUDABackend.make_ttir`。`python -m triton_llm.tt_opt --make-ttir` 跑的是同一串 pass，里面没有本仓库的手写 pass。跑完之后仍是 `tt` 方言，张量上还没有 layout。

| pass | 作用 |
|---|---|
| `inline` | 内联 `tt.call` |
| `triton-rewrite-tensor-pointer` | `tt.make_tensor_ptr` 改成普通指针上的 load / store |
| `triton-combine` | 折叠连续的 `addptr`、dot 的累加等 |
| `canonicalize` | 各 op 自带的常量折叠 |
| `triton-reorder-broadcast` | 把 `tt.splat` 挪到逐元素运算之后 |
| `cse`、`licm`、`symbol-dce` | 公共子表达式、循环不变量、无用符号 |

```bash
source .venv/bin/activate
export PYTHONPATH=python
python -m triton_llm.tt_opt tests/tt/combine_addptr.mlir --triton-combine
python -m triton_llm.tt_opt --list --sm 70
```

每个命令行 flag 对应哪个 pass，写在 [`pipeline.py`](../python/triton_llm/compiler/pipeline.py)。

### 4.3 `make_ttgir`

同一个文件里的 `make_ttgir`。`convert-triton-to-tritongpu` 把 `tt` 降到 `triton_gpu`，并给值加上 `#blocked` layout。后面的 pass 做三件事：合并访存、去掉多余的 `convert_layout`、给 `tt.dot` 选择 MMA layout。

`sm // 10 >= 8` 时才会加入 software pipeline，以及 f32 dot 走 tensor core 的 pass。`sm // 10 >= 9` 时才会加入 fence 和 TMA。V100 的 sm 是 70，这四项都不会放进它的 `make_ttgir`。在 V100 上，`num_stages` 只作为 kernel 的启动参数传下去，Triton 的软件流水线 pass 不会执行。对应源码在 `lib/Dialect/TritonGPU/Transforms/Pipeliner/` 和 `lib/Dialect/TritonNvidiaGPU/Transforms/`。本仓库只阅读这两个目录，不编译它们。

lesson kernel 在 sm 70 上经过 `--make-ttir --convert-triton-to-tritongpu --tritongpu-accelerate-matmul` 之后，dot 的 layout 是：

```text
#triton_gpu.nvidia_mma<{versionMajor = 1, versionMinor = 11, warpsPerCTA = [1, 1], instrShape = [16, 16]}>
#triton_gpu.dot_op<{opIdx = 0, parent = #mma}>
triton_gpu.convert_layout
```

`versionMajor = 1` 是 Volta 的 `mma.sync`。sm 80 是 2，sm 90 的 `wgmma` 是 3。`test_frontend_middle.py` 检查了 `versionMajor = 1` 和 `dot_op` 的 parent。

自己写的 `tt.func` 上还没有 layout 属性，不能作为 `--convert-triton-to-tritongpu` 的输入。要跑这个 pass，输入用 `scripts/dump_triton_ir.py` 写出的 `.ttir`。那份文件已经过 `make_ttir`。

### 4.4 LLVM、PTX、cubin

仍是 `compiler.py` 里的三个函数：

- `make_llir`：`allocate-shared-memory`、`TritonGPUToLLVM`（`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`）、`nvgpu-to-llvm` 等，再 `llvm.to_module` 和 O3。Volta 的 `mma.sync` 在 `DotOpToLLVM/MMAv1.cpp`，Hopper 的 `wgmma` 在 `WGMMA.cpp`。
- `make_ptx`：`llvm.translate_to_asm`，triple `nvptx64-nvidia-cuda`。这是 LLVM 的 NVPTX 后端。
- `make_cubin`：调用 wheel 里的 `ptxas`。没有对应的 Triton pass。

本仓库不修改这三个函数。

## 5. 本仓库的手写 pass：`build/bin/tt-opt`

入口是 [`compiler/tt-opt.cpp`](../compiler/tt-opt.cpp)。它和补丁里的 `AnnotateDotStages.cpp`、`FuseAndTileDot.cpp` 编成 `build/bin/tt-opt`，链接静态库 `TritonIR` 和 LLVM 23。不加载 `libtriton.so`。

```bash
cmake -B build -DTRITON_LLM_CUDA_ARCH=70
cmake --build build --target tt-opt -j"$(nproc)"
./build/bin/tt-opt --help
```

`build/bin/tt-opt` 能解析的方言有 `tt`、`arith`、`func`、`math`、`scf`、`cf`、`tensor`、`llvm`。没有 `triton_gpu`，所以不能读 TTGIR。`--help` 里列出的 pass 就是这里注册的：`canonicalize` 和下面四个。

| flag | 作用 |
|---|---|
| `--triton-annotate-dot-stages` | 每个 `tt.dot` 写上 `triton_llm.num_stages`：`sm < 80` 为 2，否则 3 |
| `--triton-fuse-dot-epilogue` | 单次使用的 `tt.dot` 乘一个同类型标量，改写成 `tt.fused_dot_mul` |
| `--triton-lower-fused-dot-mul` | 展开回 `tt.dot` 和循环外的乘法 |
| `--triton-tile-dot` | K 能被 `BLOCK_K` 整除时，改成 `scf.for` + `tensor.extract_slice` + 内层 `tt.dot` |

`tt.fused_dot_mul` 加在 `tt` 方言上，不是新方言。操作数 `c`、`scale` 和结果的类型必须相同。

tile 的候选写在 `FuseAndTileDot.cpp` 的 `gemmTileForSm`，和 [`arch/tiling.py`](../python/triton_llm/arch/tiling.py) 是同一组。每个四元组依次是 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`num_stages`：`(64,64,32,2)`、`(64,128,32,2)`、`(128,128,32,2)`、`(128,128,64,2)`、`(128,128,64,3)`。从前往后看，最后一个放得进 shared memory、并且 stage 数不超过该 sm 上限的候选被选中。每个 sm 只对应其中一个候选，运行时不再搜索。sm 70 和 sm 90 选中的都是 `BLOCK_K=64`，stage 分别是 2 和 3。这两个属性写在 `scf.for` 结尾的 `}` 上。`tests/python/test_tile_table_sync.py` 对 sm 70、75、80、86、90 比较这两份表。

这些 flag 来自补丁改过的 `Passes.td`。补丁打完之后，文件在 `build/triton-patched/include/triton/Dialect/Triton/Transforms/Passes.td`。命令行写成 `--triton-annotate-dot-stages=sm=90` 时，MLIR 先用默认的 sm 构造 pass，再把 `sm=90` 写进这个 pass 的选项。

补丁里的 `passes.cc` 写了 `add_annotate_dot_stages`，已安装的 `libtriton.so` 里没有这个函数。融合和切分没有 `add_*`。`@triton.jit` 编译 kernel 时不会执行这四个 pass。和工业界差在哪、要补哪几步，见第 2.2 节。submodule 里的 `compiler.py` 保持和已安装的 Triton 相同的 `make_ttir` 顺序，这样 `python -m triton_llm.tt_opt --make-ttir` 和 `@triton.jit` 用的是同一串 pass。

### 5.1 再加一个 pass

1. `Passes.td` 里加 pass 名和构造函数。
2. 新的 `.cpp` 里写 pass 类和 `create*Pass()`。可以照 `AnnotateDotStages.cpp`：`GEN_PASS_DECL`、`GEN_PASS_DEF`，然后 `runOnOperation`。
3. `compiler/tt-opt.cpp` 里再 `registerPass` 一次。新方言也要放进同一个 `DialectRegistry`。
4. 把 `.cpp` 加进 `compiler/CMakeLists.txt` 的 `add_executable(tt-opt ...)`。
5. 在 `tests/tt/` 里放 `.mlir`。`// RUN:` 行写 `TT_OPT_CPP`，不要写单词 `tt-opt`。`tests/shell/run_tt_opt_tests.sh` 先把单词 `tt-opt` 换成 `python -m triton_llm.tt_opt`，再把 `TT_OPT_CPP` 换成 `build/bin/tt-opt`。写成 `tt-opt` 时，脚本会去调用 `python -m triton_llm.tt_opt`，那个进程里的 `libtriton.so` 没有这个新 pass。

```bash
cmake --build build --target tt-opt -j"$(nproc)"
./tests/shell/run_tt_opt_tests.sh
```

Python 取出的 TTIR 可以直接交给这个二进制：

```bash
PYTHONPATH=python python - <<'EOF'
from pathlib import Path
from triton_llm.compiler.frontend import capture_frontend_ttir
from triton_llm.compiler.lesson import _frontend_lesson, launch_frontend_lesson
Path("/tmp/lesson.mlir").write_text(capture_frontend_ttir(_frontend_lesson, launch_frontend_lesson))
EOF
./build/bin/tt-opt /tmp/lesson.mlir --triton-annotate-dot-stages
```

`tests/python/test_python_cpp_handoff.py` 检查 `tt.dot` 上写出了 `triton_llm.num_stages`。

## 6. 改哪个文件，哪个程序会重新编译

| 改动 | `@triton.jit` | `python -m triton_llm.tt_opt` | `build/bin/tt-opt` |
|---|---|---|---|
| `python/triton_llm/ops/*.py` | 会。下次调用该 kernel 时重新编译 | 不会 | 不会 |
| `arch/tiling.py` | 会 | 不会 | 不会。要和 `FuseAndTileDot.cpp` 一起改。`test_tile_table_sync.py` 会检查两份表是否一致 |
| 补丁里的 pass、`Passes.td`、`TritonOps.td`、`compiler/tt-opt.cpp` | 不会 | 不会 | 会。`cmake --build build --target tt-opt` |
| submodule 工作区里的其余 `.cpp`，以及其中的 `compiler.py`、`code_generator.py` | 不会 | 不会 | 不会。这些文件没有被任何进程加载。`@triton.jit` 读的是 `.venv` 里安装的同名文件 |
| 重编 Triton 并换掉 `libtriton.so` | 会，缓存全部失效 | 会 | 不会 |

## 7. 调试、测试，以及本仓库没有做的

```bash
./build/bin/tt-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue --mlir-print-ir-after-all
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 PYTHONPATH=python python scripts/dump_triton_ir.py
```

| 命令 | 内容 |
|---|---|
| `./tests/shell/run_tt_opt_tests.sh` | `tests/tt/*.mlir` |
| `pytest tests/python/test_frontend_middle.py` | 前端 op、rewrite-tensor-pointer、Volta MMA layout |
| `pytest tests/python/test_python_cpp_handoff.py` | 前端 TTIR 送给 `build/bin/tt-opt` |
| `pytest tests/python/test_tile_table_sync.py` | 两份 tile 表一致 |
| `ctest --test-dir build` | 上面这些，加上 CUTLASS 和数值测试 |

| 现象 | 原因 |
|---|---|
| `Dialect 'triton_gpu' not found`（`build/bin/tt-opt`） | 这个二进制没有注册 `triton_gpu`。TTGIR 交给 `python -m triton_llm.tt_opt` |
| `unknown pass --triton-tile-dot`（`python -m triton_llm.tt_opt`） | 手写 pass 只在 `build/bin/tt-opt`。`// RUN:` 要写 `TT_OPT_CPP` |
| `AttributeError: add_annotate_dot_stages` | 已安装的 `libtriton.so` 没有这个函数 |
| 改了 submodule 里的 `code_generator.py`，kernel 的编译结果没变 | `@triton.jit` 加载的是 `.venv` 里的 Python 文件和 `libtriton.so` |
| 期望中的 pass 没有执行 | `~/.triton/cache` 里已有编译结果。设 `TRITON_ALWAYS_COMPILE=1` 后会重新编译 |

本仓库没有单独实现、也没有测试的：

- 自己写的 DialectConversion。上游的 TTIR→TTGIR 在 `TritonToTritonGPUPass.cpp`，由已安装的 `.so` 执行。
- software pipeline、fence、TMA。V100 上不会跑。
- 自定义 `BaseBackend`。现在用的是 wheel 里的 NVIDIA backend。
- 编译并运行上游的 `triton-opt` 和 lit。源码在 submodule 的 `bin/` 和 `test/`，测试仍用上面的 shell 脚本读 `// RUN:`。
- 把四个手写 pass 编进 `libtriton.so` 并在 `make_ttir` 里调用。对照见第 2.2 节。
