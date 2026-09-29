# Triton 编译器前中端：流水线、Python 与 C++ 的交互、各种运行状态

这份文档回答四件事：一条 `@triton.jit` kernel 怎么变成 TTIR、TTGIR 再到 cubin；Python 和 C++ 在哪里交接；本仓库里有哪几种「跑编译器的东西」，各自是什么状态；工业界怎么组织这些东西，本仓库缺什么、补了什么。

术语：TTIR 指 `tt` 方言，TTGIR 指 `triton_gpu`（`ttg`）方言。本仓库走 Triton 自己的 MLIR，不经过 StableHLO、XLA 或 Linalg。

## 0. 一页总览

```text
┌────────────────────────────── Python 进程 ──────────────────────────────┐
│ @triton.jit 函数                                                        │
│   │ inspect.getsource + ast.parse            （Python 标准库）          │
│   ▼                                                                     │
│ CodeGenerator.visit_*  ──调用 tl.* 的 @builtin，传入 _builder──┐        │
│                                                               ▼        │
│                                       ir.builder.create_*  ══ pybind11 ═╪══► C++：TritonOpBuilder
│                                                                         │      创建 tt.* / scf.* / arith.* op
│ 第一份 TTIR（C++ 对象，Python 只持有句柄）  ◄═════════════════════════════╪═════
│   │                                                                     │
│ compile(): stages = {ttir, ttgir, llir, ptx, cubin}，按顺序调用          │
│   │  每个 stage 里：pm = ir.pass_manager(ctx)                           │
│   │                 passes.xxx.add_yyy(pm, ...)   ══ pybind11 ══► C++：pm.addPass(createYyyPass())
│   │                 pm.run(module)                ══ pybind11 ══► C++：PassManager::run（期间不回调 Python）
│   ▼                                                                     │
│ ttir ─► ttgir ─► llir ─► ptx ─► cubin                                   │
│  tt      ttg     LLVM IR  文本    subprocess: ptxas                      │
└─────────────────────────────────────────────────────────────────────────┘
```

`libtriton.so` 是一个 pybind11 模块（`triton._C.libtriton`）。Python 负责遍历 AST、排 pass 顺序、缓存和 launch；op 的创建、所有 pass、layout、LLVM lowering 都是 C++。本仓库额外有一个独立的 C++ 可执行文件 `build/bin/tt-opt`，它和 Python 之间只共享 IR 文本。

## 1. 先分清五种「东西」

改代码之前先问：我改的东西属于哪一行。

| 名称 | 是什么 | 语言 | MLIR / LLVM | 有 `triton_gpu` | 有本仓库手写 pass | 被 `@triton.jit` 使用 |
|---|---|---|---|---|---|---|
| A. 已安装的 Triton 3.1.0 | venv 里的 wheel：Python 包 + `_C/libtriton.so`（约 450MB）+ `backends/nvidia/bin/ptxas` | Python + C++ | `libtriton.so` 静态带着 Triton 固定的 LLVM | 有 | 无 | 是，唯一被 JIT 运行的一份 |
| B. Triton submodule | `third_party/triton`，v3.1.0，不进本仓库的提交 | 源码 | 上游自己的 CMake。本仓库不拿它编 `libtriton.so` | 源码有 | 无。改动在 `third_party/patches/`，打进 `build/triton-patched` | 否。只用来读 |
| C. `build/bin/tt-opt` | 本仓库 CMake 编出的 MLIR opt，入口 `compiler/tt-opt.cpp` | C++ | 系统 LLVM 23（`/opt/torch-mlir/.../build`） | 没有 | 有 | 否 |
| D. `python -m triton_llm.tt_opt` | Python 驱动，把 flag 映射到 A 里的 `passes.*.add_*` | Python | 用 A 的 `libtriton.so` | 有 | 无 | 否，但和 A 同一份 C++ |
| E. 工业界自编的 Triton | 自己编 `libtriton.so`（JIT 用）和 `triton-opt`（测试和调试用），共用一套 C++ pass 库 | Python + C++ | 自己固定的 LLVM | 有 | 有 | 是 |

本仓库没有 E。A 的 wheel 不带 `triton-opt`，`.venv/bin` 里也没有。B 是 submodule，源码文件不进本仓库的提交；哪些目录参与编译、哪些只供阅读，见 [`third_party/patches/README.md`](../third_party/patches/README.md)。上游的 `bin/triton-opt.cpp` 在 B 里，只供阅读，本仓库不编译它。C 承担的是「E 里 `triton-opt` 那一半」，但只注册了 `tt`，没有 `triton_gpu`。

三个容易混的事实：

1. CMake 把 B 复制到 `build/triton-patched` 并打上 `third_party/patches/*.patch`，再编译补丁后的 `.cpp`。产物只进 C，不进 A 的 `.so`。B 里的 Python 也不会被 JIT 执行，JIT 加载的是 A 的 `triton/*.py`。
2. A 和 C 用不同版本的 LLVM。它们之间没有 ABI，只有 IR 文本。见 7.7。
3. A 的 `passes.ttir` 里现在只有 `add_combine`、`add_convert_to_ttgpuir`、`add_reorder_broadcast`、`add_rewrite_tensor_pointer`。`add_annotate_dot_stages` 写在补丁里的 `passes.cc`，要等 Triton 重编并装回 venv 后才存在。

## 2. Python 与 C++ 怎么交互

### 2.1 一个 pybind 模块，几个子模块

[`python/src/main.cc`](../third_party/triton/python/src/main.cc) 用 `PYBIND11_MODULE(libtriton, m)` 建立模块，再挂子模块：

| 子模块 | 由谁注册 | 主要内容 |
|---|---|---|
| `ir` | [`ir.cc`](../third_party/triton/python/src/ir.cc) `init_triton_ir` | `context`、`builder`、`module`、`pass_manager`、`parse_mlir_module`、`load_dialects` |
| `passes` | [`passes.cc`](../third_party/triton/python/src/passes.cc) | 子模块 `common`、`convert`、`ttir`、`ttgpuir`、`llvmir`、`analysis`，每个 `add_*` 往 `pass_manager` 里追加一个 pass |
| `llvm` | [`llvm.cc`](../third_party/triton/python/src/llvm.cc) | MLIR LLVM dialect → LLVM IR、优化、NVPTX 汇编 |
| `nvidia` | `third_party/nvidia/triton_nvidia.cc` | `load_dialects`、`ClusterInfo`、`passes.ttgpuir` / `passes.ttnvgpuir` |
| `interpreter` | `interpreter.cc` | `TRITON_INTERPRET=1` 时 host 原子操作用的 C++ 支撑。解释器本身不跑 MLIR pass，见 2.6 |

在本机 venv 里 `dir()` 可以直接验证，例如 `ir` 里有 `builder context load_dialects module operation parse_mlir_module pass_manager`。

### 2.2 三种交互方式

**(a) builder 调用：前端 → C++。** `CodeGenerator` 持有一个 `ir.builder`。`tl.dot` 之类的函数写成 Python，但最终调用 `builder.create_dot(...)`，这是 `ir.cc` 里 `TritonOpBuilder` 的 pybind 绑定，在 C++ 里创建 `tt.dot`。返回值 `tl.tensor.handle` 是 C++ 的 `Value`，Python 只是拿着它。因此要新增一个前端可见的 op，必须改 C++ 并重编。

**(b) pass 注册：Python 排序，C++ 执行。** [`passes.h`](../third_party/triton/python/src/passes.h) 的 `ADD_PASS_WRAPPER_N` 展开成：

```cpp
m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); });
```

`passes.ttir.add_combine(pm)` 只是把一个 `createCombineOpsPass()` 对象放进 `pm`。`pm.run(mod)` 才进入 C++ 并阻塞到跑完，中途不回调 Python；失败时抛 `RuntimeError("PassManager::run failed")`。带参数的 pass 用 `ADD_PASS_WRAPPER_1..4`，例如 `add_annotate_dot_stages(pm, sm)`。

**(c) 文本 IR：任意两端都靠它交换。** `module.str()` 输出文本，`ir.parse_mlir_module(path, ctx)` 读文本。编译缓存里每个 stage 的产物就是文本文件；`TRITON_KERNEL_OVERRIDE=1` 用改过的 `.ttir` / `.ttgir` 替换 stage 输出；`IRSource` 允许直接从 `.ttir` / `.ttgir` 文件开始编译。`build/bin/tt-opt` 与 Python 之间唯一的接口就是这个。

### 2.3 `pass_manager` 上的环境变量

[`ir.cc`](../third_party/triton/python/src/ir.cc) 的 `pass_manager.enable_debug` / `run` 读这些环境变量：

| 变量 | 作用 |
|---|---|
| `MLIR_ENABLE_DUMP=1` | 每个 pass 前后打印 IR。会 `disableMultithreading` |
| `MLIR_ENABLE_DIAGNOSTICS=1` | 诊断带 op 文本和栈 |
| `MLIR_ENABLE_TIMING=1` | 每个 pass 的耗时 |
| `TRITON_REPRODUCER_PATH` | 写 MLIR reproducer，可以拿去给 `triton-opt` 复现 |
| `TRITON_ENABLE_LLVM_DEBUG` / `TRITON_LLVM_DEBUG_ONLY` | 打开 LLVM `-debug` / `-debug-only` |

看每一步的 IR 要同时设 `TRITON_ALWAYS_COMPILE=1`，否则命中缓存后 pass 根本不跑。

### 2.4 Python 标准库、`triton.language.standard` 和 C++ 各管什么

| 层 | 出现的位置 | 做什么 |
|---|---|---|
| Python 标准库 | `runtime/jit.py`：`inspect.getsource` / `getsourcelines` / `signature`、`textwrap.dedent`、`ast.parse` | 拿到 kernel 源码并解析成 `ast` |
| | `compiler/compiler.py`：`hashlib.sha256`、`json`、`functools.lru_cache`、`pkgutil.walk_packages`、`os.environ` | 缓存 key、元数据、环境变量 |
| | `nvidia/compiler.py`：`subprocess.run` + `tempfile`、`re` | 调 `ptxas`；用正则从 PTX 里取 `.visible .entry` 的名字 |
| | `nvidia/driver.py` + `runtime/build.py`：`subprocess.check_call`、`importlib` | 生成 launcher 的 C 源码，用 `gcc` 或 `clang`（或 `CC`）编译再当 Python 扩展加载 |
| `triton.language`（`core.py`、`semantic.py`） | 已安装包 | `@builtin` 函数、类型检查，最后调 `builder.create_*` |
| `triton.language.standard` | 已安装包，Python 写成的 `@jit` 函数 | `tl.sum` 的 combine 函数、`softmax` 等。它们和用户 kernel 走同一个前端被内联进来。lesson 的 IR 里能看到 `loc(.../triton/language/standard.py:267)` |
| C++（`libtriton.so`） | 所有 op、类型、layout、pass、LLVM lowering | 见第 4 到 6 节 |

Python 标准库只做「拿源码、解析、缓存、调外部工具」，不做编译。`triton.language.standard` 不是标准库，是 Triton 自己用 Python 写的函数库。

### 2.5 一次 launch 的时序，以及缓存 key

`JITFunction.run`（`runtime/jit.py`）：

1. `create_binder` 第一次调用时把 `triton.compiler.compile` 等引用存到函数对象上。
2. 用 `sig_and_spec` 和 constexpr 值拼出内存缓存 key。命中就直接 launch。
3. 未命中：`backend.parse_options`，然后 `ASTSource(self, signature, constants, attrs)`，再调 `compile(src, target, options)`。
4. `compile()` 先算磁盘缓存 key：

   ```text
   triton_key() - src.hash() - backend.hash() - options.hash() - 会使缓存失效的环境变量
   ```

   `triton_key()` 哈希已安装包的 `compiler/*.py`、`backends/*`、`language/*.py`，以及 **`libtriton.so` 的全部字节**；`backend.hash()` 是 `ptxas` 版本和 compute capability。缓存目录默认 `~/.triton/cache`，由 `TRITON_CACHE_DIR` 改。
5. 未命中磁盘缓存时按 `stages` 顺序跑：`make_ir` 得第一份 TTIR，然后 `ttir → ttgir → llir → ptx → cubin`，每个 stage 的结果写进缓存。
6. 返回 `CompiledKernel`，`kernel.run(...)` 调用 launcher，launcher 是上面 C 源码编出来的扩展，里面调 CUDA driver API。

对本仓库的影响：换掉 `libtriton.so`（重编 Triton）会让所有缓存失效；改 B 里的文件、重编 `build/bin/tt-opt` 都不影响这个 key，也就不会改变 JIT 的行为。

### 2.6 `TRITON_INTERPRET=1` 不进编译流水线

已安装的 `runtime/jit.py` 里，`jit()` 在这个环境变量为 `"1"` 时返回 `InterpretedFunction`，不返回 `JITFunction`。因此 `compile()`、`make_ir`、`make_ttir` 都不会被调用。

`runtime/interpreter.py` 的 `GridExecutor` 先把带 `data_ptr` 的参数 `.cpu()` 拷到 host，再对 grid 的每个 `(x, y, z)` 在 Python 里调用原函数，最后把结果拷回设备。`_patch_lang` 把 `tl.*` 换成 `InterpreterBuilder`，张量数据是 numpy 数组。原子操作走 `libtriton.interpreter`（`interpreter.cc` 里的 `__atomic_*`）。它还 import 了 `libtriton.ir`，只用其中的枚举（`ROUNDING_MODE`、`MEM_SEMANTIC`、`ATOMIC_OP`、`PADDING_OPTION`），不建 `ModuleOp`，也不跑 pass。本仓库没有测试这条路径。

## 3. 前端：`@triton.jit` → 第一份 TTIR

### 3.1 调用链

1. [`compiler.py`](../third_party/triton/python/triton/compiler/compiler.py) 的 `compile()` 建 `ir.context()`，`ir.load_dialects`、`backend.load_dialects` 注册方言，然后 `src.make_ir(...)`。
2. `ASTSource.make_ir` 调 `ast_to_ttir`。
3. [`code_generator.py`](../third_party/triton/python/triton/compiler/code_generator.py) 的 `ast_to_ttir` 构造 `CodeGenerator`，`generator.visit(fn.parse())` 遍历 AST，往 `ir.builder` 写 `tt.func`、`tt.load`、`tt.dot`、`scf.for`。`fn.parse()` 是 `ast.parse(self.src)`。返回的 `generator.module` 就是第一份 TTIR，此时还没跑 `make_ttir`。

### 3.2 `CodeGenerator` 里几个要认的点

- **`visit_Call`**：被调用的对象是 `@builtin`（例如 `tl.dot`）时，追加 `_builder=self.builder` 再调用，Python 端的检查和 `create_*` 都在这次调用里发生。被调用的是另一个 `@triton.jit` 函数时走 `call_JitFunction`，生成 `tt.call`，`make_ttir` 开头的 inliner 再把它内联。
- **`visit_For`**：`range(...)` 生成 `scf.for`。循环体里要有一个归纳变量的占位，前端用 `builder.create_undef(...)`，在文本里就是 `llvm.mlir.undef : i32`（`code_generator.py` 约 952 行）。这意味着任何要读真实前端 TTIR 的工具都必须注册 `llvm` 方言，本仓库的 `tt-opt` 就是因为这一点补上了 `LLVMDialect`。
- **`visit_If`**：`if` 生成 `scf.if`，两个分支里被修改的变量成为它的结果。
- **specialization**：进入 `CodeGenerator` 之前，`JITFunction._get_config` 已经算出 `divisible_by_16` 和 `equal_to_1`。前者在 `tt.func` 参数上变成 `tt.divisibility = 16`，后者的参数被折成常量。`do_not_specialize=["M", "N", "K"]` 可以让它们留作参数。
- **`tl.constexpr`**：值不出现在参数列表，直接决定张量形状，例如 `BLOCK_M=16` 的 tile 是 `tensor<16x16x...>`。不同 constexpr 值是不同的编译单元。

### 3.3 一个 kernel 对上哪些 op

[`lesson.py`](../python/triton_llm/compiler/lesson.py) 的 `_frontend_lesson` 把这些构造放在同一个 kernel 里。[`frontend.py`](../python/triton_llm/compiler/frontend.py) 的 `capture_frontend_ttir(jit_fn, launch)` 在 `compile()` 进入 `make_ttir` 之前取出 `make_ir` 的结果。`CompiledKernel.asm["ttir"]` 已经跑完 `make_ttir`，里面没有 `tt.make_tensor_ptr`，所以不能用它演示前端。

| Python | 第一份 TTIR |
|---|---|
| `tl.program_id` | `tt.get_program_id` |
| `tl.arange` | `tt.make_range` |
| 指针加偏移 | `tt.addptr` |
| `tl.load` / `tl.store`，带 mask | `tt.load` / `tt.store` |
| `for` | `scf.for`（带 `llvm.mlir.undef` 占位） |
| `if` | `scf.if` |
| `tl.dot` | `tt.dot` |
| `tl.sum` | `tt.reduce` |
| 标量铺到 tile | `tt.splat` |
| `tl.make_block_ptr` | `tt.make_tensor_ptr` |
| `tl.constexpr` 的 `BLOCK_*` | 张量形状 `tensor<16x16x...>` |
| 16 字节对齐的指针 | 参数属性 `tt.divisibility = 16` |
| `do_not_specialize` 的 `M`、`N`、`K` | 留在 `tt.func` 参数里 |

[`tests/python/test_frontend_middle.py`](../tests/python/test_frontend_middle.py) 锁住这张表。

### 3.4 新增一个 `tl.xxx` 要改哪里

1. C++：在 `TritonOps.td` 定义 op，在 `ir.cc` 里 `.def("create_xxx", ...)` 暴露。
2. Python：`triton/language/core.py` 用 `@builtin` 写 `tl.xxx`，`semantic.py` 写检查并调 `builder.create_xxx`。
3. 只有需要新语法时才动 `code_generator.py` 的 `visit_*`。

改完必须重编 Triton。只把已有 op 组合成新函数，写一个 `@triton.jit` 辅助函数就行，不用重编。本仓库的 `tt.fused_dot_mul` 由 pass 生成，前端看不到，所以不需要 `create_fused_dot_mul`。

## 4. 中端（一）：TTIR 上的 pass，即 `make_ttir`

顺序在 [`compiler.py`](../third_party/triton/third_party/nvidia/backend/compiler.py) 的 `CUDABackend.make_ttir`，也是 `python -m triton_llm.tt_opt --make-ttir` 展开的顺序：

| pass | 做什么 | 源码 |
|---|---|---|
| `inline` | 内联 `tt.call` | 上游 MLIR |
| `triton-rewrite-tensor-pointer` | `tt.make_tensor_ptr` / `tt.advance` 改成普通指针张量的 `tt.load` / `tt.store` | `RewriteTensorPointer.cpp` |
| `triton-combine` | 折叠 `addptr(addptr(...))`、dot 加 acc 等 | `Combine.cpp` + `Combine.td` |
| `canonicalize` | 每个 op 自己的 canonicalize / fold | 上游 MLIR |
| `triton-reorder-broadcast` | 把 `tt.splat` / `tt.broadcast` 挪到逐元素运算之后 | `ReorderBroadcast.cpp` |
| `cse`、`licm`、`symbol-dce` | 公共子表达式、循环不变量外提、无用函数 | 上游 MLIR |

这里没有本仓库的手写 pass。引进的 `compiler.py` 与上游 3.1 的 `make_ttir` 相同。这一阶段仍是 `tt` 方言，张量没有 layout。

单独跑其中一个 pass（用 D，也就是已安装的 `.so`）：

```bash
source .venv/bin/activate
export PYTHONPATH=python
python -m triton_llm.tt_opt tests/tt/combine_addptr.mlir --triton-combine
python -m triton_llm.tt_opt tests/tt/reorder_broadcast.mlir --triton-reorder-broadcast
python -m triton_llm.tt_opt --list --sm 70
```

`--triton-combine` 调用 `passes.ttir.add_combine`。flag 表在 [`pipeline.py`](../python/triton_llm/compiler/pipeline.py)。

## 5. 中端（二）：TTGIR，即 `make_ttgir`，以及 layout 和分析

### 5.1 顺序和 sm 门槛

`make_ttgir` 的顺序在同一个 `compiler.py`。关键 pass 与源码：

| pass | 做什么 | 源码 |
|---|---|---|
| `convert-triton-to-tritongpu` | `tt` → `triton_gpu`，给每个值加 `#blocked` layout，参数 `num_warps`、warp size 32、`num_ctas` | `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` |
| `tritongpu-coalesce` | 让 load / store 的 layout 有利于合并访问 | `Coalesce.cpp` |
| `tritongpu-plan-cta` | 多 CTA 规划 | `TritonNvidiaGPU/Transforms/PlanCTA.cpp` |
| `tritongpu-remove-layout-conversions` | 删掉多余的 `convert_layout`（会多次运行） | `RemoveLayoutConversions.cpp` |
| `tritongpu-accelerate-matmul` | 给 `tt.dot` 选 MMA layout | `AccelerateMatmul.cpp` |
| `tritongpu-optimize-dot-operands` | 选 dot 操作数 layout | `OptimizeDotOperands.cpp` |
| `tritongpu-pipeline` | software pipeline，**仅 `sm // 10 >= 8`** | `Pipeliner/SoftwarePipeliner.cpp` |
| `tritongpu-prefetch`、`reduce-data-duplication`、`reorder-instructions` | 预取、减少 shared memory 重复、指令重排 | 同目录 |
| `tritongpu-fence-insertion`、`tma-lowering` | Hopper 专用，**仅 `sm // 10 >= 9`** | `TritonNvidiaGPU/Transforms/` |

本机 V100 是 70：pipeline、f32-dot-tc、fence、TMA 都不会出现在 `--list --sm 70` 的 `make-ttgir` 里，所以 `num_stages=2` 在 sm70 上只是 launch 提示，Triton 的软件流水线没有运行。单独把 `--tritongpu-pipeline` 写在命令行上，仍会调用那个 C++ pass。

### 5.2 layout（V100 上的实际输出）

TTIR 的张量没有 layout。lesson kernel 在 `sm=70` 上经 `--make-ttir --convert-triton-to-tritongpu --tritongpu-accelerate-matmul` 后：

```text
#mma = #triton_gpu.nvidia_mma<{versionMajor = 1, versionMinor = 11, warpsPerCTA = [1, 1], instrShape = [16, 16]}>
%126 = triton_gpu.convert_layout %122 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
                                        -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
%128 = tt.dot %126, %127, %125, inputPrecision = tf32 : ... -> tensor<16x16xf32, #mma>
%129 = triton_gpu.convert_layout %128 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
```

- `#blocked<{sizePerThread, threadsPerWarp, warpsPerCTA, order}>`：元素怎么分给线程和 warp。
- `#nvidia_mma`：`versionMajor = 1` 是 Volta 的 `mma.sync`；sm_80 是 2，sm_90 用 `wgmma`（v3）。
- `#dot_op<{opIdx, parent}>`：dot 的 A / B 操作数要的 layout。
- `#slice<{dim, parent}>`：把一维值放回二维 layout 的降维视图。
- `convert_layout`：两个 layout 之间的搬运，落到 LLVM 时经过 shared memory。`remove-layout-conversions` 的目标就是删掉它们。

[`test_frontend_middle.py`](../tests/python/test_frontend_middle.py) 断言了 `blocked`、`nvidia_mma versionMajor = 1`、`dot_op parent = #mma` 和 `convert_layout`。手写的 `tt.func` 参数没有 layout，喂给 `--convert-triton-to-tritongpu` 会因 signature 和 entry block 类型不一致报错，所以 TTGIR pass 的输入要用从 kernel 打出来的 `.ttir`（第 8 节）。

### 5.3 分析（不是单独的 pass）

`lib/Analysis/`：

| 文件 | 用途 |
|---|---|
| `AxisInfo.cpp` | 推算指针 / 整数的连续性、可整除性、常量性。Coalesce 和 LLVM lowering 的向量化都读它 |
| `Allocation.cpp` | 算 shared memory 需要多大、各 buffer 何时活跃，`allocate-shared-memory` 用 |
| `Membar.cpp`、`Alias.cpp` | 同步屏障插入和别名判断 |
| `Utility.cpp`、`ReshapeDecomposition.cpp` | 通用工具 |

## 6. 中端之后：LLVM、PTX、cubin

同一个 `compiler.py`：

- `make_llir`：又建一个 `pass_manager`，跑 `decompose-unsupported-conversions`、`scf-to-cf`、`allocate-shared-memory`、`TritonGPUToLLVM`（`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`）、`nvgpu-to-llvm`、`arith-to-llvm` 等；然后 `llvm.to_module` 把 MLIR 的 LLVM dialect 变成真正的 LLVM IR，`llvm.optimize_module(O3)`，返回文本。
- `make_ptx`：`llvm.translate_to_asm`，triple `nvptx64-nvidia-cuda`，CPU `sm_<capability>`。这是 LLVM 的 NVPTX 后端，不是 Triton 自己写的汇编器。
- `make_cubin`：`subprocess.run` 调 wheel 里的 `ptxas`。没有对应的 Triton C++ pass。

Volta 的 `mma.sync` 落在 `DotOpToLLVM/MMAv1.cpp`，Hopper 的 `wgmma` 在 `WGMMA.cpp`。本仓库不重写这一段，这些文件只供阅读。`tt-opt` 编译的是打过补丁的 `build/triton-patched`，补丁在 [`third_party/patches/`](../third_party/patches/README.md)。

## 7. 手写中端：`build/bin/tt-opt`（第 1 节的 C）

### 7.1 它是什么

[`compiler/tt-opt.cpp`](../compiler/tt-opt.cpp) 是一个 MLIR opt 的 `main`。CMake 把它和 pass 的 `.cpp` 链成 `build/bin/tt-opt`：

```cmake
add_executable(tt-opt
  compiler/tt-opt.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/Transforms/AnnotateDotStages.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/Transforms/FuseAndTileDot.cpp)
```

`TRITON_SRC` 是 `build/triton-patched`。这两个 `.cpp` 不在上游树里，由 [`third_party/patches/`](../third_party/patches/README.md) 加进去。

定义在 [`compiler/CMakeLists.txt`](../compiler/CMakeLists.txt)。它链接本仓库编译的静态库 `TritonIR`（`tt` 的 op，加上为了能编过 `Traits.cpp` 而带上的 GPU dialect 实现）和系统 LLVM 23 的 `MLIROptLib`，不加载 venv 的 `libtriton.so`。

```bash
cmake -B build -DTRITON_LLM_CUDA_ARCH=70
cmake --build build --target tt-opt -j"$(nproc)"
./build/bin/tt-opt --help
```

`--help` 里能看到的 pass，就是 `main` 里注册过的那些：`canonicalize`、`triton-annotate-dot-stages`、`triton-fuse-dot-epilogue`、`triton-lower-fused-dot-mul`、`triton-tile-dot`。

### 7.2 `main` 做的三件事

1. `DialectRegistry` 里插入 `tt`、`arith`、`func`、`math`、`scf`、`cf`、`tensor`、`llvm`。解析器只认识这些方言。`llvm` 是为了能解析前端输出里的 `llvm.mlir.undef`；`tensor` 是 K 切分要用的 `tensor.extract_slice`。没有 `triton_gpu`，所以不能读 TTGIR，也不能跑 `--convert-triton-to-tritongpu`。
2. `registerCanonicalizerPass()` 和四次 `registerPass(...)` 把 pass 放进 MLIR 全局 pass registry。lambda 调用 `create*Pass()`，默认 `sm=70`。
3. `MlirOptMain(argc, argv, ..., registry)` 读位置参数上的 `.mlir`，按命令行 flag 建 pipeline，跑完把模块打到 stdout。

flag 不写在 `tt-opt.cpp` 里。补丁改的是上游 `Passes.td`，打完之后在 `build/triton-patched/include/triton/Dialect/Triton/Transforms/Passes.td`：

```tablegen
def TritonAnnotateDotStages : Pass<"triton-annotate-dot-stages", "mlir::ModuleOp"> {
  let constructor = "mlir::triton::createAnnotateDotStagesPass()";
  let options = [
    Option<"targetSm", "sm", "int32_t", "70", "compute capability">
  ];
}
```

Tablegen 生成 `getArgument() == "triton-annotate-dot-stages"` 和选项 `sm`。`--triton-annotate-dot-stages=sm=90` 在 pass 已经构造之后由 MLIR 把 `sm` 写进 `targetSm`，所以 lambda 里不读 `argv`，`sm=90` 仍然生效。

`createAnnotateDotStagesPass(int32_t)` 的整型参数是给 `passes.cc` 的 `add_annotate_dot_stages(pm, sm)` 用的，`tt-opt` 命令行不走它。

### 7.3 四个 pass 各做什么

| flag | 源码 | 作用 |
|---|---|---|
| `--triton-annotate-dot-stages` | `AnnotateDotStages.cpp` | 给每个 `tt.dot` 写 `triton_llm.num_stages`：`sm < 80` 为 2，否则 3 |
| `--triton-fuse-dot-epilogue` | `FuseAndTileDot.cpp` | `arith.mulf` 的一边是单次使用的 `tt.dot`、scale 类型与 dot 结果相同，就折成 `tt.fused_dot_mul` |
| `--triton-lower-fused-dot-mul` | 同上 | 把 `tt.fused_dot_mul` 展开回 `tt.dot` 加外面的 `arith.mulf` |
| `--triton-tile-dot` | 同上 | 静态二维 `tt.dot`：K 大于所选 `BLOCK_K` 且能整除时，改写成 `scf.for` + `tensor.extract_slice` + 内层 `tt.dot` |

```bash
./build/bin/tt-opt tests/tt/annotate_dot_stages.mlir --triton-annotate-dot-stages
./build/bin/tt-opt tests/tt/annotate_dot_stages_sm90.mlir --triton-annotate-dot-stages="sm=90"
./build/bin/tt-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue
./build/bin/tt-opt tests/tt/fuse_then_tile.mlir \
  --triton-fuse-dot-epilogue --triton-lower-fused-dot-mul --triton-tile-dot
./build/bin/tt-opt tests/tt/tile_dot.mlir --triton-tile-dot
./build/bin/tt-opt tests/tt/tile_dot_sm90.mlir --triton-tile-dot="sm=90"
```

**融合。** 补丁在上游 `TritonOps.td` 里加上 `TT_FusedDotMulOp`（`tt.fused_dot_mul`），这就是本仓库的 TTMyIR：在 `tt` 上加一个 op，而不是另起方言。它带 `AllTypesMatch<["c", "scale", "d"]>`，类型对不上时报 `failed to verify that all of {c, scale, d} have same type`。

**先融合、再下沉、再切 K。** lower 之后 mul 又回到 dot 外面，`triton-tile-dot` 把 dot 拆进 `scf.for`，循环的初值是原来的累加器，mul 仍留在 `scf.for` 之后。

**确定的 tiling。** 切分表在 `FuseAndTileDot.cpp` 的 `gemmTileForSm`，与 [`arch/tiling.py`](../python/triton_llm/arch/tiling.py) 的候选相同：`(64,64,32,2)`、`(64,128,32,2)`、`(128,128,32,2)`、`(128,128,64,2)`、`(128,128,64,3)`。最后一个同时满足 stage 上限和 shared memory 的候选胜出。sm<80 时 stage 上限 2、shared memory 96KB；sm>=80 时 stage 上限 3、164KB；sm>=90 时 228KB。sm70 和 sm90 都得到 `BLOCK_K=64`，stage 分别是 2 和 3，属性写在 `scf.for` 结尾的 `{triton_llm.block_k, triton_llm.num_stages}` 上。

这不是 `@triton.autotune`（搜索），也不是 `triton.heuristics`（运行时 Python 回调）。一个 `sm` 对应一张确定的 tile。Python 侧的 `choose_gemm_tile` 给 JIT kernel 的 `tl.constexpr` 用同一张表，C++ pass 在已生成的 TTIR 上改 K 循环。两份拷贝由 [`tests/python/test_tile_table_sync.py`](../tests/python/test_tile_table_sync.py) 对 sm 70、75、80、86、90 逐个比较，谁改了表而另一份没改，测试会失败。

### 7.4 再加一个 pass 时改哪里

1. 在 `Passes.td` 里加 `def`，给出 CLI 名和 `constructor`。
2. 新的 `.cpp` 里定义 pass 类和 `create*Pass()`。`AnnotateDotStages.cpp` 是模板：`GEN_PASS_DECL_*` 加 `GEN_PASS_DEF_*`，再写 `runOnOperation`。
3. 在 `tt-opt.cpp` 的 `main` 里再 `registerPass` 一次。pass 如果产生新方言，把方言插进同一个 `DialectRegistry`。要读 TTGIR，就在这里 `registry.insert<triton::gpu::TritonGPUDialect>()`。
4. 把新的 `.cpp` 加进 `compiler/CMakeLists.txt` 的 `add_executable(tt-opt ...)`。
5. 在 `tests/tt/` 放一份 `.mlir`。`// RUN:` 必须写成 `TT_OPT_CPP`，不能写成 `tt-opt`。`tests/shell/run_tt_opt_tests.sh` 先把单词 `tt-opt` 换成 `python -m triton_llm.tt_opt`，再把 `TT_OPT_CPP` 换成 `build/bin/tt-opt`。写成 `tt-opt` 会跑到 Python 驱动上，那个进程里没有你的新 pass。
6. 新增 op 时，同时写 verifier（可以用 ODS trait）和一个 `-verify-diagnostics` 负例，参考 `tests/tt/fused_dot_mul_verify.mlir`。

然后：

```bash
cmake --build build --target tt-opt -j"$(nproc)"
./tests/shell/run_tt_opt_tests.sh
```

`tests/tt/combine_addptr.mlir` 和 `reorder_broadcast.mlir` 的 `// RUN:` 是 `tt-opt`，所以它们走 Python 驱动。`--triton-combine`、`--triton-reorder-broadcast` 只存在于已安装的 `libtriton.so`。

### 7.5 和 JIT 的关系

A 的 `@triton.jit` 不会跑到 `build/bin/tt-opt`。`libtriton.so` 的符号是 hidden 的，改 `AnnotateDotStages.cpp` 不会进那个 `.so`。

`passes.cc` 里已经写了 `add_annotate_dot_stages(pm, sm)`，要等 Triton 重编并装回 venv 之后才存在，然后在**安装目录**的 `compiler.py` 里调用 `passes.ttir.add_annotate_dot_stages(pm, capability)`。不要改引进的这份 `compiler.py`，否则 `--make-ttir` 和已安装的 Triton 就不再是同一条顺序。融合和切分两个 pass 同理：`passes.cc` 里目前没有它们的绑定。

### 7.6 Python 抓到的 TTIR 直接喂给 C++ 的 tt-opt

```bash
PYTHONPATH=python python - <<'EOF'
from pathlib import Path
from triton_llm.compiler.frontend import capture_frontend_ttir
from triton_llm.compiler.lesson import _frontend_lesson, launch_frontend_lesson
Path("/tmp/lesson.mlir").write_text(capture_frontend_ttir(_frontend_lesson, launch_frontend_lesson))
EOF
./build/bin/tt-opt /tmp/lesson.mlir --triton-annotate-dot-stages
```

[`tests/python/test_python_cpp_handoff.py`](../tests/python/test_python_cpp_handoff.py) 做的就是这件事，并检查 `tt.dot` 上写出了 `triton_llm.num_stages`。这条路成立，靠的是 `tt-opt` 注册了 `llvm` 方言。

### 7.7 IR 文本是两个 MLIR 之间唯一的接口

A 的 `libtriton.so` 用 Triton 固定的那版 LLVM，C 用 LLVM 23。所以：

- 文本语法可能出现版本差异。为了在 LLVM 23 上编译 `tt-opt`，补丁里对上游源码做过 API 移植。改了哪些文件看 [`third_party/patches/`](../third_party/patches/README.md)。
- 不要拿 A 里的 C++ 对象去和 C 的对象互传。两边没有共享内存，也没有共享的类型注册。
- 从 C 输出的 IR 回到 A 里继续跑 pass，要先确认 A 能 `parse_mlir_module`。自定义的 `tt.fused_dot_mul` A 就不认识，所以融合后必须先 `--triton-lower-fused-dot-mul`。

## 8. 用 kernel 产生中端夹具

```bash
source .venv/bin/activate
PYTHONPATH=python python scripts/dump_triton_ir.py
```

`artifacts/triton_ir/<op>/*.ttir` 是已安装 Triton 的 `make_ttir` 整段跑完之后的模块。`tt.make_tensor_ptr` 已经不在里面，因为 `RewriteTensorPointer.cpp` 在 `make_ttir` 内部。要看这个 pass 本身，输入得还带着 `tt.make_tensor_ptr`，用 `capture_frontend_ttir` 取。

把这份 `.ttir` 送进已安装的 TTGIR pass：

```bash
python -m triton_llm.tt_opt artifacts/triton_ir/swiglu/_swiglu_kernel.ttir \
  --convert-triton-to-tritongpu --tritongpu-accelerate-matmul
```

stdout 里 `tt.func` 参数会带上 `#triton_gpu.blocked` / `dot_op` layout。这份 TTGIR 不能喂给 `build/bin/tt-opt`，见 7.2。

DSL 入口在 [`python/triton_llm/ops/`](../python/triton_llm/ops/)。数值参考与 op 签名一一对应，不参与 pass：

| op | 参考 |
|---|---|
| `ops.flash_attention` | [`reference/flash_attention.py`](../python/triton_llm/reference/flash_attention.py) |
| `ops.paged_attention` | [`reference/paged_attention.py`](../python/triton_llm/reference/paged_attention.py) |
| `ops.rmsnorm_rope` | [`reference/norm_rope.py`](../python/triton_llm/reference/norm_rope.py) |
| `ops.swiglu` | [`reference/swiglu.py`](../python/triton_llm/reference/swiglu.py) |

## 9. 工业界的 pipeline，以及每一段要不要自己写

### 9.1 工业界怎么组织

- **一套 C++ pass 库，两个入口。** JIT 用的 `libtriton.so` 通过 `passes.cc` 把 pass 暴露给 Python。调试和测试用的 `triton-opt` 在上游标签 v3.1.0（`cf34004b`）的 `bin/triton-opt.cpp`：`main` 只调用 `registerTritonDialects(registry)`，然后 `MlirOptMain`。注册内容在同目录的 `RegisterTritonDialects.h`：`registerAllPasses()`、`registerTritonPasses()`、`registerTritonGPUPasses()`、`registerTritonNvidiaGPUPasses()`，再加上 Triton→TTGIR、TTGIR→LLVM、NVGPU→LLVM、AMD 的 conversion，以及四个 `mlir::test` 分析 pass。方言插进去的是 `tt`、`triton_gpu`、`triton_nvidia_gpu`、`nvgpu`、`arith`、`math`、`scf`、`cf`、`gpu`、`llvm`、`nvvm`、`rocdl`，不是 MLIR 的全部方言（没有单独插入 `func`、`tensor`）。同一 `bin/` 里还有 `triton-llvm-opt`、`triton-lsp`、`triton-reduce`。`test/lit.cfg.py` 用 `lit.formats.ShTest`，后缀 `.mlir` 和 `.ll`，把工具名 `triton-opt`、`triton-llvm-opt` 替换成构建目录里的二进制，并设 `FILECHECK_OPTS=--enable-var-scope`。`bin/` 和 `test/` 在 B 里，本仓库不编译、也不跑这套 lit。C 对应「`triton-opt` 那一半」，`passes.cc` 对应「`libtriton.so` 那一半」。
- **新 pass 有两个注册点。** 一个 pass 想同时被 JIT 和 `triton-opt` 使用，要同时改 `passes.cc`（加 `add_*` 绑定，并在 `compiler.py` 的 `make_ttir` / `make_ttgir` 里调用）和 `triton-opt` 的注册代码。本仓库在 C 里做了后者，前者只写了 `add_annotate_dot_stages`，没有装回 venv。
- **后端是 Python 插件。** 已安装的 `triton/backends/__init__.py` 扫描 `backends/<name>/compiler.py` 和 `driver.py`，各找一个具体子类（`BaseBackend`、`DriverBase`）。[`BaseBackend`](../third_party/triton/python/triton/backends/compiler.py) 要实现 `supports_target`、`hash`、`parse_options`、`add_stages`、`load_dialects`。`add_stages` 往 `stages` 字典里填 `ir_name → (src, metadata) -> str|bytes`，最后一个 stage 返回 `bytes` 交给 launcher。自定义芯片公司的前端通常保留 Python，把 `add_stages` 里的 `make_ttgir` / `make_llir` 换成自己的 pass 序列，并提供自己的 driver。
- **自有芯片才替换两段 conversion。** NVIDIA 上是在 `TritonToTritonGPUPass.cpp` 和 `TritonGPUToLLVM.cpp` 的前后插入 pass，不重写它们；自有芯片才换成自己的 GPU dialect 和 lowering。

### 9.2 逐段对照

| 问题 | 工业界 | 本仓库 |
|---|---|---|
| 自定义 pass 是不是自己编 `tt-opt` 并 `registerPass` | 是。`triton-opt` 通过 `MlirOptMain`，JIT 通过 `passes.cc` | `compiler/tt-opt.cpp` 的四次 `registerPass`。JIT 绑定只有 `add_annotate_dot_stages`，且不在已安装的 `.so` 里 |
| 融合是不是只能 JIT | 否。epilogue 融合是 TTIR 上的 rewrite，在 layout 分配之前 | `--triton-fuse-dot-epilogue` 生成 `tt.fused_dot_mul`。`ops.swiglu` 的单 kernel 融合是另一条路：手写 `@triton.jit` |
| 自定义 tiling，不用 autotune / heuristics | 编译器里一张按架构查的表，或 cost model，一次算出 `BLOCK_*` | `--triton-tile-dot` 的 `gemmTileForSm`，Python 侧同一张表在 `arch/tiling.py`，有同步测试 |
| AST → TTIR 要不要改 | 只有新语法或新 intrinsic 才改 `CodeGenerator.visit`。已有 `tt.dot` / `arith` 的融合和切分不经过前端 | 第 3 节用上游 `ast_to_ttir` 把每条 Python 构造对到 op 并锁住。不改 `code_generator.py`，因为 JIT 加载的是 A 的 `triton/*.py` |
| TTIR → TTGIR / 自有 GPU IR | NVIDIA：在 `--convert-triton-to-tritongpu` 前后插入 pass，layout 仍用上游 `AccelerateMatmul`。自有芯片：换成自己的 GPU dialect 和 conversion | 上游这一段用 `python -m triton_llm.tt_opt --make-ttgir`。`build/bin/tt-opt` 不注册 `triton_gpu`。自有 op 的例子是 `tt.fused_dot_mul`，lower 回 `tt.dot` 后仍能走上游 conversion |
| TTGIR → LLVM IR / PTX | NVIDIA：在 `TritonGPUToLLVM.cpp` 前后插入 pass，PTX 仍是 LLVM NVPTX。自有芯片：自己的 LLVM backend 或另一套指令选择 | 已安装 Triton 的 `make_llir` / `make_ptx`。本仓库不重写这段 |
| 后端插件（`BaseBackend` + `DriverBase`） | 自定义芯片必须写 | 本仓库没有做，使用的是 wheel 里 NVIDIA 后端 |
| `lit` 管理 `.mlir` 测试 | 是 | `run_tt_opt_tests.sh` 读同样的 `// RUN:` 行，没有引入 `lit` |

## 10. 我改了 X，谁会看到

| 改动 | JIT (A) | `python -m triton_llm.tt_opt` (D) | `build/bin/tt-opt` (C) | 需要做的事 |
|---|---|---|---|---|
| `python/triton_llm/ops/*.py` 里的 kernel | 是，下次 launch（源码 hash 变） | 否 | 否 | 无 |
| `arch/tiling.py` 的候选表 | 是，`tl.constexpr` 变，重新编译 | 否 | 否 | 同步 `FuseAndTileDot.cpp`，`test_tile_table_sync.py` 会检查 |
| `FuseAndTileDot.cpp`、`AnnotateDotStages.cpp`、`Passes.td` | 否 | 否 | 是 | `cmake --build build --target tt-opt` |
| `TritonOps.td`（例如 `tt.fused_dot_mul`） | 否 | 否 | 是 | 同上，且 A 不认识新 op |
| `compiler/tt-opt.cpp` 的注册 | 否 | 否 | 是 | 同上 |
| B 里其它 `.cpp` / `.td` | 否 | 否 | 否，除非被 CMake 列出 | 要生效必须重编 Triton |
| B 里的 `compiler.py`、`code_generator.py` | 否 | 否 | 否 | 这些文件不被任何进程加载 |
| A 里的 `triton/*.py`（venv） | 是，且 `triton_key()` 变，缓存失效 | 否，D 只用 `libtriton` 里的 pass | 否 | 不建议改 |
| 重编并替换 `libtriton.so` | 是，缓存全部失效 | 是 | 否 | 才能让 `passes.ttir.add_annotate_dot_stages` 出现 |
| `pipeline.py` 的 flag 表 | 否 | 是 | 否 | 无 |

## 11. 调试、测试与排错

### 调试开关

```bash
# 显式 pipeline，与 --pass-a --pass-b 等价
./build/bin/tt-opt tests/tt/fuse_then_tile.mlir \
  --pass-pipeline='builtin.module(triton-fuse-dot-epilogue,triton-lower-fused-dot-mul,triton-tile-dot)'

# 每个 pass 之后的 IR。MlirOptMain 自带，不用改 tt-opt.cpp
./build/bin/tt-opt tests/tt/fuse_dot_epilogue.mlir --triton-fuse-dot-epilogue --mlir-print-ir-after-all

# 已安装 Triton：每个 pass 前后的 IR，写到标准错误
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 PYTHONPATH=python python scripts/dump_triton_ir.py
```

### 测试

| 命令 | 内容 |
|---|---|
| `./tests/shell/run_tt_opt_tests.sh` | `tests/tt/*.mlir` 的 FileCheck 和 `-verify-diagnostics` |
| `pytest tests/python/test_frontend_middle.py` | 前端 op 对照、rewrite-tensor-pointer、blocked / MMA v1 / dot_op |
| `pytest tests/python/test_python_cpp_handoff.py` | Python 抓 TTIR → C++ `tt-opt` |
| `pytest tests/python/test_tile_table_sync.py` | C++ 与 Python 的 tile 表一致 |
| `ctest --test-dir build` | 上面全部，加 CUTLASS 和数值测试 |

### 常见报错

| 现象 | 原因 |
|---|---|
| `Dialect 'llvm' not found for custom op 'llvm.mlir.undef'`（`tt-opt`） | `main` 没有注册 `LLVMDialect`。现已注册 |
| `Dialect 'triton_gpu' not found`（`tt-opt`） | `tt-opt` 不带 `triton_gpu`。TTGIR 用 `python -m triton_llm.tt_opt` |
| `unknown pass --triton-tile-dot`（Python 驱动） | 手写 pass 只在 `tt-opt`。测试的 `// RUN:` 要写 `TT_OPT_CPP` |
| `AttributeError: ... add_annotate_dot_stages`（`passes.ttir`） | 已安装的 `.so` 没有这个绑定，见 7.5 |
| `launch did not compile` (`capture_frontend_ttir`) | JIT 命中了缓存。该函数已清内存缓存并重置 binder，仍失败说明 `launch` 没有真正调用 kernel |
| 改了 `code_generator.py` 但行为没变 | 改的是 B，JIT 加载的是 A |
| 缓存导致 pass 没跑 | 设 `TRITON_ALWAYS_COMPILE=1` |

## 12. 前中端知识点清单，以及本仓库没有覆盖的部分

覆盖并有测试锁住的：

| 知识点 | 位置 |
|---|---|
| AST → TTIR 的 op 映射、constexpr、specialization、`do_not_specialize` | 3.3；`test_frontend_middle.py` |
| `tt.make_tensor_ptr` 被改写成普通 load | `test_frontend_middle.py` |
| `addptr` 折叠、splat 下沉 | `combine_addptr.mlir`、`reorder_broadcast.mlir` |
| blocked / MMA / dot_op layout | 5.2；`test_frontend_middle.py` |
| 自定义 op、verifier、负例 | `TT_FusedDotMulOp`；`fused_dot_mul_verify.mlir` |
| RewritePattern + greedy driver（融合）、循环重写（tile） | `FuseAndTileDot.cpp`；`fuse_*.mlir`、`tile_dot*.mlir` |
| pass 选项（`sm`）、Passes.td、`registerPass` | 7.2；`annotate_dot_stages*.mlir` |
| Python 与 C++ 的交接、缓存 key | 2；`test_python_cpp_handoff.py` |
| 两份 tile 表不漂移 | `test_tile_table_sync.py` |

只在文档里解释、本仓库没有实现或没有单独测试的：

- **DialectConversion**（`ConversionTarget`、`TypeConverter`）：`TritonToTritonGPUPass.cpp` 用它，本仓库没有写类似的 conversion，因为 `build/bin/tt-opt` 不含 `triton_gpu`。
- **Software pipeline、fence、TMA**：源码在 5.1，V100 上不会执行。
- **后端插件**：`BaseBackend` / `DriverBase` 的自定义实现。
- **上游 `triton-opt` 和 `lit`**：源码在 `third_party/triton/bin/` 和 `test/lit.cfg.py`，本仓库不编译、不运行。行为见 9.1。本仓库的测试用 shell 脚本读同样的 `// RUN:` 行。
- **Triton 解释器**（`TRITON_INTERPRET=1`）：行为见 2.6，本仓库没有测试。
- **把手写 pass 装回 venv 的 JIT**：需要重编 Triton，见 7.5。
