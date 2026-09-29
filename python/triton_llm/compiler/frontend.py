"""TTIR at the AST boundary, before ``make_ttir``.

``CompiledKernel.asm["ttir"]`` is the module after ``CUDABackend.make_ttir``.
The text returned here is ``ASTSource.make_ir``, which is ``ast_to_ttir``.
"""
from __future__ import annotations

from collections.abc import Callable


def capture_frontend_ttir(jit_fn, launch: Callable[[], None]) -> str:
    """Run ``launch`` and return the TTIR emitted by ``CodeGenerator.visit``.

    ``JITFunction.create_binder`` copies ``triton.compiler.compile`` onto the
    function the first time it runs, so the in-memory cache and the binder are
    reset here. Otherwise a second capture in the same process never reaches
    the patched ``compile``. The disk cache is not consulted: ``make_ir`` is
    called before the original ``compile``.
    """
    import triton.compiler as compiler_pkg
    from triton.compiler import compiler as compiler_mod
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    captured: dict[str, str] = {}
    original = compiler_mod.compile

    def wrapped(src, target=None, options=None):
        if target is None:
            target = driver.active.get_current_target()
        backend = make_backend(target)
        parsed = backend.parse_options(dict(options or {}))
        from triton._C.libtriton import ir

        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        module = src.make_ir(parsed, backend.get_codegen_implementation(), context)
        captured["ttir"] = module.str() if hasattr(module, "str") else str(module)
        return original(src, target=target, options=options)

    jit_fn.cache.clear()
    jit_fn.binder = None
    compiler_mod.compile = wrapped
    compiler_pkg.compile = wrapped
    try:
        launch()
    finally:
        compiler_mod.compile = original
        compiler_pkg.compile = original
        jit_fn.binder = None
    if "ttir" not in captured:
        raise RuntimeError("launch did not compile")
    return captured["ttir"]
