"""Run Triton compiler passes on one MLIR file.

``python -m triton_llm.tt_opt file.mlir --triton-combine`` parses the file
with ``ir.parse_mlir_module`` and runs ``createCombineOpsPass`` from the
installed Triton 3.1 ``libtriton.so``. Pass order is the order of flags.

``--make-ttir`` / ``--make-ttgir`` expand to the pipeline in
``third_party/triton/third_party/nvidia/backend/compiler.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_llm.compiler.pipeline import BY_FLAG, PassOptions, expand_flags, format_catalog

_OPTION_FLAGS = {
    "--sm": 1,
    "--num-warps": 1,
    "--num-stages": 1,
    "--num-ctas": 1,
}
_META_FLAGS = {"--list"}


def _flag_name(token: str) -> str:
    return token.split("=", 1)[0]


def _sm_override(token: str) -> int | None:
    if "=" not in token:
        return None
    payload = token.split("=", 1)[1]
    for part in payload.split(","):
        if part.startswith("sm="):
            return int(part.split("=", 1)[1])
    return None


def _pass_flags(argv: list[str]) -> list[str]:
    flags = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _OPTION_FLAGS:
            i += 1 + _OPTION_FLAGS[token]
            continue
        name = _flag_name(token)
        if token in _META_FLAGS or name in BY_FLAG or name in {
            "--make-ttir",
            "--make-ttgir",
        }:
            if token not in _META_FLAGS:
                flags.append(token)
            i += 1
            continue
        i += 1
    return flags


def _load(path: Path):
    from triton._C.libtriton import ir, nvidia

    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    return module, context


def run_file(path: Path, flags: list[str], options: PassOptions) -> str:
    module, context = _load(path)
    for raw in expand_flags(flags, options.sm):
        flag = _flag_name(raw)
        step = BY_FLAG.get(flag)
        if step is None:
            known = ", ".join(sorted(BY_FLAG))
            raise SystemExit(f"unknown pass {raw}. known: {known}, --make-ttir, --make-ttgir")
        sm = _sm_override(raw)
        opts = options if sm is None else PassOptions(sm, options.num_warps, options.num_stages, options.num_ctas)
        try:
            step.run(module, context, opts)
        except Exception as exc:
            raise SystemExit(f"{raw} failed: {exc}") from exc
    return module.str()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="tt-opt", description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="TTIR or TTGIR file")
    parser.add_argument("--sm", type=int, default=70)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--num-ctas", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="print pass flags and the make-ttir/make-ttgir order")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        bad = [token for token in unknown if token.startswith("-") and _flag_name(token) not in BY_FLAG and _flag_name(token) not in {
            "--make-ttir",
            "--make-ttgir",
        }]
        # Pass flags are unknown to argparse; anything else is an error.
        stray = [token for token in unknown if not token.startswith("--")]
        if bad or stray:
            parser.error("unrecognized arguments: " + " ".join(bad + stray))
    options = PassOptions(args.sm, args.num_warps, args.num_stages, args.num_ctas)
    if args.list:
        print(format_catalog(options.sm))
        return 0
    if args.input is None:
        parser.error("input file is required")
    flags = _pass_flags(argv)
    if not flags:
        parser.error("name at least one pass, or pass --make-ttir / --make-ttgir")
    sys.stdout.write(run_file(args.input, flags, options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
