#!/usr/bin/env bash
# Build Triton 3.1.0 from third_party/triton plus the LLVM 19 frontend patch,
# and point this repo's .venv at that build.
#
# The kernel path is this libtriton.so: DSL create_* , make_ttir, make_ttgir,
# make_llir, make_ptx, make_cubin. triton-opt from the same cmake is copied to
# build/bin/triton-opt. This does not modify /home/xdlhzdh/Codes/TritonQAttn/.venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/build/triton-src"
PY="${ROOT}/.py-triton"
PATCHES=(
  "${ROOT}/third_party/patches/0001-frontend-llvm19.patch"
  "${ROOT}/third_party/patches/0002-ttir-passes-llvm19.patch"
)
BASE_SITE="/home/xdlhzdh/Codes/TritonQAttn/.venv/lib/python3.12/site-packages"
STAMP="${SRC}/.frontend-stamp"

if [[ ! -f "${ROOT}/third_party/triton/include/triton/Dialect/Triton/IR/TritonOps.td" ]]; then
  git -C "${ROOT}" submodule update --init third_party/triton
fi

STAMP_HASH="$(cat "${PATCHES[@]}" | sha256sum | awk '{print $1}')"
need_copy=1
if [[ -f "${STAMP}" ]] && [[ "$(cat "${STAMP}")" == "${STAMP_HASH}" ]]; then
  need_copy=0
fi
if [[ "${need_copy}" -eq 1 ]]; then
  rm -rf "${SRC}"
  mkdir -p "${SRC}"
  cp -a "${ROOT}/third_party/triton/." "${SRC}/"
  rm -rf "${SRC}/.git"
  for _patch in "${PATCHES[@]}"; do
    patch -p1 --forward --batch --directory "${SRC}" --input "${_patch}"
  done
  python3 - "${SRC}/python/setup.py" << 'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = text.replace('"-DLLVM_ENABLE_WERROR=ON"', '"-DLLVM_ENABLE_WERROR=OFF"')
old = '"-DTRITON_BUILD_TUTORIALS=OFF",'
new = '"-DTRITON_BUILD_TUTORIALS=OFF", "-DTRITON_BUILD_UT=OFF",'
if old not in text:
    raise SystemExit("setup.py cmake args not found")
text = text.replace(old, new, 1)
old = '    packages += ["triton/profiler"]\n'
new = '    if check_env_flag("TRITON_BUILD_PROTON", "ON"):\n        packages += ["triton/profiler"]\n'
if old not in text:
    raise SystemExit("profiler package line not found")
path.write_text(text.replace(old, new, 1))
PY
  printf '%s\n' "${STAMP_HASH}" > "${STAMP}"
fi

if [[ ! -x "${PY}/bin/python" ]]; then
  /usr/bin/python3.12 -m venv "${PY}"
  site="$("${PY}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "${BASE_SITE}" > "${site}/torch-from-tritonqattn.pth"
fi

# Ubuntu's /usr/lib/python3.12/sitecustomize.py occupies that name, so this
# hook lives in a .pth. It runs after the editable installer's .pth and moves
# the built triton finder ahead of PathFinder. Otherwise PathFinder selects
# the triton package inside the torch site-packages directory.
site="$("${PY}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
# A .pth file executes only lines that start with "import".
printf '%s\n' 'import sys, __editable___triton_3_1_0_finder as _f; _f.install(); sys.meta_path.remove(_f._EditableFinder); sys.meta_path.insert(0, _f._EditableFinder)' > "${site}/zz-prefer-built-triton.pth"

export TRITON_BUILD_PROTON=OFF
export MAX_JOBS="$(nproc)"
export PATH="/home/xdlhzdh/.local/bin:/usr/bin:${PATH}"
"${PY}/bin/pip" install -e "${SRC}/python" --no-build-isolation --no-deps

opt="$(find "${SRC}/python/build" -type f -name triton-opt -perm -111 | head -n 1)"
if [[ -z "${opt}" ]]; then
  echo "triton-opt was not produced" >&2
  exit 1
fi
mkdir -p "${ROOT}/build/bin"
cp -a "${opt}" "${ROOT}/build/bin/triton-opt"

# This repo's .venv is a symlink. Retarget it at the new interpreter.
# The TritonQAttn tree is left as it was.
ln -sfn "${PY}" "${ROOT}/.venv"

"${PY}/bin/python" - << 'PY'
import os, triton
so = os.path.realpath(triton._C.libtriton.__file__)
src = os.path.realpath(triton.__file__)
print("triton", src)
print("libtriton.so", so)
if "TritonQAttn" in so or "TritonQAttn" in src:
    raise SystemExit("still importing TritonQAttn's triton")
if not hasattr(triton.language, "chip_rcp"):
    raise SystemExit("tl.chip_rcp missing from the built language package")
PY

echo "build_libtriton: OK"
echo "python: ${ROOT}/.venv/bin/python"
echo "triton-opt: ${ROOT}/build/bin/triton-opt"
