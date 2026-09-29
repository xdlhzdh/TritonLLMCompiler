# Acceptance matrix

| Gate | Mechanism | Target |
|------|-----------|--------|
| Triton FP16 atol<1e-3 rtol<1e-2 | pytest, also `ctest -R triton_ops_pytest` (option default ON) | hard |
| CUTLASS numeric | ctest | hard |
| tt-opt FileCheck and verifier negatives | `tests/shell/run_tt_opt_tests.sh` | hard |
| Front-end op mapping, layouts, Python→C++ tt-opt handoff | pytest `test_frontend_middle.py`, `test_python_cpp_handoff.py` | hard |
| C++ and Python tile tables agree (sm 70..90) | pytest `test_tile_table_sync.py` | hard |
| Tiling legality | pytest `test_tiling_config.py`; `ops.swiglu` launches `arch.tiling.choose_gemm_tile` | hard |
| Sm90 compile; cc<90 SKIP; cc>=90 checks `silu(X@W1)*(X@W3)` | ctest | hard |
| Bandwidth 85% / TC 60% | `scripts/profile_ncu.sh` only checks that `ncu` exists and prints a notice. It collects no metrics, so these targets are not measured | not enforced |
