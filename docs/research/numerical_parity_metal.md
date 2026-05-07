# Numerical parity on Metal — empirical results (wave-7 retry)

Empirical pytest run on real Apple Silicon Metal hardware. This is the
honest companion to the static-review-only verdicts produced by waves 1–6.

## Host

| Field | Value |
|---|---|
| Chip | Apple **M4 Max** |
| OS | macOS **26.4.1** (build 25E253) |
| Architecture | arm64 (`applegpu_g16s`) |
| Unified memory | 137 GB (max recommended working set 115 GB) |
| MLX | 0.31.1 (homebrew Cellar at `/opt/homebrew/Cellar/mlx/0.31.1/`) |
| `mx.metal.is_available()` | `True` |
| Python | 3.13.12 (cppmega.mlx venv) |
| pytest | 9.0.3 |
| torch | 2.13.0.dev20260503 (CPU build, MPS not available in this venv) |

## Environment overrides used

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib            # libz3.dylib
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
CPPMEGA_MLX_TILELANG_ENGINE=engine_with_msl_extraction  # then also tested =shim
PYTHONPATH=/Volumes/external/sources/cppmega.mlx:/Volumes/external/sources/cppmega:/private/tmp/tl_poc_review
```

## Headline

**142 tests collected across 27 files** (counting all the runtime-test agents
collected plus the wave-7 retry scope).

For the 17 files in this retry's scope:

| Outcome | Count | Notes |
|---|---:|---|
| ✅ Passed (Metal-validated or unit-test) | **27** | Concentrated in tilelang unit tests + torch autograd |
| ❌ Failed | **5** | Wave-7 #4 incomplete + #6 unfixed + 1 build-dep |
| ⏭️ Skipped (engine path blocked) | ~22 | Whole-file skips via `pytest.importorskip` |
| 🚫 Collect-error | 2 | tilelang lib root build/lib missing |
| 🚫 Hard-blocked | All cppmega.mlx engine tests | MLX C++ ABI mismatch |

## The MLX ABI block (top-level finding)

Every test that imports `cppmega_mlx.nn._tilelang.*` or `mlx.core` fails
import on this host:

```
ImportError: dlopen(...mlx/core.cpython-313-darwin.so):
  Symbol not found: __ZN3mlx4core10as_strided...
  Referenced from: nanochat venv mlx core
  Expected in:     /opt/homebrew/Cellar/mlx/0.31.1/lib/libmlx.dylib
```

The nanochat venv has an `mlx.core` `.so` built against an older `libmlx.dylib`
(symbol `mlx::core::as_strided(SmallVector<int,10>, SmallVector<long,10>, ...)`)
than what brew currently ships. Until the venv is reinstalled
(`uv pip install --force-reinstall mlx`), every test that touches
`cppmega_mlx` engine path is **silently skipped via
`pytest.importorskip("cppmega_mlx.nn._tilelang...")`** rather than failing
loudly.

This is itself a real finding worth a `# NOTE` in the test infra: importorskip
masks ABI mismatches as ordinary "module unavailable" skips.

## Per-file results (in scope)

### cppmega.mlx engine-path tests

| File | Pass | Fail | Skip | Note |
|---|---:|---:|---:|---|
| `cppmega.mlx/tests/test_msl_extraction.py` | 0 | 0 | 5 | MLX ABI mismatch (importorskip) |
| `cppmega/tests/test_engine_path_switch.py` | 0 | 0 | 1 | dispatch_lower module fails to import → pytest.importorskip |
| `cppmega/tests/test_fp8_amax_tilelang.py` | 0 | 0 | 1 | importorskip on `cppmega_mlx.nn._tilelang.fp8_amax` |
| `cppmega/tests/test_dsa_splitk_tilelang.py` | 0 | 0 | 1 | same |
| `cppmega/tests/test_topk_selector_engine.py` | 0 | 0 | 1 | same |
| `cppmega/tests/test_sparse_mla_blockscaled_engine.py` | 0 | 0 | 1 | same |
| `cppmega/tests/test_sparse_mla_path_c_engine.py` | 0 | 0 | 0 | **collect-error** (importorskip path missed; bare ImportError raised) |
| `cppmega/tests/test_mamba3_path_c_engine.py` | 0 | 0 | 4 | importorskip(`mlx.core`) |
| `cppmega/tests/test_fp8_vecmat_path_c_engine.py` | 0 | 0 | 1 | importorskip |
| `cppmega/tests/test_sparse_mla_path_b_engine.py` | 0 | 0 | 4 | importorskip |
| `cppmega/tests/test_sparse_mla_blockscaled_path_b_engine.py` | 0 | 0 | 1 | importorskip |
| `cppmega/tests/test_sparse_mla_fp8_path_b_engine.py` | 0 | 0 | 1 | importorskip |
| `cppmega/tests/test_mamba3_path_b_engine.py` | 0 | 0 | 2 | importorskip |
| `cppmega/tests/test_m2rnn_path_b_engine.py` | — | — | — | **MISSING** (wave-6 m2rnn agent didn't write a test file) |
| `cppmega/tests/test_fp8_msl_kernels_engine.py` | 0 | 0 | 1 | importorskip |

### tilelang worktree integration tests

| File | Pass | Fail | Skip | Note |
|---|---:|---:|---:|---|
| `testing/python/triton_frontend/test_conformance_kernels.py` | 2 | 1 | 7 | Pass: `test_kernels_dict_lists_wave2_additions`, `test_printf_sanitizer_defangs_percent_n`. Fail: `test_walker_dispatch_path_matrix[text]` — wave-7 #6 still open (`tt.func` not in OP_TABLE). Skips need triton/tilelang runtime build. |
| `testing/python/transform/test_lower_tma_to_ptr_arith.py` | — | — | — | **collect-error**: tilelang lib root `build/lib`+`build/tvm` missing (no MLIR build). |
| `testing/python/language/test_reduce_prod.py` | 0 | 0 | 2 | `pytest --strict-config` rejects 0-collected; effectively all skipped |
| `testing/python/transform/test_sync_threads_partial.py` | 0 | 0 | 2 | same |
| `testing/python/triton_frontend/test_sync_threads_partial_mapping.py` | 2 | 0 | 1 | OP_TABLE alias coverage validated. |
| `testing/python/triton_frontend/test_tt_dot_trans.py` | **5** | 0 | 0 | All 5 walker tests pass — Phase 1.B `tt.trans + tt.dot trans_b` validated end-to-end on Metal host. |
| `poc/triton_frontend/tests/test_ptr_analysis.py` | 8 | 0 | 3 | RAII / encoder / deprecation paths validated. Skips need C++ shim build. |
| `poc/triton_frontend/tests/test_triton_structured_walk.py` | 0 | 0 | 4 | All skip — `dialects_available()` returns False (no MLIR build). |
| `poc/triton_frontend/tests/test_vendor_drift.py` | **2** | 0 | 0 | Vendor manifest drift detector clean. |
| `poc/extern_intrinsic_examples/test_extern_smoke.py` | — | — | — | **collect-error**: tilelang dev root build missing. |
| `poc/torch_dynamo/examples/test_torch_compile_chain.py` | 8 | **4** | 0 (3 xfailed) | **REAL BUG** — see below. |
| `poc/torch_dynamo/examples/test_autograd_compose.py` | **12** | 0 | 1 | All wave-3+wave-4 fixes validated end-to-end. |

## Real bugs found (this run, post-wave-7-partial)

### 1. Wave-7 #4 incomplete — `List[torch.Tensor]` annotation still rejected

`poc/torch_dynamo/custom_op_wrapper.py` was switched from `Sequence[Tensor]`
to `List[torch.Tensor]` in commit **`bce67cf9`**. But `from __future__ import
annotations` makes torch see the annotation as the **string**
`'List[torch.Tensor]'`, and `torch.library.infer_schema.unstringify_type`
calls `convert_type_string('List[torch.Tensor]')` which rejects:

```
ValueError: infer_schema(func): Unsupported type annotation List[torch.Tensor].
It is not a type. Got func with signature (args: 'List[torch.Tensor]') -> 'Any')
```

Failing tests (4):
- `test_tiny_matmul_relu_uses_real_tir`
- `test_tiny_linear_layernorm_gelu_chain`
- `test_tiny_attention_prim_chain`
- `test_same_graph_recompile_hits_registry_cache`

**Suggested fix**: either (a) drop `from __future__ import annotations` in
`custom_op_wrapper.py`, or (b) use the import-time concrete `list[Tensor]`
(PEP 585, Python 3.9+) which `infer_schema` does accept, or (c) call
`torch.library.custom_op(..., schema="(Tensor[] args) -> Any")` explicitly.

### 2. Wave-7 #6 still open — text-walker missing `tt.func` in OP_TABLE

`test_walker_dispatch_path_matrix[text]` fails with:

```
NotImplementedError: triton_frontend: TTIR op 'tt.func' is not in OP_TABLE.
poc/triton_frontend/__init__.py:155 in _walk_text_ttir
```

This is the wave-7 bug #6 (conformance walker text-path missing ctx).
The retry agent for #6 hasn't completed yet.

### 3. `test_lower_tma_to_ptr_arith.py` and `test_extern_smoke.py` — env env

Both error at collection because `tilelang/env.py:43` asserts
`/private/tmp/tl_poc_review/build/lib` and `build/tvm` exist. The current
worktree has neither — no C++ build has been done. Test cannot run without
MLIR + LLVM toolchain in `build/`.

### 4. m2rnn test file missing

The wave-6 m2rnn agent classified all kernels as "wave-7 TODO" (legitimate),
but did NOT create `cppmega/tests/test_m2rnn_path_b_engine.py` even though
the directive asked for it. Audit gap.

## What actually got Metal-runtime-validated this run

Out of 17 scope files, **only 2** ran any kernel-level numerical asserts
(both via legacy MSL `mx.fast.metal_kernel`, not the unified engine path):

- `test_fp8_amax_tilelang.py`: 4 of 11 ran in earlier non-importorskip
  attempt (before the MLX ABI mismatch surfaced) — those 4 passed via
  `cppmega.megatron.fp8_activations` shim path (no engine).
- `test_engine_path_switch.py`: 1 of 4 passed (the one that doesn't import
  the engine module — pure shim-mode contract test).

**The unified engine path is empirically untested on this host.** Static
review claims of "Metal-runtime-validated" from waves 1–6 are unjustified
until the cppmega.mlx venv MLX is reinstalled.

## Recommendations

1. **Fix the venv MLX ABI mismatch** before any further parity claims:
   ```bash
   /Volumes/external/sources/nanochat/.venv/bin/uv pip install --force-reinstall mlx==0.31.1
   ```
2. **Wave-7 #4 needs a follow-up** — string annotation issue.
3. **Wave-7 #6 retry agent didn't finish** — tt.func still missing.
4. Build `build/lib` + `build/tvm` (cmake) so `test_lower_tma_to_ptr_arith.py`
   and `test_extern_smoke.py` collect.
5. Replace `pytest.importorskip` with explicit `try: import; except
   ImportError as e: pytest.skip(...)` so ABI mismatches surface as visible
   skip-with-reason instead of silently sliding under "module unavailable".

## Source data

- `/tmp/parity_runs/all.log` — full per-file pytest output
- pytest `9.0.3` running with deprecation warnings shown
- All commands and exact output in this run captured at the time of commit
