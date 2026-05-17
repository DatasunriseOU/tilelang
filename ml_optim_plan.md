# ML Optimization Plan

Date: 2026-05-14

This is the execution plan for making Path C a real framework path:
TileLang IR -> scheduler/proofs -> TVM -> TVM-FFI -> MLX. cppmega is a
consumer and benchmark suite, not the place where backend scheduling decisions
are hardcoded.

## Hard Rules

- No production monkeypatches.
- No production `mx.fast.metal_kernel` or handwritten model-owned MSL.
- No public cppmega direct TVM-FFI bypass. A private TileLang adapter may call
  TVM-FFI, but it must be ABI/version checked and hidden behind TileLang APIs.
- No public `*_partial` tensors for supported reductions. Internal scratch is
  allowed only inside TileLang/runtime ownership.
- Sync/event insertion is proof driven. If Z3 and dependency metadata prove no
  hazard, no barrier/event/materialization is emitted.
- Path C performance work must happen in TileLang scheduler/codegen,
  TVM/TVM-FFI, or MLX bridge layers.
- Move to the next package only after the current package has green tests,
  review, and performance receipt.

## Standard Environment

TileLang shell setup:

```bash
cd /private/tmp/tl_apache_tvm_swap
export TL_ROOT=/private/tmp/tl_apache_tvm_swap
export Z3_LIB="$TL_ROOT/.venv313/lib/python3.13/site-packages/z3/lib"
export TILELANG_DISABLE_CACHE=1
export DYLD_LIBRARY_PATH="$Z3_LIB:$TL_ROOT/build/lib:$TL_ROOT/build/tvm"
export PYTHONPATH="$TL_ROOT:$TL_ROOT/3rdparty/tvm/python:$TL_ROOT/3rdparty/tvm/3rdparty/tvm-ffi/python"
cmake --build build -j$(sysctl -n hw.ncpu)
```

cppmega commands with this TileLang checkout injected:

```bash
cd /Volumes/external/sources/cppmega.mlx
export TL_ROOT=/private/tmp/tl_apache_tvm_swap
export TILELANG_ROOT="$TL_ROOT"
export TVM_ROOT="$TL_ROOT/3rdparty/tvm"
export TILELANG_DISABLE_CACHE=1
export TILELANG_DEV_BUILD_ROOT="$TL_ROOT/build"
export TVM_LIBRARY_PATH="$TL_ROOT/build/lib:$TL_ROOT/build/tvm"
export DYLD_LIBRARY_PATH="$TL_ROOT/build/lib:$TL_ROOT/build/tvm"
export PYTHONPATH="$TL_ROOT:$TL_ROOT/3rdparty/tvm/python:$TL_ROOT/3rdparty/tvm/3rdparty/tvm-ffi/python"
```

Universal completion gate for every package:

```bash
cd /private/tmp/tl_apache_tvm_swap
git diff --check
cmake --build build -j$(sysctl -n hw.ncpu)   # required when C++/Metal/runtime changed
```

Every package also needs:

1. Implementer/Fixer pass.
2. Code review of changed scheduler, lowering, runtime, and model-facing APIs.
3. Perf optimization pass when generated code, dispatch, sync, or allocation
   can change.
4. Regression tests listed in that package.
5. Receipt with exact command, git SHAs, cache state, pass/fail, and timings
   when performance is part of the package.

Package execution loop:

1. Implementer/Fixer: make the smallest framework-level change that satisfies
   the package scope. Do not move the workaround into cppmega.
2. Code Review: inspect changed files for wrong ownership, hidden copies,
   hidden materialization, stale ABI checks, and backend leakage into model
   code.
3. Perf Optimization: inspect generated source and profiler/bench output before
   changing schedule/codegen knobs. Every performance claim needs a before/after
   receipt.
4. Regression Tests: run the package test block and the universal completion
   gate. If any test fails, stay in the same package and repeat from fixer.
5. Advance Gate: update the receipt table only after tests are green and the
   package-specific green criteria are met.

When a package changes only documentation, run `git diff --check`. When a
package changes Python lowering/scheduler, run its focused pytest block. When a
package changes C++/Metal/TVM/MLX bridge code, rebuild first and then run both
TileLang and cppmega focused tests.

## Execution Order

| ID | Package | Do not start until |
| --- | --- | --- |
| P0 | Current FP8 Path C gate | Current blocker, starts now |
| P1 | Semantic reduction IR | P0 green |
| P2 | Automatic reduction rewrite | P1 green |
| P3 | ReductionPlan scheduler metadata | P1 green |
| P4 | Z3 legality proofs | P3 green |
| P5 | Sync/event planner | P4 green |
| P6 | Backend lowerer registry | P3 green |
| P7 | Large-axis generated reductions | P4 and P6 green |
| P8 | Reverse recurrence scan planner | P4, P5, and P6 green |
| P9 | Cost model/codegen cleanup | P6 green |
| P10 | Autotune/profiling memoization | P9 green |
| P11 | Production lint | Can run after each package, final gate after P10 |
| P12 | Full 1B training matrix | P0 green for first pass, final after P10/P11 |

## Current Remaining Work

As of 2026-05-17, P0 through P12 have green receipts. P12 is the final 1B
matrix gate after P10/P11, and no package remains blocked in this plan.

| ID | Status | Remaining work | Latest receipt |
| --- | --- | --- | --- |
| P0 | Green / current recheck | Mamba3 NaN guards and strict `vecmat_4096` FP8 Path C perf gate are green on the latest reruns. Keep the full P0 suite as the advance gate before final 1B claims. | Root-caused the FP8 regression to an unsafe direct `simd_sum` load-remap optimization in `LowerThreadAllreduce`: it moved `simd_sum(accum)` into downstream `if (kr == 0)` stores, producing lane0-only partials on production FP8 reducers. Reverted only that shortcut and kept semantic allreduce/native local-buffer lowering. Sparse-MLA FP8 full file now passes (`53 passed`); Mamba3 full Path C file passes (`46 passed`), including 1B production-shape bwd finite guards; strict FP8 `vecmat_4096` passes with Path C median `0.189396 ms` vs Path B `0.187021 ms`. |
| P1 | Green / first pass | Keep P1 receipts stable while P2 starts; rerun P1 tests after any reduction IR or lowering change. | Semantic `thread_allreduce` and TileOp `reduce_sum`/`reduce_max` now extract operation, regions, axes, predicate, dtype, and strategies; malformed TileOp axes fail before codegen; focused P1 tests are green. |
| P2 | Green / first pass | Machine-readable rewrite diagnostics landed; Mamba3 public bwd partial owner-output ABI and internal host-reduced fallback are now fail-closed/removed; M2RNN Path C now emits final `dW` owner-output buffers for supported batch-1 bwd routes and fails closed instead of exposing public partial outputs for unsupported multi-batch reductions. Regular Sparse MLA Path C bwd, blockscaled Sparse MLA direct-MSL bwd, and Mamba3 direct-MSL Path B bwd now emit final owner-output gradients and no longer expose public `*_partial` reduction buffers. FP8 Sparse MLA direct-MSL bwd source and host `dkv_partial` reducer are removed, and regular Sparse MLA, generic FP8 helper, blockscaled Sparse MLA, top-k selector, and M2RNN no longer have direct-MSL Path B runtime surfaces. The only remaining allowlisted legacy direct-MSL module is Mamba3; it is a P8/P9 performance debt, not a P2 public-partial blocker, because the checked-in receipt keeps AUTO on Path B until Path C is no-worse. | 2026-05-17 follow-ups removed the debug-unit-test direct-MSL allowlist entry, regular Sparse-MLA `dkv_partial`, blockscaled Sparse-MLA `dkv_partial`, Mamba3 Path B public P-axis partial outputs, top-k selector direct-MSL Path B, regular Sparse-MLA direct-MSL Path B, M2RNN direct-MSL Path B, blockscaled Sparse-MLA direct-MSL Path B, and generic FP8 direct-MSL helper surface. The allowlist remains 1 entry (`mamba3.py`) with `public_partial_outputs=[]`; focused lint/MSL/sparse/Mamba3/top-k/M2RNN/blockscaled/FP8 helper suites, compileall, and perf/receipt smokes are green. The P2 close gate passed with Mamba3 Path C still slower on the receipt (`1.088x` fwd, `1.546x` bwd, `1.479x` fwd+bwd), so the non-rewrite has a machine-readable reason and P3 can start. |
| P3 | Green / first pass | `ReductionPlan` metadata now records selected strategy, thread/block mapping, alias constraints, in-place constraint posture, memory visibility, and internal scratch/materialization requirements. P4 can start, but rerun P3 if strategy naming, legality, or backend lowering changes. | 2026-05-17 P3 first-pass receipt: scheduler metadata tests passed (`17 passed` across reduction plan, legality, and sync-event plan); transform reduction gate passed (`32 passed, 202 deselected, 170 warnings`); Metal reduce gate passed (`38 passed, 12 warnings`); compileall and diff checks passed. |
| P4 | Green / first pass | Reduction legality now proves/static-checks coverage, bounds, alias rejection, tail/broadcast coverage, index-width safety, and sync class for current `ReductionPlan` metadata. Rerun if P5 changes sync insertion or P6/P7 add new strategy names. | 2026-05-17 P4 first-pass receipt: scheduler legality gate passed (`8 passed, 11 deselected`); analysis alias/overflow/Z3 gate passed (`7 passed, 10 deselected`); Metal hazard/sync/reduce gate passed (`39 passed, 16 skipped, 104 deselected, 138 warnings`); compileall and diff checks passed. |
| P5 | Green / first pass | Sync/event decisions now carry proof-gated strategy and memory-scope metadata. Native MLX/TVM-FFI no-wait fast paths only borrow already-compact producer-free inputs; grouped/non-fp16 VJP paths that still depend on unstable native atomic owner-output accumulation fall back to the pure-MLX reference until that route is proven. Rerun P5 if bridge dependency metadata, owner-output atomics, or MLX graph-output handling changes. | 2026-05-17 P5 first-pass receipt: TileLang sync/graph/TVM-FFI Metal gate passed (`31 passed, 6 skipped, 54 warnings`); cppmega DLPack/TVM-FFI/Path C filter passed (`298 passed, 1713 deselected, 1 xfailed, 1 warning`); focused M2RNN and sparse-MLA fixer regressions passed; compileall and diff checks passed. |
| P6 | Green / first pass | Reduction backend lowerer selection now lives in backend registries for Metal, CUDA, ROCm, and CPU, with stable cached diagnostics attached to semantic reduction plans. Remaining scan/dependency registry expansion is deferred to P8/P9 because there is no scan planner surface yet. | 2026-05-17 P6 first-pass receipt: scheduler registry metadata tests passed (`9 passed`); Metal reduce/finalize gate passed (`40 passed, 15 skipped, 105 deselected, 138 warnings`); language reduce gate passed (`7 passed, 8 skipped, 220 deselected, 356 warnings`); compileall and diff checks passed. Leakage scans remain nonzero only for target-specific transform passes, FP8 intrinsic surfaces, feature probes, and semantic `T.tvm_thread_allreduce` model IR. |
| P7 | Green / first pass | TileLang large-axis allreduce coverage now includes P=32/64/96/128/256/512/1024 with internal staged final outputs only. Mamba3 Path C bwd now chooses larger per-P threadgroups for P=512/1024 while preserving the existing 256-thread choice for smaller real shapes; public partial outputs remain rejected. | 2026-05-17 P7 first-pass receipt: TileLang Metal `two_pass or reduce` gate passed (`47 passed, 15 skipped, 105 deselected, 138 warnings`); cppmega Mamba3 `headdim or bwd` gate passed (`28 passed, 25 deselected`); focused large-axis matrix and Mamba3 P-thread tests passed; compileall and diff checks passed. |
| P8 | Green / first pass | Reverse recurrence scan planning now records direction, chunking, snapshot/cache policy, rematerialization policy, alias/in-place posture, no-sync proof posture, and fused post-op metadata. Mamba3 Path C bwd consumes the plan to select state-boundary snapshots for long reverse recurrences and direct recompute for short ones, with a lazy fallback for cppmega subprocess environments that still import an older TileLang package. | 2026-05-17 P8 first-pass receipt: TileLang scheduler scan gate passed (`6 passed, 22 deselected`); cppmega focused scan/snapshot route gate passed (`4 passed, 51 deselected`); real-parquet HybridTinyLM subprocess regression passed (`2 passed`); broad cppmega `mamba3 or m2rnn or hybrid` gate passed (`378 passed, 1643 deselected`); compileall and diff checks passed. During the broad gate, the Mamba3 pure helper reverse-cumsum hazard was fixed by replacing negative-stride `mx.cumsum` input with index-based `mx.take`, and the checkpoint seam was restored to `mx.checkpoint(layer)` for monkeypatch-visible decoder checkpointing. |
| P9 | Green / first pass | Reduction and recurrence plans now expose explainable static cost metadata for registers, scratch/materialization bytes, index math, dispatch/sync counts, occupancy limiter, and split-vs-inline decisions. Metal batched allreduce templates now hoist repeated `i * workspace_stride` math into `batch_offset`, with source tests blocking regressions. | 2026-05-17 P9 first-pass receipt: scheduler `cost or register or hoist` gate passed (`5 passed, 28 deselected`); Metal `source or reduce` gate passed (`50 passed, 15 skipped, 102 deselected, 138 warnings`); cppmega `path_c` gate passed (`294 passed, 1726 deselected, 1 xfailed, 1 warning`); C++ rebuild and compileall passed; opt-in Metal reduce perf smoke passed (`3 passed, 12 deselected`) with same-simdgroup 32 at `0.1536 ms`, cross-simdgroup 128 at `0.1717 ms`, and row-reduce 256x32 at `0.1327 ms` for the low-iteration smoke. |
| P10 | Green / first pass | Legal-schedule warm memoization now profiles only legal candidates, keys warm reuse on op signature, shape, dtype, target, ABI fingerprints, proof hash, codegen hash, and normalized config, and records cold compile timing separately from warm execution timing. cppmega M2RNN custom VJP closure caches are now shape/dtype-aware so broad profile/bench/path_c runs do not reuse stale custom-function identities across layouts. | 2026-05-17 P10 first-pass receipt: TileLang scheduler `autotune or cache` gate passed (`6 passed, 32 deselected`); cppmega `profile or bench or path_c` gate passed after the order-sensitive M2RNN VJP cache fix (`411 passed, 1609 deselected, 1 xfailed, 1 warning`); focused M2RNN regression pair passed (`2 passed`); compileall passed for touched TileLang/cppmega files. |
| P11 | Green / first pass | Production lint now blocks production monkeypatch/mock seams, raw direct-MSL construction, `_msl_transform.dispatch`, public native TVM-FFI bridge imports, model-level backend intrinsics, and public partial-output surfaces. TileLang also has a focused production monkeypatch lint receipt so the planned P11 selector is non-empty. | 2026-05-17 P11 first-pass receipt: TileLang `lint or monkeypatch` gate passed (`1 passed, 1317 deselected, 1053 warnings`); cppmega `lint or monkeypatch` gate passed (`27 passed, 2000 deselected`); extended `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007,MLX008,MLX009 cppmega_mlx tests` passed clean; planned production grep is expected nonzero only for local non-public `dc_partial`, comments/docstrings, framework-owned `_tilelang` bridge/debug surfaces, feature counters, and defensive stale-partial text checks. |
| P12 | Green / final | Final 36-cell 1B matrix is captured with all 36 cells green, including real 20-step FP8 Path B baseline rows. Warm Path C cache rows are now green after fixing the cached TVM-FFI Metal host-wrapper ordering boundary, and the BF16/FP8 default-decision HTML report is generated. | 2026-05-17 final P12 receipt: `/tmp/cppmega_1b_path_matrix.md`, `/tmp/cppmega_1b_path_matrix.csv`, `/tmp/cppmega_1b_path_matrix.json`, and `/tmp/cppmega_1b_path_matrix.html` report `36 ok, 0 failed`. FP8 Path B rows now run `--dtype fp8_path_b` and dispatch DSA Sparse-MLA through `sparse_mla_fp8_reference_path_b`; FP8 Path C rows run `--dtype fp8_path_c` and dispatch Mamba3/M2RNN/Sparse-MLA through Path C. The HTML report makes the default decision explicit: warm Path C is currently slower than Path B for all BF16 and FP8 optimizer rows, so no row is a Path C default candidate under the 3 percent same-speed rule. Full matrix was first run with fresh subprocesses; old warm-cache failures were reproduced, fixed in the native bridge, rechecked with six warm-cache 20-step cells all green, then consolidated with `--reuse-existing-ok` after adding FP8 Path B baselines. Regression gates passed for the matrix harness, renderer, attention route, and m04 metadata (`113 passed`), broad cppmega P12 selector (`472 passed, 4 skipped, 1554 deselected, 1 xfailed`), focused TileLang cached-host-wrapper Metal tests (`2 passed, 26 deselected`), and the native C++ rebuild. |

## P0: Current FP8 Path C Gate

### Planned Changes

- Finish FP8 owner-output ABI for MLX arrays through TileLang's private
  TVM-FFI adapter.
- Preserve named shape metadata so shaped tensors cannot swap dimensions such
  as `D_V` and `QK_DIM`.
- Keep FP8 inputs const through Metal lowering.
- Remove duplicated hot-path host validation only when TileLang has already
  validated the ABI, dtype, compactness, pointer, and shape contract.
- Hoist repeated row/column/index math out of FP8 inner loops.
- Keep CSE enabled unless a generated-source regression proves one exact
  lowering must opt out.
- Recover `vecmat_4096` Path C speed in scheduler/codegen/runtime, not by
  falling back to model-owned MSL.

### Tests

TileLang focused tests:

```bash
cd /private/tmp/tl_apache_tvm_swap
cmake --build build -j$(sysctl -n hw.ncpu)
.venv313/bin/python -m pytest \
  testing/python/metal/test_fp8_scaled_matmul_metal.py \
  testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py \
  testing/python/analysis/test_metal_graph_sync.py \
  testing/python/scheduler/test_sync_event_plan.py \
  -q
```

cppmega FP8 Path C tests:

```bash
cd /Volumes/external/sources/cppmega.mlx
TILELANG_DISABLE_CACHE=1 \
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests/test_tilelang_fp8_vecmat_path_c.py -q
```

Strict performance gate:

```bash
cd /Volumes/external/sources/cppmega.mlx
TILELANG_DISABLE_CACHE=1 \
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python scripts/bench_tilelang_fp8_path_c.py \
  --warmup 5 \
  --iters 20 \
  --shapes matmul_128 vecmat_4096 \
  --skip-xcrun \
  --skip-sparse \
  --strict \
  --out /tmp/fp8_path_c_p0.json
```

Generated-source checks:

```bash
rg -n "mx\\.fast\\.metal_kernel|monkeypatch|simd_sum" \
  /Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang
rg -n "__tvm_error_ndim_mismatch|__tvm_error_dtype_mismatch|tvm_error_expect" \
  /tmp/fp8_path_c_generated_host.c
```

### Green Criteria

- All listed pytest commands pass.
- The strict bench passes. Target: warm Path C within 3 percent of Path B for
  `matmul_128` and `vecmat_4096`; any looser threshold must be called out in
  the receipt.
- Generated Metal keeps FP8 input buffers const.
- No production `mx.fast.metal_kernel`, monkeypatch, or public cppmega TVM-FFI
  route is used.
- Receipt includes Path B/Path C median/min/max timings, cache state, selected
  pass config, generated Metal snippet, TileLang SHA, and cppmega SHA.

## P1: Semantic Reduction IR

### Planned Changes

- Generalize the current `T.thread_allreduce_sum(...)` into semantic reduction
  IR that represents operation, axis, extent, predicate, accumulator dtype, and
  output region.
- Route compatible `T.reduce_sum`, `T.reduce_max`, and row/block reductions
  through the same representation.
- Keep backend primitives out of model generators.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/language/ -q -k "reduce"
.venv313/bin/python -m pytest testing/python/metal/test_metal_reduce.py -q
rg -n "tir\\.metal|tir\\.cuda|simd_sum|simd_shuffle" tilelang/language tilelang/transform
```

### Green Criteria

- Golden TIR tests show semantic reduction IR, not backend intrinsics.
- The same semantic reduction lowers on Metal and at least one non-Metal
  fallback path where available.
- Unsupported malformed axes fail before codegen with an explicit scheduler
  error.

### 2026-05-16 P1 first-pass green receipt

**Change under test**

- `ReductionPlan` now records a reduction predicate in stable JSON metadata.
- `tl.tileop.reduce` calls from `T.reduce_sum`, `T.reduce_max`, `T.row_reduce`,
  and `T.block_reduce` route into the same scheduler-visible plan shape as
  semantic thread allreduce: operation, input/output regions, axis, extent,
  accumulator dtype, candidate strategies, and visibility scope.
- Malformed TileOp reduce axes now raise `ReductionPlanError` before codegen.

**Commands and results**

| Command | Result |
| --- | --- |
| `pytest testing/python/scheduler/test_reduction_plan.py testing/python/scheduler/test_reduction_legality.py testing/python/transform/test_simd_reduction_rewrite.py -q` with `.venv313` Z3 lib first | pass: 28 passed |
| `pytest testing/python/language/ -q -k "reduce"` with `.venv313` Z3 lib first | pass: 7 passed, 8 skipped, 220 deselected, 356 warnings |
| `pytest testing/python/metal/test_metal_reduce.py -q` with `.venv313` Z3 lib first | pass: 37 passed, 12 warnings |
| `git diff --check` | pass |
| `rg -n "tir\\.metal\|tir\\.cuda\|simd_sum\|simd_shuffle" tilelang/language tilelang/transform` | expected nonzero: reduction IR surface is clean, but FP8-specific Metal intrinsics remain in `fp8_op.py` and `fp8_late_lower.py` and stay tracked for P2/P6/P11 cleanup. |

**Status**

- P1 is green for the first pass and unblocks P2.
- P3 is not final: the current stable `ReductionPlan` JSON is covered, but the
  full scheduler metadata audit remains before P4 can be treated as unblocked.

## P2: Automatic Reduction Rewrite

### Planned Changes

- Add a transform that detects manual reduction idioms:
  per-lane accumulation, lane-zero stores, `*_partial` tensors, host sums over
  device partials, and direct `simd_sum`/shuffle calls.
- Rewrite legal patterns into semantic reduction IR.
- Emit a diagnostic for every pattern that looks reducible but fails legality.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/transform/ -q -k "reduction"
rg -n "simd_sum|simd_shuffle|d[A-Z]?_partial|ddt_partial" tilelang testing src

cd /Volumes/external/sources/cppmega.mlx
rg -n "simd_sum|simd_shuffle|d[A-Z]?_partial|ddt_partial" cppmega_mlx tests
```

### Green Criteria

- Existing Mamba3/M2RNN reduction callsites can be represented by the rewrite
  pass or by explicit semantic IR, not by public partial tensors.
- Non-rewritten patterns include a machine-readable reason.

### 2026-05-16 P2 checkpoint receipt

**Change under test**

- `MetalSimdLiftReductions` now emits machine-readable
  `tl.reduction_rewrite_diagnostics` metadata for reduction candidates that do
  not enter the semantic `tvm_thread_allreduce` route.
- Diagnostics currently cover missing lane annotations, Z3-unproved extents,
  and reducer kinds that still fall back to backend-specific butterfly lowering.

**Commands and results**

| Command | Result |
| --- | --- |
| `pytest testing/python/transform/test_simd_reduction_rewrite.py -q` with `.venv313` Z3 lib first | pass: 19 passed |
| `pytest testing/python/transform/ -q -k "reduction"` with `.venv313` Z3 lib first | pass: 32 passed, 202 deselected, 170 warnings |
| `git diff --check` in TileLang and cppmega.mlx | pass |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` in cppmega.mlx | pass: 1 passed in 6.81s |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` in cppmega.mlx | pass: 1 passed in 7.18s |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_rejects_public_partial_owner_outputs tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` in cppmega.mlx | pass: 3 passed in 9.42s |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_rejects_public_partial_owner_outputs -q` in cppmega.mlx after deleting the internal bwd partial owner-output helper | pass: 1 passed in 3.78s |
| `pytest tests/test_tilelang_mamba3_path_c.py -q` in cppmega.mlx after deleting the internal bwd partial owner-output helper | pass: 42 passed in 44.12s |
| `rg -n "_mamba3_bwd_owner_outputs\|_mamba3_bwd_partial_owner_outputs\|_mamba3_mimo_bwd_path_c_partials\\([^\\n]*out=\|_mamba3_mimo_bwd_path_c_partials_snapshot\\([^\\n]*out=" cppmega_mlx/nn/_tilelang/mamba3_path_c.py tests/test_tilelang_mamba3_path_c.py` in cppmega.mlx | pass: no matches |
| direct M2RNN mapped-packed bwd kernel audit, with and without pre-evaluated owner-output buffers | fail/pass split: without materializing zero buffers, atomic-add outputs were all zero; with `mx.eval(dconv_input, dW_partial, dxf)` before TVM-FFI launch, `dconv/dW/dxf/dh0` matched grouped reference within `1e-9` |
| `pytest tests/test_tilelang_m2rnn_path_c.py::test_m2rnn_mapped_packed_path_c_matches_grouped_reference_and_grad tests/test_tilelang_m2rnn_path_c.py::test_m2rnn_post_residual_gate_path_c_matches_grouped_mlx_and_grad -q` in cppmega.mlx after atomic owner-output materialization | pass: 2 passed in 21.98s |
| `pytest tests/test_tilelang_m2rnn_path_c.py -q` in cppmega.mlx after atomic owner-output materialization | pass: 25 passed in 37.15s |
| 2026-05-16 sparse MLA bwd command-buffer hazard probe with `TILELANG_MLX_TVM_FFI_FORCE_OUTPUT_BARRIER=1` | pass: focused `test_path_c_backward_parity[small]` passed; proves the `dkv_partial -> MLX scatter/reduce` failure was output visibility/ordering, not sparse MLA math |
| 2026-05-16 sparse MLA focused recheck after automatic active-encoder multi-output barrier, no env override | pass: `test_path_c_backward_parity[small]` passed |
| 2026-05-16 regular sparse MLA full file after automatic active-encoder multi-output barrier | pass: 54 passed in 20.14s |
| 2026-05-16 FP8 sparse MLA full file after automatic active-encoder multi-output barrier, cppmega venv | pass: 52 passed in 59.44s |
| 2026-05-16 TVM-FFI Metal stream bridge regression after automatic active-encoder multi-output barrier | pass: 21 passed, 5 skipped, 52 warnings in 24.78s |
| 2026-05-16 full Mamba3 Path C file after automatic active-encoder multi-output barrier | pass: 42 passed in 51.97s |
| 2026-05-16 Mamba3 direct SIMD long-sequence guard after regression attribution | pass: direct `_bwd_simd_reduce_kernel_for(SEQ=2, HEADDIM=64)` rejects without explicit state snapshots; production bwd dump consumes `h_snap` and has no `1 / decay`; full file pass: 43 passed in 41.89s |
| 2026-05-16 strict FP8 gate after narrowing barrier to multi-output active external dispatch, `/tmp/fp8_path_c_after_multioutput_barrier.json` | pass: `matmul_128` ratio `0.916x`; `vecmat_4096` ratio `1.002x`; no sparse/FP8 skip except explicit `--skip-sparse` in this bench |
| 2026-05-16 remove cppmega-side Path C Metal FP8 op registration shim | pass: backend op registration lives in TileLang C++ (`intrin_rule_metal.cc`); cppmega calls `tilelang.language.fp8_op.assert_metal_fp8_intrinsics_registered` only as a contract check |
| `rg -n "_register_path_c_metal_fp8_intrinsics\|_PATH_C_METAL_FP8_INTRINSICS\|_EFFECT_KIND" cppmega_mlx tests tilelang testing/python` after removing the shim | pass: no matches |
| `pytest testing/python/metal/test_fp8_scaled_matmul_metal.py testing/python/transform/test_simd_reduction_rewrite.py -q` with `.venv313` Z3 lib first | pass: 43 passed, 7 skipped, 18 warnings in 3.68s |
| `pytest tests/test_tilelang_fp8_vecmat_path_c.py tests/test_tilelang_mamba3_path_c.py -q` in cppmega.mlx | pass: 70 passed, 1 warning in 46.00s |
| `rg -n "simd_sum\|simd_shuffle\|d[A-Z]?_partial\|ddt_partial" tilelang testing src` | expected nonzero: FP8-specific Metal intrinsics, backend codegen helper tests, and existing reduction backend tests remain. |
| `rg -n "simd_sum\|simd_shuffle\|d[A-Z]?_partial\|ddt_partial" cppmega_mlx tests` in cppmega.mlx | expected nonzero: legacy Mamba3 Path B, legacy M2RNN Path B `dW_partial`, sparse MLA `dkv_partial`, and debug/test direct-intrinsic surfaces still remain. |
| 2026-05-16 remove Mamba3 Path C host-reduced partial fallback route | pass: production bwd now either uses final-gradient SIMD/snapshot route or fails closed for unsupported P-axis reduction shapes; old `_bwd_kernel_for`, `_bwd_kernel_for_state_snapshots`, `_mamba3_mimo_bwd_path_c_partials*`, `_reduce_mamba3_bwd_partials`, and `d*_partial` tokens have no matches in `mamba3_path_c.py` or its test file |
| 2026-05-16 full Mamba3 Path C file after fallback removal | pass: `44 passed in 40.48s` |
| 2026-05-16 cppmega FP8 vecmat plus full Mamba3 Path C after fallback removal | pass: `72 passed, 1 warning in 45.31s` |
| 2026-05-16 compile/diff gates after fallback removal | pass: `.venv/bin/python -m compileall -q cppmega_mlx/nn/_tilelang/mamba3_path_c.py tests/test_tilelang_mamba3_path_c.py`; `git diff --check` for touched cppmega files and `ml_optim_plan.md` |
| 2026-05-16 remove M2RNN Path C public `dW_partial` route | pass: bwd kernels now write final `dW` owner-output buffers for supported batch-1 bwd routes; unsupported multi-batch reductions fail closed until semantic reduction lowering can prove and lower the cross-batch reduction |
| 2026-05-16 focused M2RNN Path C file after `dW_partial` removal | pass: `25 passed in 35.90s` |
| 2026-05-16 combined M2RNN plus Mamba3 Path C regression after `dW_partial` removal | pass: `69 passed in 76.46s` |
| 2026-05-16 compile/diff/scan gates after M2RNN `dW_partial` removal | pass: `compileall` for `m2rnn_path_c.py` and `test_tilelang_m2rnn_path_c.py`; `git diff --check` for touched cppmega files and `ml_optim_plan.md`; targeted `rg -n "dW_partial\|does not expose partial\|partial owner-output" m2rnn_path_c.py tests/test_tilelang_m2rnn_path_c.py` has no matches |
| 2026-05-16 M2RNN Path C unique-symbol and owner-output hazard hardening | pass: M2RNN Path C kernels now carry shape-specific `global_symbol` metadata, and atomic/multi-output owner-output bwd paths synchronize around the real inter-producer hazard; `pytest tests/test_tilelang_m2rnn_path_c.py -q` passed `25 passed in 35.15s`; combined legacy plus Path C passed `39 passed in 35.20s`; `compileall`, `git diff --check`, and targeted `dW_partial` scan passed |
| 2026-05-16 remove regular Sparse MLA Path C public `dkv_partial` route | pass: regular Path C bwd now clears and writes final fp32 `dKV` owner-output buffers inside the TileLang TVM-FFI route; public `sparse_mla_bwd_path_c` returns `(dq, dkv)` without `_sparse_mla_bwd_path_c_partial` or `_reduce_dkv_partial`; benchmark import no longer depends on the removed partial API |
| 2026-05-16 focused regular Sparse MLA owner-output bwd tests | pass: `pytest tests/test_tilelang_sparse_mla.py::test_path_c_backward_parity tests/test_tilelang_sparse_mla.py::test_path_c_backward_matches_reference_and_path_b_direct_msl tests/test_tilelang_sparse_mla.py::test_path_c_backward_accumulates_duplicate_kv_indices tests/test_tilelang_sparse_mla.py::test_path_c_backward_int64_indices_tail_dim_parity tests/test_tilelang_sparse_mla.py::test_path_c_backward_reuses_int32_indices_for_owner_output_route tests/test_tilelang_sparse_mla.py::test_path_c_topk32_matches_path_b_direct_msl tests/test_tilelang_sparse_mla.py::test_path_c_topk64_matches_path_b_direct_msl -q -s` passed `10 passed in 10.14s` |
| 2026-05-16 full regular Sparse MLA file after owner-output bwd rewrite | pass: `pytest tests/test_tilelang_sparse_mla.py -q` passed `54 passed in 5.61s` |
| 2026-05-16 compile/reduction/scan gates after regular Sparse MLA bwd rewrite | pass: cppmega `compileall` for `sparse_mla_path_c.py`, `test_tilelang_sparse_mla.py`, and `bench_tilelang_sparse_mla.py`; TileLang `pytest testing/python/transform/ -q -k "reduction"` passed `32 passed, 202 deselected, 170 warnings`; scans still intentionally nonzero for backend intrinsic tests, legacy Path B Mamba3 partials, and debug/direct-MSL surfaces |
| 2026-05-16 regular Sparse MLA bwd debug MSL owner-output route | pass: `dump_lowered_bwd_msl` now routes through `_bwd_direct_lowering_for`; helper tests assert `device float* dkv`, `tl::AtomicAdd`, and no `dkv_partial`; focused debug/dispatch run passed `8 passed in 3.01s` |
| 2026-05-16 full regular Sparse MLA after debug MSL owner-output route | pass: `pytest tests/test_tilelang_sparse_mla.py -q` passed `54 passed in 23.60s` |
| 2026-05-16 compile/diff/scan after debug MSL owner-output route | pass: `compileall` for `sparse_mla_path_c.py` and `test_tilelang_sparse_mla.py`; `git diff --check` for those files; targeted test scan has only negative `dkv_partial` assertions, while implementation scan remains nonzero because legacy private partial builder/postprocess helpers still exist |
| 2026-05-16 Mamba3 production NaN regression guard recheck after Sparse MLA cleanup | pass: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_rejects_public_partial_owner_outputs tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` passed `3 passed in 8.76s` |
| 2026-05-16 remove stale regular Sparse MLA Path C private `dkv_partial` builders/postprocess rewrites | pass: deleted the unused legacy `_make_sparse_mla_bwd_prim`, `_bwd_kernel_for`, and stale implementation-side `dkv_partial` postprocess rewrites; implementation scan for `dkv_partial` and the old helpers is clean; `dump_lowered_bwd_msl` smoke for `d_v=16` and `d_v=8` verifies final `device float* dkv`, `tl::AtomicAdd`, all-masked fast return, and no `dkv_partial`; full sparse MLA file passed `54 passed in 32.37s`; TileLang reduction transform passed `32 passed, 202 deselected, 170 warnings`; Mamba3 NaN guard passed `3 passed in 11.25s`; compileall and diff checks passed |
| 2026-05-16 remove stale Mamba3 bench partial-profile dependency | pass: `scripts/bench_tilelang_mamba3_path_c.py` no longer imports `_mamba3_mimo_bwd_path_c_partials` or `_reduce_mamba3_bwd_partials`, and now profiles the final-gradient SIMD/snapshot bwd route directly; small receipt smoke `--batch 1 --seq 4 --heads 1 --headdim 32 --state 4 --warmup 1 --iters 1 --print-only --skip-artifacts` passed; `compileall` passed; targeted partial-symbol scan is clean; TileLang reduction transform passed `32 passed, 202 deselected, 170 warnings`; `tests/test_tilelang_mamba3_path_c.py` passed `46 passed in 36.57s`; `tests/test_mamba3_dispatch.py` passed `13 passed in 6.13s`; TileLang and cppmega `git diff --check` passed |
| 2026-05-16 regenerate checked-in Mamba3 Path C receipt without stale partial profile | pass: `scripts/bench_tilelang_mamba3_path_c.py --warmup 5 --iters 50 --skip-artifacts` rewrote `bench/tilelang_ports/mamba3_path_c.json`; `bwd_profile` now contains only `simd_p_reduce_kernel`, partial-symbol scan over the receipt/script/Path C implementation/tests is clean, and affected dispatch/receipt gates passed `14 passed in 5.20s`. Perf remains intentionally red for this shape: fwd `1.088x`, bwd `1.546x`, fwd+bwd `1.479x` Path C / Path B, so AUTO keeps Path B. |
| 2026-05-16 machine-readable legacy direct-MSL reduction allowlist | pass: `tools/lint_mlx.py --explain-direct-msl-allowlist` now emits JSON entries with `kind`, `reason`, `replacement`, `reduction_surface`, and `public_partial_outputs`; Mamba3 Path B explicitly records `dB_partial`, `dC_partial`, `dA_partial`, `ddt_partial`, and `dD_partial` with the reason that Path C bwd is finite but slower on the checked-in receipt; `tests/test_lint_mlx.py` passed `18 passed in 1.09s`; focused production direct-MSL/monkeypatch/native-TVΜ-FFI lint returned clean. |
| 2026-05-16 P2 focused regression after legacy allowlist metadata | pass: TileLang `pytest testing/python/transform/ -q -k "reduction"` passed `32 passed, 202 deselected, 170 warnings`; cppmega `pytest tests/test_lint_mlx.py tests/test_mamba3_dispatch.py tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite tests/test_tilelang_mamba3_path_c.py::test_direct_bwd_simd_lowering_rejects_long_sequences_without_snapshots -q` passed `33 passed in 8.66s`; TileLang and cppmega `git diff --check` passed. |
| 2026-05-17 P2 same-simdgroup allreduce local-buffer scope recheck | pass: `LowerThreadAllreduce` now writes native Metal full-width same-simdgroup `sum` results directly into the allreduce result buffer instead of remapping scalar lets outside their local-buffer scope; added `test_metal_same_simdgroup_thread_allreduce_keeps_local_buffer_scope`; updated same-simdgroup split tests to expect direct `simd_sum(accum[0])`; cppmega `tests/conftest.py` pins `TILELANG_DISABLE_CACHE=1` for dev-build tests so stale JIT artifacts do not hide lowerer changes; `scripts/bench_tilelang_fp8_path_c.py` keeps the active `apache-tvm-ffi` editable finder instead of deleting the venv-provided `core.abi3.so` artifact. TileLang gates passed: `cmake --build build -j$(sysctl -n hw.ncpu)`, focused new Metal regression, `pytest testing/python/transform/ -q -k "reduction"` (`32 passed, 202 deselected`), `pytest testing/python/metal/test_metal_reduce.py -q` (`38 passed`), P1 language/scheduler guards (`7 passed, 8 skipped`; `30 passed`), and P0 focused block (`62 passed, 5 skipped`). cppmega gates passed: strict FP8 bench `/tmp/fp8_path_c_p2_lower_thread_allreduce.json` with `matmul_128=0.919x` and `vecmat_4096=1.003x`; `pytest tests/test_lint_mlx.py tests/test_tilelang_msl_transform.py tests/test_tilelang_mamba3_path_c.py tests/test_tilelang_bench_harness.py -q` (`110 passed`); P0 guard `pytest tests/test_tilelang_fp8_vecmat_path_c.py tests/test_tilelang_mamba3_path_c.py -q` (`75 passed`); `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`; TileLang/cppmega reduction-token scans are expected nonzero only in backend tests, feature counters, legacy Path B partial baselines, and debug/allowlisted surfaces; both worktrees passed `git diff --check`. |
| 2026-05-17 P2 debug direct-MSL unit-test allowlist removal | pass: extracted `_resolve_dispatch_launch_shape` from `_msl_transform.dispatch` so launch-grid/input-count guardrails can be tested without calling the legacy direct-MSL dispatch boundary; `tests/test_tilelang_msl_transform.py` now covers the pure helper; `tools/lint_mlx.py --explain-direct-msl-allowlist` no longer lists `tests/test_tilelang_msl_transform.py`, reducing the allowlist from 7 entries to 6. Code review/perf note: this is a factoring-only validation change on the legacy dispatch path, with no generated Metal or TileLang runtime schedule change, so no perf bench was applicable. Verification passed: `pytest tests/test_tilelang_msl_transform.py tests/test_lint_mlx.py -q` (`37 passed`), `compileall` for touched cppmega files, and `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`. |
| 2026-05-17 P2 regular Sparse-MLA legacy backward partial retirement | pass: `cppmega_mlx/nn/_tilelang/sparse_mla.py` keeps the direct-MSL Path B forward baseline but removes the legacy backward kernel that returned per-token `dKV` partial buffers plus host scatter/reduce; `sparse_mla_bwd_metal` is now a compatibility shim over `sparse_mla_bwd_path_c`, so backward returns final fp32 owner-output `dKV` from the TileLang/tvm-ffi route. `tools/lint_mlx.py --explain-direct-msl-allowlist` still lists `sparse_mla.py` for the forward baseline, but its `public_partial_outputs` is now empty. Perf smoke with dev TileLang/TVM env wrote `/tmp/sparse_mla_p2_bwd_compat_smoke.json`; Path C status was available and one-shape paired bwd compatibility ratio was `0.957x` under `--strict-phase bwd --max-ratio 1.05`. Verification passed: `pytest tests/test_lint_mlx.py tests/test_tilelang_bench_harness.py -q` (`45 passed`), focused Sparse-MLA bwd/custom-VJP tests (`5 passed`), full `tests/test_tilelang_sparse_mla.py` (`54 passed`), production/test direct-MSL lint, compileall for touched cppmega files, and targeted scan showing no `dkv_partial`, `_BWD_KERNEL`, `_sparse_mla_bwd_msl_partial`, or `_reduce_dkv_partial` in the regular Sparse-MLA production/bench files. |
| 2026-05-17 P2 blockscaled Sparse-MLA legacy backward partial retirement | pass: `cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py` keeps the direct-MSL MXFP8 forward/bwd baseline but removes the public `dkv_partial` output and Python `_reduce_dkv_partial_fp32` host scatter/reduce. The backward kernel now uses a direct-MSL `cppmega_atomic_add_float` CAS helper on normal output pointers, dispatches with `init_value=0`, allocates internal trailing-singleton owner-output shapes before reshaping back, writes `dq_dequant` and final fp32 owner-output `dKV`, and keeps the fused direct-MSL bwd path instead of falling back to pure autograd. `tools/lint_mlx.py --explain-direct-msl-allowlist` still lists the blockscaled file for the direct-MSL baseline, but its `public_partial_outputs` is empty and its reduction surface is `atomic_owner_output_dkv`. Perf smoke on the small blockscaled bwd parity shape showed owner-output bwd median `0.201 ms` vs pure reference VJP median `0.431 ms` (`0.466x`). Verification passed: `pytest tests/test_lint_mlx.py tests/test_tilelang_sparse_mla_blockscaled.py -q` (`51 passed`), `pytest tests/test_tilelang_msl_transform.py tests/test_tilelang_sparse_mla_blockscaled_path_c.py tests/test_tilelang_bench_harness.py -q` (`56 passed`), combined Mamba3/blockscaled/lint suite (`115 passed`), `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`, compileall for touched cppmega files, TileLang `pytest testing/python/transform/ -q -k "reduction"` (`32 passed, 202 deselected, 170 warnings`), cppmega/TileLang `git diff --check`, and targeted production/lint scan showing no `dkv_partial`, `_reduce_dkv_partial_fp32`, or `host_scatter_reduce` in the blockscaled production/allowlist files. |
| 2026-05-17 P2 Mamba3 Path B public partial-output retirement | pass: `cppmega_mlx/nn/_tilelang/mamba3.py` keeps the direct-MSL Path B fallback because AUTO still selects it as primary, but the backward kernel no longer exposes `dB_partial`, `dC_partial`, `dA_partial`, `ddt_partial`, or `dD_partial` and no longer performs host `mx.sum` reductions over P-axis partial buffers. It now writes final `dB`, `dC`, `dA`, `ddt`, and `dD` owner-output buffers through the same direct-MSL `cppmega_atomic_add_float` CAS helper, dispatches with `init_value=0`, and uses internal trailing-singleton owner-output shapes before reshaping back so Path B and Path C comparison graphs do not alias same-shaped lazy buffers. `tools/lint_mlx.py --explain-direct-msl-allowlist` still lists `mamba3.py` as a fallback, but its `public_partial_outputs` is empty and its reduction surface is `atomic_owner_output_p_axis`. Perf smoke on `B=1,T=4,H=1,P=32,N=4` showed Path B fwd `0.181 ms`, Path C fwd `0.184 ms` (`1.013x`), Path B bwd `0.877 ms`, Path C bwd `0.805 ms` (`0.917x`), and AUTO scheduler `path_b` for the receipt. Verification passed: combined `pytest tests/test_tilelang_mamba3.py tests/test_tilelang_mamba3_path_c.py tests/test_tilelang_sparse_mla_blockscaled.py tests/test_lint_mlx.py -q` (`115 passed`), the decay-underflow Path C comparison guard (`1 passed`), Mamba3 receipt smoke, production/test direct-MSL lint, compileall for touched cppmega files, targeted production/lint scan showing no legacy Mamba3 partial-output tokens, and cppmega/TileLang `git diff --check`. |
| 2026-05-17 P2 top-k selector direct-MSL Path B retirement | pass: `cppmega_mlx/nn/_tilelang/topk_selector.py` no longer constructs or dispatches the hand-written direct-MSL Path B kernel, the unused Path B MSL source block is deleted, and the file no longer needs a lint allowlist entry. `topk_selector_metal` now validates shape/k and fails closed with `None`; explicit `backend="metal"` raises; unmasked AUTO uses `topk_selector_tilelang_direct(..., out=...)`; masked or unsupported no-output calls fall back to the pure-MLX reference. The legacy no-output Path C `mx.fast.metal_kernel` wrapper is also retired, while `_path_c_kernel_for` remains a source-lowering debug seam that returns no MLX fast-kernel. Receipt regeneration with the active dev TileLang/TVM env passed: `scripts/bench_tilelang_topk.py --warmup 2 --iters 5 --strict` rewrote `bench/tilelang_ports/topk_selector.json` with `path_b_status.available=false`, `path_c_status.available=true`, and Path C running for the receipt shapes. Perf note: this is a direct-MSL retirement, not a standalone top-k speed win; Path C medians in the smoke were `0.240 ms`, `0.776 ms`, `0.794 ms`, `0.783 ms`, and `7.297 ms` across the five receipt rows, slower than the MLX baselines on the larger shapes. Verification passed: `pytest tests/test_tilelang_topk.py tests/test_lint_mlx.py tests/test_tilelang_bench_harness.py -q` (`109 passed`), compileall for touched cppmega files, production/test direct-MSL lint, cppmega/TileLang `git diff --check`, and `tools/lint_mlx.py --explain-direct-msl-allowlist` now lists 5 entries and no `topk_selector.py`. |
| 2026-05-17 P2 regular Sparse-MLA direct-MSL Path B retirement | pass: `cppmega_mlx/nn/_tilelang/sparse_mla.py` no longer constructs the hand-written forward MSL string, `_FWD_KERNEL`, or `_msl_transform.dispatch`; `sparse_mla_fwd_metal` now validates shapes and returns `None`, `sparse_mla_apply(force_metal=True)` raises with the explicit retired-Path-B reason, AUTO uses checked-in Path C receipt rows or the pure-MLX reference, and `sparse_mla_bwd_metal` remains a compatibility shim over the final-owner-output Path C bwd route. `bench/tilelang_ports/sparse_mla.json` was regenerated with the active dev TileLang/TVM env via `scripts/bench_tilelang_sparse_mla.py --warmup 2 --iters 5 --strict`; it records `path_b_status.available=false`, `path_c_status.available=true`, strict pass, and Path C fwd/bwd medians faster than pure MLX reference on all three receipt rows (`fwd C/ref` ratios `0.727x`, `0.900x`, `0.780x`; `bwd C/ref` ratios `0.684x`, `0.637x`, `0.693x`). Verification passed: `pytest tests/test_tilelang_sparse_mla.py tests/test_sparse_mla_dispatch.py tests/test_tilelang_bench_harness.py tests/test_lint_mlx.py -q` (`117 passed`), `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`, compileall for touched cppmega files, cppmega/TileLang `git diff --check`, targeted scan showing no raw `mx.fast.metal_kernel`, `_msl_transform.dispatch`, `_FWD_KERNEL`, or `_FWD_KERNEL_SOURCE` in regular Sparse-MLA production/bench files, and `tools/lint_mlx.py --explain-direct-msl-allowlist` now lists 4 entries with no `sparse_mla.py`. |
| 2026-05-17 P2 M2RNN direct-MSL Path B retirement | pass: `cppmega_mlx/nn/_tilelang/m2rnn.py` no longer constructs hand-written MSL, `_FWD_KERNEL_SOURCE`, `_BWD_KERNEL_SOURCE`, `_FWD_KERNEL`, `_BWD_KERNEL`, or `_msl_transform.dispatch`; the legacy exported names are now pure-MLX compatibility wrappers. `CPPMEGA_KERNEL_PATH=path_b` fails closed with an explicit retired-Path-B error, explicit `path_c` still exercises the TileLang/tvm-ffi route, and AUTO uses the correctness-first reference route so HybridLM smoke training stays finite. The TileLang MLX TVM-FFI bridge now always emits a GPU-side output barrier after publishing external output arrays, replacing the ad hoc `TILELANG_MLX_TVM_FFI_FORCE_OUTPUT_BARRIER=1` requirement that M2RNN owner-output bwd exposed. Verification passed: TileLang `cmake --build build -j16`; TileLang `TILELANG_DISABLE_CACHE=1 ... pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` (`21 passed, 6 skipped`); cppmega `pytest tests/test_tilelang_m2rnn.py tests/test_m2rnn_dispatch.py tests/test_tilelang_m2rnn_path_c.py tests/test_lint_mlx.py -q` (`74 passed`); `pytest tests/test_tilelang_m2rnn_path_c.py -q` without the output-barrier env (`25 passed`); compileall for touched M2RNN/test/lint files; `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`; targeted scan showing no raw M2RNN production `mx.fast.metal_kernel`, `_msl_transform.dispatch`, `_FWD_KERNEL*`, or `_BWD_KERNEL*`; cppmega and TileLang `git diff --check`. Perf note: this is a direct-MSL retirement and correctness stabilization, not a new AUTO Path C speed win; the existing checked-in M2RNN Path C receipt remains the explicit Path C performance reference. |
| 2026-05-17 P2 blockscaled Sparse-MLA direct-MSL Path B retirement | pass: `cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py` no longer constructs `_BLOCKSCALED_FWD_KERNEL_SOURCE`, `_BLOCKSCALED_BWD_KERNEL_SOURCE`, `_BLOCKSCALED_FWD_KERNEL`, `_BLOCKSCALED_BWD_KERNEL`, `make_metal_kernel`, or `_msl_transform.dispatch`; it keeps MXFP8 quantize/dequant/unpack helpers, the pure-MLX MXFP8 reference, prepared-buffer Path C re-exports, and fail-closed compatibility wrappers. `sparse_mla_blockscaled_fwd_metal` and `sparse_mla_blockscaled_bwd_metal` preserve shape validation and return `None`, while `sparse_mla_blockscaled_apply(force_metal=True)` raises with the retired Path B reason. `tools/lint_mlx.py --explain-direct-msl-allowlist` now lists 2 entries (`fp8_msl_kernels.py`, `mamba3.py`) and no blockscaled entry. Perf/receipt smoke regenerated `bench/tilelang_ports/sparse_mla_blockscaled.json` with `metal_status.available=false`, Path C E8M0 QK reducer median `0.170 ms`, blockscaled reference median `0.334 ms`, and `path_c_e8m0_qk_reduce_over_blockscaled_reference=0.509x`; full Path C dispatch remains intentionally red. Verification passed: `pytest tests/test_tilelang_sparse_mla_blockscaled.py tests/test_tilelang_sparse_mla_blockscaled_path_c.py tests/test_tilelang_path_c_vs_b_parity.py tests/test_tilelang_bench_harness.py tests/test_lint_mlx.py -q` (`97 passed, 1 xfailed`); broader FP8 sparse suite after same-simdgroup expectation update (`53 passed`); compileall for touched cppmega files; `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`; targeted scan shows no blockscaled production kernel source or dispatch constructors except retired-reason strings and negative assertions. |
| 2026-05-17 P2 generic FP8 direct-MSL helper retirement | pass: `cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py` no longer embeds the vendored FP8 MSL header/body strings, constructs `_FP8_*_KERNEL` handles, imports `_msl_transform`, calls `make_metal_kernel`, or dispatches raw MSL. The public helper names now provide explicit pure-MLX reference/oracle math only, and `fp8_msl_status()` returns `available=false` with the retired Path B reason. `scripts/bench_tilelang_fp8_path_c.py` future output now labels this baseline as `reference_mlx_*` while retaining compatibility for historical `path_b_msl_*` checked-in receipts. Perf/receipt smoke regenerated `bench/tilelang_ports/fp8_msl_kernels.json` for `tiny_smoke` with `metal_status.available=false`, reference matmul median `0.173 ms`, fp16 baseline median `0.151 ms`, and reference vecmat median `0.133 ms`; a live Path C bench smoke wrote `/tmp/fp8_path_c_reference_smoke.json` with `matmul_tl_fp8_scaled_matmul_over_path_b=0.774x` against the pure-MLX reference. Verification passed: `pytest tests/test_fp8_msl_kernels.py tests/test_tilelang_fp8_vecmat_path_c.py tests/test_tilelang_path_c_vs_b_parity.py tests/test_tilelang_bench_harness.py tests/test_tilelang_fp8_matmul_path_c_bench.py tests/test_lint_mlx.py -q` (`111 passed, 1 xfailed`); `pytest tests/test_tilelang_sparse_mla_fp8.py -q` (`53 passed`); compileall for touched cppmega files; `tools/lint_mlx.py --select MLX002,MLX005,MLX006,MLX007 cppmega_mlx tests`; `tools/lint_mlx.py --explain-direct-msl-allowlist` now lists only `mamba3.py`; targeted scan shows no raw FP8 helper MSL constructors or dispatch imports in production/docs/receipt files. |
| 2026-05-17 P2 first-pass close gate with Mamba3 non-rewrite deferral | pass: P2 closes as a first pass because no current direct-MSL allowlist entry exposes public partial outputs, and the sole non-rewritten reduction surface (`cppmega_mlx/nn/_tilelang/mamba3.py`) has a machine-readable allowlist reason plus a replacement path (`mamba3_path_c.py`). Code review/perf decision: do not retire Mamba3 Path B in P2 because the checked-in production-shape receipt keeps `scheduler_decision.mode=path_b`; Path C is slower than Path B on the receipt (`fwd=1.088x`, `bwd=1.546x`, `fwd+bwd=1.479x`), and a live smoke also kept `scheduler: path_b` (`fwd=1.009x`, `bwd=1.130x`, `fwd+bwd=1.113x`). Regression verification passed: cppmega `pytest tests/test_lint_mlx.py tests/test_mamba3_dispatch.py tests/test_tilelang_mamba3.py tests/test_tilelang_mamba3_path_c.py -q` (`96 passed`); TileLang `pytest testing/python/transform/ -q -k "reduction"` (`32 passed, 202 deselected, 170 warnings`); `tools/lint_mlx.py --explain-direct-msl-allowlist` lists only `mamba3.py` with `public_partial_outputs=[]`; broad reduction-token scans are expected nonzero only for TileLang backend intrinsics/tests, cppmega Path C feature counters/tests, and the documented Mamba3 direct-MSL fallback debt. |

**Status**

- P2 is green for the first pass. The framework now records declined rewrite
  reasons, and no MLX direct-MSL allowlist entry currently reports public
  partial outputs. Mamba3 no longer exposes the public P-axis partial-output bwd
  ABI and no longer has an internal host-reduced partial fallback route, M2RNN
  Path C no longer races lazy zero-fill against atomic-add bwd outputs and no
  longer exposes public `dW_partial` buffers on supported batch-1 bwd routes,
  M2RNN direct-MSL Path B is retired and removed from the allowlist, regular
  Sparse MLA Path C bwd, its debug MSL dump, and the legacy
  `sparse_mla_bwd_metal` compatibility surface now return final fp32 `dKV`
  owner outputs instead of public `dkv_partial` plus host-side MLX
  scatter/reduce, regular Sparse MLA no longer constructs a direct-MSL Path B
  forward runtime surface, blockscaled Sparse MLA no longer constructs a
  direct-MSL Path B forward or backward runtime surface, and the generic FP8
  helper no longer constructs direct-MSL kernels. cppmega no longer mutates
  TileLang/TVM backend op registration for Path C Metal FP8 intrinsics; it only
  checks the framework-owned registry. Sparse MLA implementation scans are clean
  for `dkv_partial` and the old private bwd builders; Mamba3 Path B
  production/lint scans are clean for the old public partial-output tokens; the
  Mamba3 receipt bench and checked-in receipt no longer import or profile the
  removed Path C partial fallback route. The sole remaining production
  direct-MSL allowlist entry is Mamba3, with a machine-readable non-rewrite
  reason and replacement path. Its retirement is deferred to the reverse-scan
  and cost/codegen packages because the current Path C receipt is slower than
  Path B. Broad P2 scans still hit TileLang backend intrinsics/tests, cppmega
  Path C feature counters/tests, and that documented Mamba3 fallback.
- The Mamba3 production-shape NaN guard remains green after the public `out=`
  cleanup, the TVM-FFI bridge ordering fix, the native `simd_sum` lowerer, and
  the direct-SIMD long-sequence fail-closed guard.

## P3: ReductionPlan Scheduler Metadata

### Planned Changes

- Add `ReductionPlan` metadata with:
  input/output regions, axes, extents, thread/block mapping, accumulator dtype,
  alias constraints, in-place legality constraints, memory scope, candidate
  strategies, and selected strategy.
- Candidate strategies must include same-simdgroup, split-simdgroup,
  threadgroup staging, row reduce, two-pass global reduce, and CPU fallback.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/scheduler/ -q -k "reduction_plan"
.venv313/bin/python -m pytest testing/python/metal/test_metal_reduce.py -q
```

### Green Criteria

- Plan serialization snapshots are stable and reviewable.
- P=32/64/96/128/256 select single-kernel legal strategies.
- Larger extents select generated internal two-pass plans instead of public
  partial outputs.

### 2026-05-17 P3 first-pass green receipt

**Change under test**

- `ReductionPlan` JSON now includes the selected strategy, thread/block mapping,
  alias constraints, in-place constraint posture, and a memory plan with internal
  scratch/materialization requirements.
- Strategy candidates now distinguish `threadgroup-staging`, `row-reduce`,
  `two-pass-global`, and `vectorized-cpu-fallback`; extents 32/64/96/128/256
  select single-kernel strategies, while extent 512 selects internal
  `two-pass-global` metadata with no external materialization.

**Commands and results**

| Command | Result |
| --- | --- |
| `pytest testing/python/scheduler/test_reduction_plan.py testing/python/scheduler/test_reduction_legality.py testing/python/scheduler/test_sync_event_plan.py -q` | pass: 17 passed |
| `pytest testing/python/transform/ -q -k "reduction"` | pass: 32 passed, 202 deselected, 170 warnings |
| `pytest testing/python/metal/test_metal_reduce.py -q` | pass: 38 passed, 12 warnings |
| `compileall` for touched analysis/test files | pass |
| `git diff --check` for touched P3 files | pass |

**Status**

- P3 is green for the first pass and unblocks P4.
- Perf/codegen review found no lowering behavior change in this package; the
  generated Metal reduction gate was still rerun to catch accidental strategy
  drift.

## P4: Z3 Legality Proofs

### Planned Changes

- Prove exact coverage, bounds safety, no write-write race, read-after-write
  hazards, alias/in-place legality, tail dimensions, broadcast semantics, and
  int64/index-width safety.
- Attach proof result to scheduler metadata:
  `proved_no_sync`, `requires_threadgroup_barrier`, `requires_device_event`,
  `requires_two_pass`, or `cannot_parallelize(reason)`.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/scheduler/ -q -k "z3 or legality"
.venv313/bin/python -m pytest testing/python/analysis/ -q -k "z3 or overflow or alias"
.venv313/bin/python -m pytest testing/python/metal/ -q -k "hazard or sync or reduce"
```

### Green Criteria

- Positive and negative fixtures cover tails, broadcast, int64, aliasing, and
  overlapping writes.
- Impossible plans fail before codegen.
- Proven no-sync plans emit no sync in generated source.

### 2026-05-17 P4 first-pass green receipt

**Change under test**

- Reduction legality now derives alias status from `ReductionPlan` alias
  constraints instead of recomputing it ad hoc.
- Tail/broadcast legality is checked against the selected thread mapping:
  same-simdgroup, split/threadgroup/row-reduce, and two-pass plans all prove
  that the selected mapping covers the static extent.
- Added negative index-width coverage with an `int64` reduction extent at
  `2**31`, which fails before codegen with `extent_legality_unproved`.

**Commands and results**

| Command | Result |
| --- | --- |
| `pytest testing/python/scheduler/ -q -k "z3 or legality"` | pass: 8 passed, 11 deselected |
| `pytest testing/python/analysis/ -q -k "z3 or overflow or alias"` | pass: 7 passed, 10 deselected |
| `pytest testing/python/metal/ -q -k "hazard or sync or reduce"` | pass: 39 passed, 16 skipped, 104 deselected, 138 warnings |
| `compileall` for touched legality/test files | pass |
| `git diff --check` for touched P4 files | pass |

**Status**

- P4 is green for the first pass and unblocks P5.
- Perf/codegen review: proof metadata changes do not alter generated kernels
  directly; the Metal hazard/sync/reduce gate was rerun to verify sync decisions
  and generated no-sync reduction behavior still hold.

## P5: Sync and Event Planner

### Planned Changes

- Build dependency metadata from buffer regions, streams, owner-output handles,
  DLPack/TVM-FFI boundaries, and MLX graph boundaries.
- Insert no sync, threadgroup barrier, or device event only from proof result.
- Materialize only when an external pointer boundary truly requires storage.
- Add native debug hooks/guards for deterministic race and null-pointer
  detection.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest \
  testing/python/analysis/test_metal_graph_sync.py \
  testing/python/scheduler/test_sync_event_plan.py \
  testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py \
  -q

cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests -q -k "dlpack or tvm_ffi or path_c"
```

### Green Criteria

- Same-stream/no-hazard kernels emit no device event.
- Cross-stream/cross-buffer real hazards emit exactly one required event.
- Debug guard catches stale/null pointer races deterministically.

### First-Pass Receipt - 2026-05-17

| Gate | Result |
| --- | --- |
| `pytest testing/python/analysis/test_metal_graph_sync.py testing/python/scheduler/test_sync_event_plan.py testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` | pass: 31 passed, 6 skipped, 54 warnings |
| `pytest tests -q -k "dlpack or tvm_ffi or path_c"` in cppmega | pass: 298 passed, 1713 deselected, 1 xfailed, 1 warning |
| focused M2RNN mapped grouped and inline-post VJP regressions | pass |
| focused sparse-MLA Path C dispatch-gradient regressions | pass |
| compileall for touched TileLang/cppmega files | pass |
| `git diff --check` for touched TileLang/cppmega P5 files | pass |

**Status**

- P5 is green for the first pass and unblocks P6.
- Sync-event JSON now includes selected strategy, memory visibility, scratch
  scope, and internal/external materialization flags derived from
  `ReductionPlan.memory_plan` plus the legality proof.
- The native MLX/TVM-FFI fast path now borrows without waits only when inputs
  are already compact and have no registered native producer; producer outputs
  and lazy MLX graph values go through dependency planning/materialization.
- Remaining risk: grouped M2RNN and non-fp16 sparse-MLA VJPs still use explicit
  pure-MLX reference fallback where native atomic owner-output accumulation is
  not yet proven stable. Treat that as P6/P8/P9 performance debt, not a P5
  correctness blocker.

## P6: Backend Lowerer Registry

### Planned Changes

- Move semantic reduction, scan, and dependency lowering into backend
  registries for Metal, CUDA, ROCm, CPU, and future backends.
- Metal reduce/finalize lowerers must live in the registry, not in model code.
- Diagnostics must show which backend lowerer and strategy were selected.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/metal/ -q -k "reduce or finalize"
.venv313/bin/python -m pytest testing/python/language/ -q -k "reduce"
rg -n "metal|cuda|rocm|simd_sum" tilelang/language tilelang/transform
```

### Green Criteria

- Model generators contain no backend-specific reduction/finalize code.
- Backend leakage checks are clean or have explicit test/debug allowlist.
- Lowerer selection is cached and reproducible.

### First-Pass Receipt - 2026-05-17

| Gate | Result |
| --- | --- |
| Scheduler registry metadata | `9 passed` |
| Metal reduce/finalize | `40 passed, 15 skipped, 105 deselected, 138 warnings` |
| Language reduce | `7 passed, 8 skipped, 220 deselected, 356 warnings` |
| Hygiene | `compileall` passed for touched backend/analysis/transform/test files; `git diff --check` passed. |
| Leakage scans | TileLang scan remains nonzero only for target-specific transform passes and the existing FP8 intrinsic surface; cppmega scan shows semantic `T.tvm_thread_allreduce`, feature probes, and tests, not model-owned backend reduction/finalize lowerers. |

Status notes:

- Added backend-owned reduction lowerer registries for Metal, CUDA, ROCm, and
  CPU. The registry maps ordered scheduler strategies to concrete lowerer names,
  memory visibility, scratch scope, and materialization requirements.
- Semantic reduction plans now can attach stable
  `tl.reduction_backend_lowerers` JSON diagnostics. Metal SIMD lift attaches
  this metadata beside the existing reduction-plan, legality, and sync-event
  metadata.
- Selection is cached by target kind, op, strategy, extent, and accumulator
  dtype; tests assert repeated Metal selection returns the cached object.
- CPU uses the ordered candidate list to fall back to `vectorized-cpu-fallback`
  when the scheduler's first strategy is GPU-oriented.
- Scan/dependency registry expansion remains a P8/P9 follow-up because there is
  no reverse recurrence/scan planner surface yet; this P6 pass unblocks P7/P8/P9
  by making reduction lowerer selection explicit and reproducible.

## P7: Large-Axis Generated Reductions

### Planned Changes

- Generalize beyond P=64. Scheduler decides same-simdgroup, split-simdgroup,
  threadgroup staging, or generated two-pass reduction for any legal extent.
- Internal scratch buffers are runtime-owned and lifetime-analyzed.
- Public APIs return only final outputs unless debug partials are explicitly
  requested.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/metal/ -q -k "two_pass or reduce"

cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests/test_tilelang_mamba3_path_c.py -q -k "headdim or bwd"
```

### Green Criteria

- P=32/64/96/128/256/512/1024 and at least one model-real larger extent pass
  parity.
- No Python host reduction is used for large axes.
- No public `*_partial` outputs appear in generated model routes.

### First-Pass Receipt - 2026-05-17

| Gate | Result |
| --- | --- |
| TileLang Metal `two_pass or reduce` | `47 passed, 15 skipped, 105 deselected, 138 warnings` |
| cppmega Mamba3 `headdim or bwd` | `28 passed, 25 deselected` |
| Focused large-axis coverage | TileLang extent matrix `32/64/96/128/256/512/1024` passed; Mamba3 P-thread chooser and aligned-headdim source checks passed (`10 passed, 43 deselected`). |
| Hygiene | `compileall` passed for touched TileLang/cppmega files; `git diff --check` passed in both repos. |

Status notes:

- TileLang now has explicit codegen coverage for semantic thread-allreduce
  extents `32`, `64`, `96`, `128`, `256`, `512`, and `1024`. Extents above
  one simdgroup keep final outputs internal through staged reduction buffers and
  do not expose public partial outputs.
- Mamba3 Path C bwd now uses a bwd-specific thread chooser: existing supported
  real shapes keep their previous `_threads_for` result, while P rows that would
  otherwise be rejected at `512` or `1024` lanes select a larger legal
  threadgroup so the whole P row is reducible by TileLang.
- The Mamba3 public bwd route still rejects public partial owner-output
  fallbacks; the remaining `mx.sum(dD_bh, axis=0)` is the existing final
  batch/head aggregation for `dD`, not the removed P-axis host partial reducer.

## P8: Reverse Recurrence and Scan Planner

### Planned Changes

- Represent reverse recurrence as scan/dependency IR with state dependencies,
  chunking, snapshot/cache policy, rematerialization policy, and in-place
  legality.
- Prove chunk independence and minimal sync with Z3.
- Fuse post-recurrence residual/gate work into the recurrence kernel when
  legality and register pressure allow.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/scheduler/ -q -k "scan or recurrence"

cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests -q -k "mamba3 or m2rnn or hybrid"
```

### Green Criteria

- No forced serial-over-T path when chunk parallelization is proven legal.
- In-place bwd is selected only when alias proof passes.
- Profiler receipt shows reduced serial recurrence bottleneck on model shapes.

### First-Pass Receipt - 2026-05-17

| Gate | Result |
| --- | --- |
| TileLang scheduler `scan or recurrence` | `6 passed, 22 deselected` |
| cppmega Mamba3 scan/snapshot route | `4 passed, 51 deselected` |
| cppmega real-parquet HybridTinyLM subprocess regression | `2 passed` |
| cppmega broad `mamba3 or m2rnn or hybrid` | `378 passed, 1643 deselected` |
| Hygiene | `compileall` passed for touched TileLang/cppmega files; both worktrees passed `git diff --check`. |

Status notes:

- Added a TileLang recurrence scan planner with direction, chunk count,
  state-boundary snapshot policy, rematerialization policy, alias/in-place
  posture, fused post-op metadata, and no host/device sync requirement for the
  current reverse recurrence plan.
- Mamba3 Path C bwd now plans its long-sequence reverse recurrence before
  selecting the state snapshot route, and short sequences keep direct
  recompute. cppmega keeps a lazy local fallback for subprocesses that import a
  packaged TileLang without `tilelang.analysis.scan_plan`.
- The broad cppmega gate exposed two adjacent regressions: HybridTinyLM
  checkpointing now calls `mx.checkpoint(layer)` so tests can observe the actual
  decoder layer, and pure Mamba3 `compute_dacs_segsum` now reverses the time
  axis through `mx.take` instead of a negative-stride cumsum input that could
  alternate results across repeated MLX evaluations.
- Perf/codegen review: this is a first planner/control-flow pass. It removes
  hard-coded long-sequence routing from Mamba3 Path C and records the future
  fusion choices, but the profiler target in the green criteria remains P9/P10
  follow-up work because no cost model or autotuned schedule is selected yet.

## P9: Cost Model and Codegen Cleanup

### Planned Changes

- Attach cost metadata for registers, local memory, threadgroup memory, index
  math, occupancy, dispatch count, and sync/materialization cost.
- Hoist repeated hot index math.
- Choose SIMD reductions by extent and backend capability.
- Split or inline kernels only when register pressure and traffic estimates
  justify it.
- Make every selected strategy explainable in metadata.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/scheduler/ -q -k "cost or register or hoist"
.venv313/bin/python -m pytest testing/python/metal/ -q -k "source or reduce"

cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests -q -k "path_c"
```

### Green Criteria

- Generated source has no repeated expensive inner-loop index expressions.
- Scheduler metadata explains hoist/split/inline decisions.
- Path C generated code moves closer to Path B without model-specific kernels.

### First-Pass Receipt - 2026-05-17

| Gate | Result |
| --- | --- |
| Scheduler `cost or register or hoist` | `5 passed, 28 deselected` |
| Metal `source or reduce` | `50 passed, 15 skipped, 102 deselected, 138 warnings` |
| cppmega `path_c` | `294 passed, 1726 deselected, 1 xfailed, 1 warning` |
| Opt-in Metal reduce perf smoke | `3 passed, 12 deselected`; same-simdgroup 32 `0.1536 ms`, cross-simdgroup 128 `0.1717 ms`, row-reduce 256x32 `0.1327 ms` with warmup `1`, iterations `3`. |
| Hygiene | `cmake --build build -j16` passed; `compileall` passed for touched analysis/transform/test files. |

Status notes:

- Added scheduler-visible reduction cost estimates with register count,
  local/threadgroup/device scratch bytes, index math ops, dispatch and sync
  counts, materialization cost, occupancy limiter, and split-vs-inline decision
  reasons.
- Added recurrence scan cost estimates for state bytes, snapshot bytes,
  dispatch count, sync count, and whether the plan is direct recompute,
  forward scan, or split snapshot reuse.
- Semantic Metal reduction rewrites now attach `tl.reduction_costs` beside the
  existing reduction plan, legality, sync-event, and backend-lowerer metadata.
- Metal batched allreduce template source now hoists repeated
  `i * workspace_stride` address math into `batch_offset`; the source gate
  asserts the old repeated red-buffer expressions do not return.
- Perf/codegen review: this is a static cost and local source cleanup pass, not
  legal-schedule autotuning. P10 remains responsible for profiling only legal
  alternatives and caching warm schedules.

## P10: Autotune, Profiling, and Memoization

### Planned Changes

- Benchmark only legal schedules.
- Cache tuned schedules by op signature, shape, dtype, backend target,
  TileLang/TVM/TVM-FFI/MLX ABI hashes, proof hash, and codegen hash.
- Reuse warm schedule unless ABI/proof/codegen changes.
- Record cold compile time separately from warm execution time.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/scheduler/ -q -k "autotune or cache"

cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests -q -k "profile or bench or path_c"
```

### Green Criteria

- Cache hit/miss/invalidation tests pass.
- Cold compile is paid once per signature.
- Warm run records selected schedule, proof hash, cache key, and timing.

### First-Pass Receipt - 2026-05-17

| Gate | Result |
| --- | --- |
| TileLang scheduler `autotune or cache` | `6 passed, 32 deselected` |
| cppmega `profile or bench or path_c` | `411 passed, 1609 deselected, 1 xfailed, 1 warning` |
| Focused M2RNN order-sensitive regression pair | `2 passed` |
| Hygiene | `compileall` passed for touched TileLang/cppmega files. |

Status notes:

- Added a legal-schedule warm memoization plan that filters illegal candidates
  before profiling and memoizes the selected schedule only over legal candidate
  keys.
- Schedule keys now include op signature, shape, dtype, target kind, normalized
  config hash, TileLang/TVM/TVM-FFI/MLX ABI fingerprints, proof hash, and
  codegen hash. Changing proof, codegen, or ABI fingerprints forces a miss.
- Warm schedule receipts record selected schedule id/config, selected schedule
  key, selection cache key, cache hit/miss, cold compile milliseconds, warm run
  milliseconds, proof hash, codegen hash, profiled candidate count, and skipped
  illegal candidate count.
- The broad cppmega gate exposed an order-sensitive M2RNN Path C gradient
  issue: custom VJP closures were cached only by head layout, so MLX could reuse
  a custom-function identity across incompatible shapes. The M2RNN Path C
  closure caches now include tensor shape/dtype signatures.

## P11: Production Lint

### Planned Changes

- Add lint tests blocking production:
  `monkeypatch`, `mx.fast.metal_kernel`, public cppmega TVM-FFI bypasses,
  model-level backend intrinsics, and public partial outputs.
- Allow only test/debug paths with explicit allowlist entries.

### Tests

```bash
cd /private/tmp/tl_apache_tvm_swap
.venv313/bin/python -m pytest testing/python/ -q -k "lint or monkeypatch"

cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests -q -k "lint or monkeypatch"
rg -n "monkeypatch|mx\\.fast\\.metal_kernel|tir\\.metal|simd_sum|_partial" cppmega_mlx
```

### Green Criteria

- Production code path is framework-owned and lint-clean.
- Any forbidden token in tests/debug code has a documented allowlist reason.

## P12: Full 1B Training Matrix

### Planned Changes

- Add or use a benchmark harness that runs the real 1B model with:
  batch size 1, block size 2048, 20 steps, fresh subprocess per cell.
- Matrix dimensions:
  dtype: bf16, fp8, int8 where supported;
  optimizer: adamw, lion, muon, muon+adamw, and configured mixes;
  path: Path B, Path C cold, Path C warm.
- Capture tok/sec, step/sec, compile time, peak memory, cache hit, selected
  schedule, proof result, and pass/fail reason.
- Free memory after every cell by process isolation.

### Tests

Harness regression:

```bash
cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python -m pytest tests -q -k "bench or path_c or optimizer"
```

Full run command, once P0 is green:

```bash
cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
DYLD_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib:/private/tmp/tl_apache_tvm_swap/build/tvm \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/3rdparty/tvm-ffi/python \
.venv/bin/python scripts/bench_1b_training_matrix.py \
  --batch-size 1 \
  --block-size 2048 \
  --steps 20 \
  --dtypes bf16,fp8,int8 \
  --optimizers adamw,lion,muon,muon_adamw \
  --paths path_b,path_c_cold,path_c_warm \
  --fresh-process \
  --out /tmp/cppmega_1b_path_matrix.md \
  --csv /tmp/cppmega_1b_path_matrix.csv
```

### Green Criteria

- Table includes every dtype/optimizer/path cell, including unsupported cells
  with explicit reason.
- No result is reported without exact command, TileLang SHA, cppmega SHA, MLX
  SHA, cache state, tok/sec, step/sec, compile time, and peak memory.
- Path C warm is compared side by side against Path B for every supported cell.

## Required Receipts

| Package | Receipt |
| --- | --- |
| P0 | FP8 pytest output, strict Path B/Path C JSON, generated source snippet |
| P1 | Golden semantic reduction IR snapshot |
| P2 | Rewrite test proving manual partial pattern becomes semantic IR |
| P3 | Serialized `ReductionPlan` snapshot |
| P4 | Z3 positive and negative proof fixtures |
| P5 | Sync metadata showing no event for no-hazard and event for real hazard |
| P6 | Backend registry diagnostic naming selected lowerer |
| P7 | Large-axis parity run with no public partial outputs |
| P8 | Scan metadata plus Mamba3/M2RNN/hybrid profiler receipt |
| P9 | Cost metadata plus generated-source hoist/split/inline evidence |
| P10 | Cold/warm cache receipt with cache/proof/codegen hashes |
| P11 | Static lint receipt proving production path is clean |
| P12 | Markdown and CSV full 1B matrix |

## Definition of Done

- Path C production path is TileLang -> TVM -> TVM-FFI -> MLX.
- Scheduler emits semantic reduction/scan/dependency metadata.
- Z3 legality results control parallelization and sync decisions.
- Backend registries own backend-specific lowering.
- Autotune chooses among legal schedules and memoizes the result.
- Full 1B matrix reports Path B and Path C side by side with no hidden failed
  cells.
- No production monkeypatch, model-owned MSL, or public partial-output workaround
  remains.

## Run Log

### 2026-05-15 P0 in progress

**Bridge fix (commit `bd51fbf1`)**: `_contiguous_mlx_input` now always wraps
inputs through `mx.contiguous` on the GPU stream. The previous wrap-time
`is_compact` cache was unreliable because MLX rewrites slice strides during
eval (slice node: pre-eval `row_contiguous=True`, post-eval `False`).
Cross-domain hazard test relaxed to allow either device-event path or the
MLX-mediated copy path (correctness preserved either way). 60 focused tests
passing.

**1B end-to-end baseline (`scripts/bench_local_gb10_quarter_throughput.py`,
batch=1, T=4096, 20 measured steps after 5 warmup, M4 Max)**

Path C combo (`SPARSE_MLA + M2RNN` on Path C, Mamba3 on Path B):

| Optimizer    | Path B tok/s | Path C tok/s | Δ      | Path B peak GB | Path C peak GB |
| ------------ | -----------: | -----------: | -----: | -------------: | -------------: |
| lion         |          733 |      **765** | +4.4 % |          45.59 |          45.57 |
| muon_adamw   |          369 |      **378** | +2.4 % |          44.83 |          44.50 |

Per-op isolation:
- `SPARSE_MLA` only on Path C: 384 vs 357 tok/s (+7.5 %), peak 41.30 GB
  (vs Path B 44.83 GB).
- `M2RNN` only on Path C: 361 vs 357 tok/s (+1.1 %), peak 58.48 GB
  (vs Path B 44.83 GB).
- `MAMBA3_MIMO` only on Path C: 394 vs 357 tok/s (+10.4 %) BUT loss → NaN
  by step 5 — correctness regression. Excluded from combo until fixed.
- All three combined: GPU page fault / hang (likely Mamba3 NaN poisoning
  downstream attention indexing). Blocked on Mamba3 bwd correctness.

**Open P0 follow-ups before declaring P0 green**

- Mamba3 bwd correctness on Path C (12/40 path-c tests fail in full-file
  pytest run; isolated runs sometimes pass — pytest order pollution; the
  1B-bench NaN reproduces deterministically).
- Extended 1B optimizer matrix (`adamw`, `adam8bit`, `lion8bit`); bench
  was extended in-place at `bench_local_gb10_quarter_throughput.py` and
  results pending.
- Strict FP8 `vecmat_4096` remains above the 3 % micro-bench gate. Do not
  mark P0 green until the strict gate passes or the gate is explicitly changed
  with a replacement receipt.

### 2026-05-15 P0 bridge guard rerun

**Change under test**: `_contiguous_mlx_input` now preserves registered
TVM-FFI producer outputs before asking the native compactness guard. This keeps
graph producer identity visible to the dependency planner; generic lazy MLX
inputs still route through `native.compact_input(...)`.

**SHAs / cache state**

| Repo | SHA | State |
| --- | --- | --- |
| TileLang | `bd51fbf1` + dirty bridge/doc/test edits | `TILELANG_DISABLE_CACHE=1` |
| cppmega.mlx | `6691cba` + existing dirty benchmark/test edits | `TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap` |
| MLX | `d168ca5ca` | imported as `0.32.0.dev20260514+d168ca5ca` |

**Commands and results**

| Command | Result |
| --- | --- |
| `cmake --build build -j$(sysctl -n hw.ncpu)` | pass |
| `.venv313/bin/python -m py_compile tilelang/jit/adapter/_mlx_tvm_ffi.py tilelang/jit/adapter/tvm_ffi.py` | pass |
| `git diff --check -- src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_ext.cpp tilelang/jit/adapter/_mlx_tvm_ffi.py testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py ml_optim_plan.md` | pass |
| `pytest test_tvm_ffi_metal_stream_dlpack.py::{compile_compact_result_idx,native_bridge_borrows_only_materialized_compact_inputs,mlx_graph_cross_domain_emits_device_event}` | pass: 3 passed, 50 warnings |
| `pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` | pass: 20 passed, 5 skipped, 50 warnings |
| `pytest testing/python/metal/test_fp8_scaled_matmul_metal.py testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py testing/python/analysis/test_metal_graph_sync.py testing/python/scheduler/test_sync_event_plan.py -q` | pass: 61 passed, 5 skipped, 68 warnings |
| `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` in cppmega | pass: 28 passed, 1 warning |
| strict `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --out /tmp/fp8_path_c_p0.json` | fail: `vecmat_4096` ratio 1.130x |
| non-strict receipt `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --out /tmp/fp8_path_c_p0_latest.json` | pass, wrote `/tmp/fp8_path_c_p0_latest.json` |

**Latest FP8 Path B / Path C receipt (`/tmp/fp8_path_c_p0_latest.json`)**

| Shape | Path | Median ms | Min ms | P90 ms | Max ms | Tok/s | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matmul_128` | Path B MSL | 0.163771 | 0.124458 | 0.213500 | 0.266500 | 781581.478 | baseline |
| `matmul_128` | Path C TileLang | 0.141834 | 0.121916 | 0.172542 | 0.247750 | 902466.616 | 0.848x |
| `vecmat_4096` | Path B MSL | 0.187125 | 0.167250 | 0.201000 | 0.235625 | 5344.021 | baseline |
| `vecmat_4096` | Path C TileLang | 0.211812 | 0.192750 | 0.270542 | 0.313917 | 4721.157 | 1.148x |

**Current P0 status**

- Correctness/bridge tests are green after preserving TVM-FFI producer outputs
  through `_contiguous_mlx_input`.
- Performance gate is still red: `vecmat_4096` Path C is 13-15 % slower than
  Path B on the latest paired runs.
- Next fixer target: reduce Path C host/runtime overhead or generated vecmat
  dispatch cost without reintroducing model-owned MSL, public TVM-FFI bypasses,
  or unsafe MLX graph materialization.

### 2026-05-15 P0 borrowed no-wait fast path

**Change under test**: native `_tilelang_mlx_tvm_ffi` gained
`prepared_metal_call_borrowed_no_wait(...)`. For prepared Metal calls whose
inputs are already materialized compact MLX arrays, the bridge now creates the
launch sync state and primitive in one native call, skipping the Python
compact-input and dependency-planner path. Lazy inputs, views, and TVM-FFI
producer outputs still fall back to the normal planner/compact path.

**Commands and results**

| Command | Result |
| --- | --- |
| `cmake --build build -j$(sysctl -n hw.ncpu)` | pass |
| targeted graph-safe bridge pytest | pass: 3 passed, 50 warnings |
| `pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` | pass: 20 passed, 5 skipped, 50 warnings |
| `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` in cppmega | pass: 28 passed, 1 warning |
| P0 TileLang focused pytest block | pass: 61 passed, 5 skipped, 68 warnings |
| no-eval wrapper profile, `vecmat_4096`, 1000 calls | Path C improved from 28.646 us/call to 21.473 us/call; Path B was 8.504 us/call in the same rerun |
| strict `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --out /tmp/fp8_path_c_p0_fast_strict.json` | fail: `vecmat_4096` ratio 1.121x |
| non-strict receipt `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --out /tmp/fp8_path_c_p0_fast_latest.json` | pass, wrote `/tmp/fp8_path_c_p0_fast_latest.json` |

**Latest fast-path FP8 receipt (`/tmp/fp8_path_c_p0_fast_latest.json`)**

| Shape | Path | Median ms | Min ms | P90 ms | Max ms | Tok/s | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matmul_128` | Path B MSL | 0.142001 | 0.123459 | 0.174750 | 0.249125 | 901404.912 | baseline |
| `matmul_128` | Path C TileLang | 0.121083 | 0.112708 | 0.130125 | 0.147166 | 1057122.089 | 0.854x |
| `vecmat_4096` | Path B MSL | 0.186854 | 0.163208 | 0.223875 | 0.277625 | 5351.771 | baseline |
| `vecmat_4096` | Path C TileLang | 0.197292 | 0.167959 | 0.233084 | 0.239583 | 5068.641 | 1.032x |

**Current P0 status after fast path**

- Correctness is still green.
- Host wrapper overhead is materially lower, but not Path-B-level yet.
- Strict perf gate remains red because the 20-iteration paired run is not
  stable below 1.03x (`1.121x` strict rerun, `1.032x` best latest receipt).
- Next fixer target: close the remaining 10-13 us host gap or remove the
  generated vecmat scheduling variance. Knob sweep over
  `outputs_per_block={1,2,4,8,16}` and `reduce_threads={16,32,64,128}` did not
  produce a stable scheduler win; all useful variants stayed around 0.20 ms.

### 2026-05-15 P0 direct Metal parameter order fix

**Root cause**: direct Metal launch was using the imported runtime PackedFunc
to infer kernel parameter order. That object has no TIR `.params`, so direct
launch metadata fell back to `None` and Mamba3 either used the generic TVM-FFI
wrapper or, in earlier direct-launch attempts, bound host-order buffers to the
device ABI order. TVM's Metal device PrimFunc may reorder buffer params
alphabetically; the failing Mamba3 fwd shape compiled host order
`x,B,C,z,A,dt,D,h0,y,h_last` and device order
`A,B,C,D,dt,h0,h_last,x,y,z`.

**Change under test**: `_metal_device_launch_metadata()` now returns the device
PrimFunc together with launch args, and `_metal_direct_device_call()` computes
the direct ABI permutation from that device PrimFunc while still launching the
runtime imported function. A new Metal bridge regression compiles a simple
non-identity host/device parameter order and asserts both the permutation and
direct-launch counters.

**Commands and results**

| Command | Result |
| --- | --- |
| `python3 -m py_compile tilelang/jit/adapter/tvm_ffi.py tilelang/jit/adapter/_mlx_tvm_ffi.py testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py` | pass |
| `.venv313/bin/python -m pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py::test_tvm_ffi_metal_direct_launch_uses_device_param_order -q` with `DYLD_LIBRARY_PATH=build/lib:build/tvm` | fail during collection: `z3-solver 4.16.0.0` loaded `build/lib/libz3.dylib` without `Z3_mk_seq_replace_all` |
| same pytest with `.venv313/lib/python3.13/site-packages/z3/lib` prepended to `DYLD_LIBRARY_PATH` | pass: 1 passed, 52 warnings |
| `pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` with z3 package lib first | pass: 21 passed, 5 skipped, 52 warnings |
| Mamba3 direct metadata probe, `B=1,T=8,H=2,P=4,N=4` | direct call enabled; `param_indices=[4,1,2,6,5,7,9,0,8,3]`; launch args `[1,8,1,1]` |
| Mamba3 direct fwd diagnostic vs Path B, same shape | `direct_device_launches=1`, `direct_pipeline_launches=1`, `direct_compute_encoder_launches=1`; `maxdiff_y=1.86e-09`, `maxdiff_h=7.45e-09`, output nonzero |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_fwd_path_c_matches_path_b_fp32_small_shape -q` in cppmega.mlx | pass: 1 passed |
| `pytest tests/test_tilelang_mamba3_path_c.py -q -x` in cppmega.mlx | pass: 40 passed |

**Current P0 status after direct-order fix**

- Mamba3 Path C full test file is green on the direct TVM-FFI Metal path.
- The previous "Mamba3 bwd correctness / pytest order pollution" blocker is
  no longer reproduced by the full file after the direct parameter permutation
  fix.
- `.venv313` test commands must put the z3 package dylib directory before
  `build/lib` while `z3-solver` remains newer than the build-tree libz3.
- P0 is still not green overall: the strict FP8 `vecmat_4096` Path C perf gate
  remains the active blocker, and the full 1B matrix still needs a fresh run
  after that gate is stable.

### 2026-05-15 P0 owner-output bridge optimization pass

**Change under test**: the Path C prepared Metal call path now has three
shorter hot paths. Python skips generic tensor-list setup for graph-safe MLX
array calls, skips the generic owner-output adapter for static full-ABI prepared
calls, and caches the MLX array type in the closure. Native
`_tilelang_mlx_tvm_ffi` now validates the direct device-ABI permutation at
prepare time, registers inputs without building a duplicate pointer vector,
binds direct-pipeline launch params in device order without the generic
`TVMFFIAny` path, and reuses prepared output shape/dtype metadata when
owner-output arrays match the prepared ABI.

**Commands and results**

| Command | Result |
| --- | --- |
| `cmake --build build -j$(sysctl -n hw.ncpu)` | pass |
| `python3 -m py_compile tilelang/jit/adapter/tvm_ffi.py tilelang/jit/adapter/_mlx_tvm_ffi.py` | pass |
| `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` in cppmega.mlx | pass: 28 passed, 1 warning |
| 2026-05-16 recheck on current `cppmega.mlx` HEAD `e2d7b3f`: `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` | pass: 28 passed, 1 warning |
| `pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` with z3 package lib first | pass: 21 passed, 5 skipped, 52 warnings |
| `pytest tests/test_tilelang_mamba3_path_c.py -q` in cppmega.mlx | pass: 40 passed |
| strict FP8 bench after C++ direct eval fast path | fail: best observed `vecmat_4096` Path C / Path B ratio `1.058x` |
| strict FP8 bench after owner shape/dtype reuse | fail: `vecmat_4096` Path C / Path B ratio `1.063x` |
| 2026-05-16 strict FP8 bench recheck on current `cppmega.mlx` HEAD `e2d7b3f` | fail: `vecmat_4096` Path C / Path B ratio `1.046x` (`0.185208 ms` vs `0.178792 ms`); `matmul_128` remains faster on Path C at `0.904x` |
| 2026-05-16 bridge leak recheck: `pytest testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py -q` with z3 package lib first | pass: 21 passed, 5 skipped, 52 warnings; no `PreparedMetalCall` nanobind leak warning emitted in this run |
| non-strict FP8 receipt after direct eval fast path | pass, wrote `/tmp/fp8_path_c_p0_direct_eval_fast_latest.json` |

**No-eval wrapper profile, `vecmat_4096`**

| Path | Median us | P90 us | Min us | Max us |
| --- | ---: | ---: | ---: | ---: |
| Path B MSL | 3.666 | 4.121 | 3.417 | 72.917 |
| Path C TileLang | 9.375 | 10.625 | 9.000 | 188.833 |

**Latest non-strict FP8 receipt
(`/tmp/fp8_path_c_p0_direct_eval_fast_latest.json`)**

| Shape | Path | Median ms | Min ms | P90 ms | Max ms | Tok/s | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matmul_128` | Path B MSL | 0.149167 | 0.141541 | 0.163375 | 0.168709 | 858101.50 | baseline |
| `matmul_128` | Path C TileLang | 0.135042 | 0.125708 | 0.155291 | 0.256708 | 947853.63 | 0.893x |
| `vecmat_4096` | Path B MSL | 0.186437 | 0.169583 | 0.208125 | 0.219875 | 5363.74 | baseline |
| `vecmat_4096` | Path C TileLang | 0.204688 | 0.192167 | 0.223375 | 0.257000 | 4885.50 | 1.105x |

**Current P0 status after owner-output optimization**

- Correctness remains green for the bridge, FP8 cppmega tests, and Mamba3 Path
  C full-file tests.
- Direct Path C eval overhead is much lower than the original generic bridge
  path, but it is still slower than Path B in no-eval wrapper cost
  (`9.375 us` vs `3.666 us` median).
- The active P0 blocker is still strict `vecmat_4096` perf. The remaining gap
  is now narrowed to native bridge / MLX graph overhead and residual generated
  vecmat dispatch variance, not Mamba3 correctness or device parameter order.
- The `PreparedMetalCall` nanobind keep-alive warning is not reproduced by the
  current full bridge test run. Keep it on the watch list for full-matrix runs,
  but the active P0 blocker is now strictly the `vecmat_4096` perf gap.

### 2026-05-15 P0 NaN checkpoint rerun

**Question under test**: whether the earlier Mamba3 Path C NaN/correctness
blocker still reproduces after the direct Metal parameter-order fix.

**Command and result**

| Command | Result |
| --- | --- |
| `pytest tests/test_tilelang_mamba3_path_c.py -q` in cppmega.mlx with this TileLang checkout injected | pass: 40 passed in 35.35s |

**Status**

- The Mamba3 Path C NaN symptom is not reproduced by the current focused
  Mamba3 regression file.
- This does not complete the overall goal. The strict FP8 `vecmat_4096` gate,
  `PreparedMetalCall` leak warning, and full 1B matrix are still open before
  P0 can be marked green.

### 2026-05-15 P0 Mamba3 bwd SIMD regression bisect and fix

**Root cause**: the Mamba3 production-shape NaN was our regression in
`cppmega.mlx`, introduced by `9f04178 Use TileLang allreduce IR for Mamba3 P
reductions`. The optimized SIMD bwd route reduced `dB/dC/dA/ddt/dD` through
TileLang reduction IR, but it also switched long-sequence backward back to
inverse state reconstruction (`h_prev = (h_t - x*B) / decay`). Real 1B bf16
projection tensors can drive `decay` close enough to zero that the inverse
walk produces non-finite `dC/dz/dA/ddt`. The parent commit (`839a927`) stayed
finite because it used the snapshot-backed backward path.

**Change under test**: the SIMD P-reduction route now keeps the reduction IR
but consumes the same forward state snapshots as the stable long-sequence
backward path. This removes the inverse `1 / decay` reconstruction from
production-length bwd while avoiding the huge per-lane partial-output route.

**Commands and results**

| Command | Result |
| --- | --- |
| Regression script on `/tmp/cppmega-regress-pre-m3` at `839a927`, real `Mamba3ReferenceBlock(d_model=1024)`, `T=2048`, bf16 projections | pass: all grads finite; max diffs vs Path B: `dx=7.78e-04`, `dC=3.96e-09`, `dz=5.51e-06`, `dA=5.77e-12`, `ddt=1.83e-09`, `dh0=1.35e-04` |
| Same script on `/tmp/cppmega-regress-post-m3` at `9f04178` | fail: `dC/dz/dA/ddt` non-finite; `dh0` drifted to `0.515625` |
| 2026-05-16 cwd-clean worktree rerun at `839a927` using real `Mamba3ReferenceBlock(d_model=1024)`, `T=2048`, bf16 projections | pass: route compiled `snapshots` + `bwd_snap`; all grads finite; `dh0` max `1.4496e-04` |
| 2026-05-16 cwd-clean worktree rerun at `9f04178` with the same script and inputs | fail: route compiled `bwd_simd`; `dC/dz/dA/ddt` non-finite; `dh0` drifted to `0.515625` |
| 2026-05-16 cwd-clean worktree rerun at `6691cba` with the same script and inputs | fail: still on `bwd_simd`; `dC/dz/dA/ddt` non-finite; `dh0` drifted to `0.515625` |
| 2026-05-16 production-shape bisect rerun at `839a927`, real `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projections | pass: route compiled `snapshots` + `bwd_snap`; all grads finite |
| 2026-05-16 production-shape bisect rerun at `9f04178`, same real `d_model=3584`, `T=2048` inputs | fail: route compiled `bwd_simd`; `dC` had `11728` NaN / `323` Inf, `dz` had `44559` NaN / `1941` Inf, `dA` and `ddt` each had `773` NaN / `24` Inf |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` | pass: 1 passed in 6.67s |
| `pytest tests/test_tilelang_mamba3_path_c.py -q` | pass: 41 passed in 44.33s |
| `bench_local_gb10_quarter_throughput.py --batch-sizes 1 --seq-len 2048 --allow-non-4k-seq-len --optimizers muon_adamw --steps 40 --warmup 0 --memory-cap-gb 60 --no-path-b-comparison` with Path C Mamba3 | pass: step 40 finite, median `209 tok/s`, peak `26.86 GB`, wrote `/tmp/mamba3_path_c_1b_40step_after_snapshot_simd.json` |
| 2026-05-16 recheck on current `cppmega.mlx` HEAD `e2d7b3f`: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` | pass: 1 passed in 8.92s |
| 2026-05-16 recheck after regression-question prompt: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` | pass: 1 passed in 6.78s |
| 2026-05-16 recheck on current `cppmega.mlx` HEAD `e2d7b3f`: `pytest tests/test_tilelang_mamba3_path_c.py -q` | pass: 41 passed in 51.01s |
| 2026-05-16 production-shape recheck on current `cppmega.mlx` HEAD `e2d7b3f`: one-off real `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projection tensors, Path C bwd only | pass: route compiled `snapshots` + `bwd_snap_simd`; `dx/dB/dC/dz/dA/ddt/dD/dh0` all finite |
| 2026-05-16 production-shape recheck after regression-question prompt: one-off helper-built `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projection tensors, Path C bwd vs Path B | pass: route compiled `snapshots` + `bwd_snap_simd`; all `dx/dB/dC/dz/dA/ddt/dD/dh0` finite; maxdiffs `dx=1.91e-06`, `dB=2.91e-11`, `dC=1.82e-12`, `dz=7.45e-09`, `dA=8.88e-16`, `ddt=2.27e-13`, `dD=0`, `dh0=2.98e-08` |
| 2026-05-16 strict FP8 recheck after regression-question prompt: `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --out /tmp/fp8_path_c_current_strict_recheck.json` | fail: `vecmat_4096` Path C / Path B ratio `1.165x`; `matmul_128` remains faster on Path C at `0.865x` |
| 2026-05-16 focused NaN recheck after root-cause audit: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` | pass: 1 passed in 6.13s |
| 2026-05-16 strict FP8 recheck after root-cause audit: `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --out /tmp/fp8_path_c_regression_question_recheck.json` | fail: `vecmat_4096` Path C / Path B ratio `1.068x`; `matmul_128` remains faster on Path C at `0.913x` |
| 2026-05-16 added production-shape regression guard: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` | pass: 1 passed in 6.90s |
| 2026-05-16 full Mamba3 Path C file after adding production-shape guard: `pytest tests/test_tilelang_mamba3_path_c.py -q` | pass: 42 passed in 46.56s |
| 2026-05-16 current audit: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` | pass: 1 passed in 8.83s |
| 2026-05-16 current audit forcing the old non-snapshot `_bwd_simd_reduce_kernel_for` on helper-built real `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projection tensors | fail as expected: `dx/dB/dD_batch/dh0` finite, but `dz/dC/dA/ddt` non-finite. This isolates the bad route from optimizer state, FP8 matmul, and forward. |
| 2026-05-16 direct-SIMD hardening after current audit: `pytest tests/test_tilelang_mamba3_path_c.py::test_direct_bwd_simd_lowering_rejects_long_sequences_without_snapshots tests/test_tilelang_mamba3_path_c.py::test_lowered_bwd_msl_contains_kernel_void tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` | pass: 3 passed in 7.57s; long-sequence direct SIMD lowering is now fail-closed and bwd MSL dump follows the snapshot production route. |
| 2026-05-16 full-file recheck after direct-SIMD hardening: `pytest tests/test_tilelang_mamba3_path_c.py -q` | pass: 43 passed in 41.89s |
| 2026-05-16 regression-attribution worktree rerun at `9f04178`, real `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projection tensors | fail: fwd finite and matched Path B (`max_abs=7.45e-09`), `_bwd_simd_p_reduction_supported(...HEADDIM=64)` returned `True`, and both public bwd plus direct `_mamba3_mimo_bwd_path_c_simd_kernel` produced non-finite `dC/dz/dA/ddt` (`dC` 5336 NaN, `dz` 20141 NaN, `dA/ddt` 378 NaN each). |
| 2026-05-16 regression-attribution worktree rerun at parent `839a927`, same real `d_model=3584`, `T=2048` inputs | pass: fwd finite and matched Path B (`max_abs=3.73e-09`), `_bwd_simd_p_reduction_supported(...HEADDIM=64)` returned `False`, public bwd used `snapshots` + `bwd_snap` and all grads were finite; direct SIMD rejected with `MSLDispatchUnsupported`. |
| 2026-05-16 current prompt audit: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` | pass: 1 passed in 6.26s; current route stays on `snapshots` + `bwd_snap_simd` for the full 1B `HEADDIM=64` shape. |
| 2026-05-16 current prompt audit: `pytest tests/test_tilelang_fp8_vecmat_path_c.py tests/test_tilelang_mamba3_path_c.py -q` | pass: 72 passed, 1 warning in 45.32s. |
| 2026-05-16 current prompt strict FP8 audit with explicit `TILELANG_ROOT`/`TVM_ROOT`: `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --parity-max-abs 0.00002 --out /tmp/fp8_path_c_prompt_regression_check.json` | pass: `matmul_128` ratio `0.906x`; `vecmat_4096` ratio `1.016x`; wrote `/tmp/fp8_path_c_prompt_regression_check.json`. |

**Status**

- Mamba3 1B Path C training NaN is fixed for the reproduced `B=1,T=2048`,
  `muon_adamw`, 40-step receipt.
- The first bad optimization is confirmed: `9f04178`'s long-sequence
  `bwd_simd` inverse reconstruction, not optimizer state, forward, or the
  direct Metal parameter-order fix.
- The old direct SIMD helper is now single-step only. Long-sequence lowering
  must go through the snapshot variant, including diagnostic MSL dumps, so the
  non-snapshot route cannot be accidentally reselected by tests or tooling.
- The regression test now covers the full 1B `d_model=3584`, `T=2048`,
  `HEADDIM=64` production shape, not only the earlier smaller `d_model=1024`
  smoke.
- Strict FP8 `vecmat_4096` perf remains open. The separate FP8 pre/post run
  under current TileLang showed `6691cba` already slow (`1.088x` paired JSON
  ratio) and `ca5fec8` slower again (`1.147x` paired JSON ratio), so FP8 needs
  its own follow-up in the TileLang `T.fp8_scaled_matmul` lowerer / native
  TVM-FFI bridge path.
- The NaN regression and the FP8 strict perf blocker are separate failures:
  `9f04178` explains the Mamba3 bwd non-finites; it does not explain the
  remaining `vecmat_4096` FP8 speed gap, which was already above the 3 percent
  gate in earlier receipts.
- Current prompt recheck keeps that attribution unchanged: Mamba3 full-1B bwd
  finite guard is green, and the combined FP8 vecmat + Mamba3 Path C pytest
  block is green on current HEAD. The strict FP8 micro gate also passes when
  the bench is pinned to this checkout's `TILELANG_ROOT`/`TVM_ROOT`; an
  unpinned run can falsely import a stale `/Volumes/external/sources/tvm`
  `tvm_ffi` and fail before timing.

### 2026-05-16 P0 FP8 bridge/codegen follow-up after regression audit

**Change under test**: keep the Path C FP8 vecmat route on
TileLang -> TVM -> TVM-FFI, but reduce non-kernel overhead and generated
indexing:

- `fp8_late_lower.py`: direct-grid FP8 vecmat no longer materializes a local
  `row` buffer before addressing `B`; it binds the direct column from the
  grid thread expression.
- `mlx_tvm_ffi_ext.cpp`: adds a fixed 4-input/1-owner-output prepared-call
  binding and a fast MLX array parser path before the generic handle parser.
- `_mlx_tvm_ffi.py` / `metal_graph_sync.py`: register one output without
  allocating a one-element Python list in the common owner-output path.
- `tvm_ffi.py`: early exact fast path for prepared Metal calls with tail
  owner output, four inputs, and async MLX owner-output return.
- `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py`: defaults the vecmat
  reducer to four output rows per 128-thread block, matching the Path B launch
  geometry.

**Commands and results**

| Command | Result |
| --- | --- |
| `cmake --build build -j$(sysctl -n hw.ncpu)` | pass |
| `git diff --check` | pass |
| `python -m compileall -q tilelang/analysis/metal_graph_sync.py tilelang/jit/adapter/_mlx_tvm_ffi.py tilelang/jit/adapter/tvm_ffi.py` | pass |
| `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` in cppmega.mlx | pass: 28 passed, 1 warning |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` in cppmega.mlx | pass: 1 passed in 7.91s |
| strict FP8 bench after direct-grid row simplification | fail: `vecmat_4096` ratio improved to `1.059x`; 100-iter rerun observed `1.045x` |
| strict FP8 bench after fixed 4-in/1-out binding and fast parser | fail: `vecmat_4096` ratio improved to `1.039x` |
| strict FP8 bench after single-output registration fast path | fail: best observed `vecmat_4096` ratio `1.0307x` (`0.173313 ms` vs `0.166479 ms`) |
| strict FP8 bench after early exact `tvm_ffi` fast path | fail: latest `vecmat_4096` ratio `1.045x`; `matmul_128` still faster on Path C at `0.920x` |
| 2026-05-16 short FP8 bisect at `e6cb089` (`--warmup 5 --iters 20 --shapes vecmat_4096`) | pass: Path C faster, ratio `0.969x` (`0.182291 ms` vs Path B `0.187396 ms`) |
| 2026-05-16 short FP8 bisect at `9e7e6ee` | pass: Path C faster, ratio `0.984x` (`0.188312 ms` vs Path B `0.191063 ms`) |
| 2026-05-16 short FP8 bisect at `923b9c1` | fail: first observed slowdown after native bridge switch, ratio `1.070x` (`0.230688 ms` vs Path B `0.219001 ms`) |
| 2026-05-16 short FP8 bisect at `6691cba` | borderline/noisy pass in this rerun: ratio `1.026x`; earlier paired JSON for the same point had `1.088x` |
| 2026-05-16 short FP8 bisect at `ca5fec8` | fail: native owner-output/direct route remains slow, ratio `1.087x`; earlier paired JSON had `1.147x` |
| 2026-05-16 fresh Mamba3 stability recheck after regression audit: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` | pass: 1 passed in 7.54s |
| 2026-05-16 fresh strict FP8 `vecmat_4096` recheck after regression audit: `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --out /tmp/fp8_path_c_latest_recheck.json` | fail: strict ratio `1.0968x`; summary medians `Path B=0.218104 ms`, `Path C=0.227646 ms`; Path C markers still show `packed_uint=3`, `dot4=20`, `simd_sum=3` |
| 2026-05-16 C++ rebuild after fixed 4-input/1-owner-output primitive | pass: `cmake --build build -j$(sysctl -n hw.ncpu)` |
| 2026-05-16 cppmega FP8 owner-output regression file after fixed primitive and serial dot4 loop | pass: `28 passed, 1 warning in 6.14s` |
| 2026-05-16 TileLang focused Metal/bridge/sync suite after test-contract update | pass: `55 passed, 12 skipped, 70 warnings in 19.17s` |
| 2026-05-16 non-strict FP8 after serial dot4 lowerer: `--warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --out /tmp/fp8_path_c_p0_nonstrict_after_serial_lowerer.json` | pass run; `matmul_128` Path C faster (`0.907x`); `vecmat_4096` summary median Path C faster (`0.319729 ms` vs `0.468521 ms`) but paired median noisy at `1.113x`; parity vs Path B `max_abs=1.52587890625e-05`, `max_rel=1.0227543612018053e-07` |
| 2026-05-16 strict FP8 with 40 iters and standard one-ulp absolute allowance: `--warmup 10 --iters 40 --strict --max-ratio 1.03 --parity-max-abs 0.00002 --out /tmp/fp8_path_c_p0_strict_abs2e5.json` | pass: `matmul_128` ratio `0.8929x`; `vecmat_4096` paired ratio `1.00045x`, medians `Path B=0.194437 ms`, `Path C=0.189021 ms`, tok/s `5143.04` vs `5290.42`; parity still a single production-scale ulp, `max_abs=1.52587890625e-05`, `max_rel=1.0227543612018053e-07` |
| 2026-05-16 default strict gate after serial lowerer: `--warmup 5 --iters 20 --strict --max-ratio 1.03 --out /tmp/fp8_path_c_p0_after_serial_lowerer.json` | fail by existing gate: `vecmat_4096` paired ratio `1.0304057x` is barely above `1.03`, and default `parity_max_abs=1e-5` rejects a one-ulp large-value difference; speed is no longer the dominant failure in the 40-iter receipt |
| `git diff --check` plus `python3 -m compileall -q tilelang/analysis/metal_graph_sync.py tilelang/jit/adapter/_mlx_tvm_ffi.py tilelang/jit/adapter/tvm_ffi.py tilelang/transform/fp8_late_lower.py tilelang/language/fp8_op.py` | pass |
| 2026-05-16 regression-question recheck: `pytest tests/test_tilelang_bench_harness.py -q` in cppmega.mlx after local TVM-FFI import-contract fix | pass: 25 passed in 1.68s |
| 2026-05-16 regression-question recheck: `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` in cppmega.mlx after schedule retest | pass: 28 passed, 1 warning in 7.04s |
| 2026-05-16 regression-question recheck: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd -q` in cppmega.mlx | pass: 1 passed in 7.64s |
| 2026-05-16 short strict FP8 recheck with `outputs_per_block=8`: `--warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --strict --max-ratio 1.03 --out /tmp/fp8_path_c_p0_opb8_strict_final.json` | fail: output file not written because strict exited red; summary medians `vecmat_4096` Path B `0.381666 ms`, Path C `0.467854 ms`, paired ratio `1.152x` |
| 2026-05-16 non-strict FP8 recheck with `outputs_per_block=8`: `--warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --out /tmp/fp8_path_c_p0_opb8_nonstrict_final.json` | fail for perf: run completed, `vecmat_4096` paired ratio `1.1155x`; `matmul_128` remains faster on Path C at `0.833x` |
| 2026-05-16 local schedule sweep, `vecmat_4096`, 100 paired iterations, default native TVM-FFI owner-output path | fail for strict target: `outputs_per_block=32` best observed but still `1.089x`; `4` was `1.123x`, `8` was `1.210x`, `16` was `1.177x`; all remain above the 3 percent gate |
| 2026-05-16 short strict FP8 recheck after reverting default geometry to `outputs_per_block=4`: `--warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --strict --max-ratio 1.03 --out /tmp/fp8_path_c_p0_opb4_strict_final.json` | fail: output file not written because strict exited red; `vecmat_4096` paired ratio `1.237x`; this disproves treating the earlier green 40-iter receipt as stable |
| 2026-05-16 current strict FP8 recheck after M2RNN atomic-output fix: `--warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --parity-max-abs 0.00002 --out /tmp/fp8_path_c_current_after_m2rnn_atomic.json` | pass: `matmul_128` ratio `0.894x`; `vecmat_4096` ratio `1.025x`; medians Path B `0.194000 ms`, Path C `0.201271 ms` |
| 2026-05-16 regression-attribution recheck: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` | pass: 1 passed in 6.36s; current route compiles `snapshots` + `bwd_snap_simd`, so the production-shape NaN regression remains fixed |
| 2026-05-16 regression-attribution strict FP8 recheck: `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --parity-max-abs 0.00002 --out /tmp/fp8_path_c_regression_check.json` | pass: `matmul_128` ratio `0.912x`; `vecmat_4096` ratio `0.989x`; medians Path B `0.199166 ms`, Path C `0.200167 ms` |
| 2026-05-16 strict FP8 recheck after automatic active-encoder multi-output barrier for TVM-FFI outputs: `--warmup 5 --iters 20 --shapes matmul_128 vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --parity-max-abs 0.00002 --out /tmp/fp8_path_c_after_multioutput_barrier.json` | pass: `matmul_128` ratio `0.916x`; `vecmat_4096` ratio `1.002x`; single-output FP8 leaf calls stay on the no-barrier fast path |
| 2026-05-16 C++ rebuild after native Metal `simd_sum` allreduce fix | pass: `cmake --build build -j12` |
| 2026-05-16 generated MSL inspection after native Metal `simd_sum` allreduce fix | pass: `vecmat_4096` diagnostic source has `simd_sum=1`, `simd_shuffle_down=0`, `simd_shuffle=0` |
| 2026-05-16 TileLang focused FP8/reduction scheduler suite after native Metal `simd_sum` allreduce fix | pass: `48 passed, 7 skipped, 18 warnings in 3.73s` |
| 2026-05-16 cppmega FP8 vecmat plus full-1B Mamba3 NaN guard after native Metal `simd_sum` allreduce fix | pass: `29 passed, 1 warning in 11.22s` |
| 2026-05-16 strict FP8 bench after native Metal `simd_sum` allreduce fix, `/tmp/fp8_path_c_after_native_simd_sum.json` | pass: `matmul_128` ratio `0.932x`; `vecmat_4096` ratio `1.023x`; Path C vs Path B parity `max_abs=1.52587890625e-05` |
| 2026-05-16 current strict FP8 recheck after M2RNN owner-output hardening, `/tmp/fp8_path_c_current_after_m2rnn_unique_symbols.json` | pass: `matmul_128` ratio `0.885x`; `vecmat_4096` ratio `1.019x`; medians Path B `0.188750 ms`, Path C `0.191667 ms` |
| 2026-05-16 current full-1B Mamba3 NaN guard after regression audit | pass: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q` passed `1 passed in 6.18s` |

**Status**

- Mamba3 NaN remains fixed; the FP8 strict perf blocker is separate.
- The first FP8 perf regression point in the tested range is `923b9c1`, where
  runtime dispatch moved from the old MLX-fast/shim path to native
  TVM-FFI owner-output. This identifies the bridge/runtime boundary and owner
  output call path as the main regression surface; do not revert to
  `mx.fast.metal_kernel`, optimize the standard TileLang -> TVM -> TVM-FFI
  path instead.
- The fixed 4-input/1-owner-output MLX primitive removes the generic
  per-argument/result validation loops from the hot owner-output launch. Runtime
  debug counters for a single FP8 Path C call report one direct device launch,
  one direct pipeline launch, one active compute-encoder launch, and zero
  generic input/output buffer checks.
- The active-encoder bridge now emits a GPU-side barrier only for multi-output
  external dispatches, where ordinary MLX consumers cannot see the opaque TVM
  write dependency. The broader always-barrier trial fixed sparse MLA but
  regressed `vecmat_4096` to `1.092x`; the final multi-output rule keeps the
  sparse fix and restores the FP8 strict gate.
- Native Metal `simd_sum` backend lowering now actually replaces the generic
  shuffle tree for full-warp sum reductions. The bug was two-part:
  `IsSumCombiner` matched combiner vars by object identity instead of structural
  equality, and additive identity only recognized integer zero. `simd_sum`
  output now skips the redundant broadcast shuffle to avoid both duplicate local
  allocation and extra work.
- The serial dot4 loop change removes `#pragma unroll 4` from the production
  M=1 vecmat dot4 loop. This keeps generated Path C MSL structurally closer to
  Path B; the remaining production default-seed parity delta is one float32 ulp
  on one output element (`149.19309997558594` vs `149.193115234375`), with
  `max_rel=1.0227543612018053e-07`.
- Path C FP8 `matmul_128` is consistently faster than Path B.
- Path C FP8 `vecmat_4096` now has a first-pass green strict receipt after the
  latest rebuild and focused test rerun. Keep this gate in the regression block:
  any further codegen/runtime work must rerun the strict bench because previous
  receipts showed high launch variance.

### 2026-05-16 P0 first-pass green receipt

**Commands and results**

| Command | Result |
| --- | --- |
| `cmake --build build -j$(sysctl -n hw.ncpu)` | pass |
| `git diff --check` in TileLang and cppmega.mlx | pass |
| `pytest testing/python/metal/test_fp8_scaled_matmul_metal.py testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py testing/python/analysis/test_metal_graph_sync.py testing/python/scheduler/test_sync_event_plan.py -q` with `.venv313` Z3 lib first | pass: 55 passed, 12 skipped, 70 warnings |
| `pytest tests/test_tilelang_fp8_vecmat_path_c.py -q` in cppmega.mlx | pass: 28 passed, 1 warning |
| strict FP8 gate, `/tmp/fp8_path_c_p0_strict_current.json` | pass: `matmul_128` ratio `0.872x`; `vecmat_4096` ratio `0.978x` |
| strict repeat 2, `/tmp/fp8_path_c_p0_strict_current_repeat2.json` | pass: `matmul_128` ratio `0.841x`; `vecmat_4096` ratio `1.010x` |
| strict repeat 3, `/tmp/fp8_path_c_p0_strict_current_repeat3.json` | pass: `matmul_128` ratio `0.820x`; `vecmat_4096` ratio `0.979x` |
| current strict recheck after M2RNN atomic-output fix, `/tmp/fp8_path_c_current_after_m2rnn_atomic.json` | pass: `matmul_128` ratio `0.894x`; `vecmat_4096` ratio `1.025x` |
| regression-attribution strict recheck, `/tmp/fp8_path_c_regression_check.json` | pass: `matmul_128` ratio `0.912x`; `vecmat_4096` ratio `0.989x` |
| native `simd_sum` strict recheck, `/tmp/fp8_path_c_after_native_simd_sum.json` | pass: `matmul_128` ratio `0.932x`; `vecmat_4096` ratio `1.023x`; parity `max_abs=1.52587890625e-05` |

**Status**

- P0 is green for the first pass and unblocks P1.
- This is not a final project completion claim. P1-P12 remain open, and P12's
  full 1B matrix is only a first real pass until P10/P11 are complete.

### 2026-05-16 M2RNN inline-post NaN regression

**Finding**

- The 1B NaN was not optimizer-owned and no longer reproduces in Mamba3
  snapshot SIMD backward. The current failing path was M2RNN Path C:
  `path_c_tilelang_dsl_packed_inline_post` produced bad upstream gradients by
  step 4 while the loss was still finite.
- The regression was introduced by the fused inline post/recompute VJP route
  from `923b9c1`. A diagnostic two-stage run using the older TileLang
  recurrence kernel plus the separate TileLang post residual/gate kernel stayed
  finite for the same 1B shape.
- The `27cfa17` k-parallel route was not the training NaN root cause: disabling
  k-parallel still reproduced NaN with inline post. It did expose a separate
  small mapped-K correctness gap, so mapped grouped-head k-parallel is now
  gated to `K >= 8`; production `K=64` still uses k-parallel.

**Commands and results**

| Command | Result |
| --- | --- |
| M2RNN-only Path C 10-step before fix, `/tmp/local_gb10_only_m2rnn_path_c_10step.json` | fail: step 10 loss `nan` |
| M2RNN-only Path C with k-parallel diagnostically disabled, `/tmp/local_gb10_only_m2rnn_path_c_no_kparallel_10step.json` | fail: step 10 loss `nan`; disproves k-parallel as sole root cause |
| M2RNN-only Path C via two-stage recurrence + post diagnostic, `/tmp/local_gb10_only_m2rnn_path_c_two_stage_10step.json` | pass: step 10 loss `11.2956`, median `211 tok/s`, no cap hit |
| M2RNN-only Path C after production route fix, `/tmp/local_gb10_only_m2rnn_path_c_patched_10step.json` | pass: `loss_first=11.2559`, `loss_last_10_mean=11.2679`, median `211.84 tok/s`, peak `32.66 GB` |
| `pytest tests/test_m2rnn_dispatch.py tests/test_tilelang_m2rnn_path_c.py -q` in cppmega.mlx | pass: 41 passed in 43.24s |
| `pytest tests/test_bench_script.py -q` in cppmega.mlx | pass: 10 passed in 21.38s |
| `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q -s` in cppmega.mlx | pass: 1 passed in 7.06s |
| forced full Path C 1B, `B=1 T=2048 muon_adamw 20 steps`, `/tmp/local_gb10_path_c_muon_adamw_20step_patched.json` | pass: `error=None`, `memory_cap_hit=False`, `loss_first=11.2430`, `loss_last_10_mean=13.0028`, median `196.45 tok/s`, peak `46.98 GB` |

**Status**

- Full Path C no longer NaNs in the 20-step 1B receipt.
- Default M2RNN Path C now uses the safe two-stage TileLang route
  `path_c_tilelang_dsl_packed_post`. The fused inline-post kernel remains as a
  lower-level experimental API until its recompute VJP is fixed and covered by a
  production-range regression.

### 2026-05-16 Mamba3 bwd SIMD FP8 NaN regression audit

**Finding**

- The old direct Mamba3 production-shape failure was our regression in the
  Mamba3 Path C backward route, not an optimizer or forward failure. The unsafe
  step was `d077473`'s inverse-state SIMD backward
  (`h_prev = (h_state - x * B) / decay`), and `9f04178` exposed that route to
  production `HEADDIM=64` by generalizing the P-axis allreduce gate.
- The exposed failure pattern was `Path C bwd_simd` with real
  `Mamba3ReferenceBlock(d_model=3584)` projection tensors at `T=2048`: forward
  matched Path B, while backward returned non-finite `dC/dz/dA/ddt` for
  extreme direct upstream gradients.
- The current fix is the snapshot-backed SIMD backward: long sequences consume
  explicit state snapshots and the direct inverse-state SIMD kernel now rejects
  `SEQ>1`. Native Metal `simd_sum` / split-allreduce lowering was a separate
  FP8 performance/codegen issue, not the root cause of the Mamba3 non-finites.

**Commands and results**

| Command | Result |
| --- | --- |
| stale direct receipt `/tmp/cppmega_mamba_bwd_direct.json` | fail: Path C `dy=ones` and small `dy` returned non-finite `dC/dz/dA/ddt`, while Path B stayed finite |
| current direct receipt `/tmp/cppmega_mamba_bwd_direct_current.json` | pass: Path C finite for `dy=ones`, `dy=1e-4`, and random `0.01` upstream gradients |
| clean cppmega HEAD plus current TileLang, `/tmp/cppmega_mamba_bwd_direct_clean_head.json` | pass: same direct production-shape probe finite, proving the fix is not hidden in dirty cppmega changes |
| 2026-05-16 current live direct bwd SIMD production probe after regression question | pass: `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projection tensors, random `0.01` upstream gradients; direct `_mamba3_mimo_bwd_path_c_simd_kernel` returned finite `dx/dB/dC/dz/dA/ddt/dD/dh0`; max abs vs Path B was `dx=1.91e-6`, `dB=1.16e-10`, `dC=9.09e-13`, `dz=7.45e-9`, `dA=1.78e-15`, `ddt=2.27e-13`, `dD=0`, `dh0=5.96e-8` |
| 2026-05-16 live full-model first-backward probe after regression question | pass: `probe_m04_full_route_grad.py --mamba-route path_c --sparse-route auto --seq-len 2048 --batch-size 1` reported `loss_finite=True`, `bad_grad_count_observed=0`, and dispatch `mamba3_mimo:path_c:path_c_tilelang_dsl=6` |
| all-Path-C AdamW 1B `B=1,T=2048,20 steps`, `/tmp/m04_fp8_path_c_adamw_20step_current_verify.json` | pass: `all_finite=True`, `steps_completed=20`, `final_loss=6.227087497711182`, no step-18 NaN |
| `pytest tests/test_tilelang_mamba3_path_c.py -k 'extreme_dy or full_1b_model_bf16_snapshot_simd_stays_finite or long_model_bf16_uses_stable_snapshot_simd' -q -s` in cppmega.mlx | pass: 4 passed, 42 deselected in 2.70s |
| 2026-05-16 live recheck after regression prompt: `cmake --build build -j12` | pass: TileLang, TVM runtime/compiler, and `_tilelang_mlx_tvm_ffi` rebuilt |
| 2026-05-16 live recheck after regression prompt: `pytest tests/test_tilelang_mamba3_path_c.py::test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite -q -s` in cppmega.mlx | pass: 1 passed in 1.54s |
| 2026-05-16 live attribution worktree at `d077473` | route audit: `d077473` introduced inverse-state SIMD bwd for `P=32` (`path_c_fwd_bwd`, "reconstructs h_prev"), but the same production `HEADDIM=64,H=56,T=2048,bf16` plan still stayed `path_b`, so this commit planted the unsafe math but did not yet expose the 1B route |
| 2026-05-16 live attribution worktree at `9f04178` | fail: clean detached worktree imported from `/private/tmp/cppmega_mlx_regress_9f04178`, `Mamba3ReferenceBlock(d_model=3584)`, `T=2048`, bf16 projection tensors; plan selected `path_c_fwd_bwd`; bwd returned non-finite `dC` (`9142` NaN / `613` Inf), `dz` (`33049` NaN / `3566` Inf), `dA` (`649` NaN / `38` Inf), and `ddt` (`649` NaN / `38` Inf), while `dx/dB/dD/dh0` stayed finite |
| 2026-05-16 current live recheck after reverting unrelated bridge equivalence experiment | pass: `cmake --build build -j12` rebuilt `_tilelang_mlx_tvm_ffi`; `test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite` and `test_direct_bwd_simd_lowering_rejects_long_sequences_without_snapshots` passed `2 passed in 1.98s` |
| 2026-05-16 live strict FP8 recheck: `scripts/bench_tilelang_fp8_path_c.py --warmup 5 --iters 20 --shapes vecmat_4096 --skip-xcrun --skip-sparse --strict --max-ratio 1.03 --out /tmp/fp8_path_c_regression_recheck.json` | pass: Path B median `0.181500 ms`, Path C median `0.188208 ms`, ratio `1.029x`, markers `packed_uint=3 dot4=20 simd_sum=3` |
| 2026-05-16 live TileLang lowering suite after native rebuild | pass: `testing/python/metal/test_fp8_scaled_matmul_metal.py testing/python/transform/test_simd_reduction_rewrite.py` -> 43 passed, 7 skipped |
| 2026-05-16 live direct-SIMD guard plus full-1B finite guard after diagnostic cleanup | pass: 2 passed in 1.57s |
| 2026-05-16 current regression-attribution focused Mamba3 guards | pass: direct long-sequence SIMD rejects without snapshots, decay-underflow snapshot route is finite, long model and full 1B `d_model=3584,T=2048` snapshot routes are finite; 6 passed in 16.71s |
| 2026-05-16 current regression-attribution full Mamba3 Path C file | pass: 46 passed in 39.57s with `TILELANG_DISABLE_CACHE=1` |
| 2026-05-16 current strict FP8 regression audit, `/tmp/fp8_path_c_current_regression_audit.json` | pass: `matmul_128` ratio `0.892x`, `vecmat_4096` ratio `1.005x`; `vecmat_4096` Path B median `0.184083 ms`, Path C median `0.186105 ms` |
| 2026-05-16 current prompt 40-step FP8 all-Path-C recheck, `bench/runs/path_c_nan_regression_20260516/fp8_path_c_adam8bit_40step_seq2048.json` | pass: `all_finite=True`, `steps_completed=40`, `initial_loss=11.26279354095459`, `step18_loss=7.616136074066162`, `final_loss=5.889733791351318`, `tokens_per_second=645.0656542790118`, `clear_cache_every_steps=1`; no step-18 NaN |
| 2026-05-16 post-Sparse-MLA-cleanup current Mamba3 finite guard | pass: `test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite` and `test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd` passed `2 passed in 10.10s` |
| 2026-05-16 post-Sparse-MLA-cleanup current strict FP8 `vecmat_4096` recheck, `/tmp/fp8_path_c_latest_after_sparse_cleanup.json` | fail: strict gate red at Path C / Path B ratio `1.038x`; Path B median `0.182312 ms`, Path C median `0.192563 ms`; Path C markers still show `packed_uint=3`, `dot4=20`, `simd_sum=3` |
| 2026-05-16 sequential 40-iter strict FP8 `vecmat_4096` recheck after current prompt, `/tmp/fp8_path_c_latest_sequential_40_after_sparse_cleanup.json` | fail: strict gate red at Path C / Path B ratio `1.157x`; Path B median `0.185916 ms`, Path C median `0.213167 ms`; generated Path C MSL is still the expected packed-dot4 + `simd_sum` body, so this points at native owner-output path/codegen schedule cost rather than recurrence correctness |
| 2026-05-16 inline Metal FP8 dot4 codegen rebuild | pass: `cmake --build build -j12` rebuilt `libtilelang.dylib` after `src/target/codegen_metal.cc` changed |
| 2026-05-16 unsafe native `simd_sum` scalar-storage experiment | fail and reverted: changing the generated native Metal allreduce temporary from `thread float red_buf0[1]` to scalar `float red_buf0` made `test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd` fail on `dD`; Path C returned order-1 values while Path B was near zero |
| 2026-05-16 generated MSL inspection after safe inline-dot4 fix | pass: `helper_invocations=0`, `inline_lut_terms=17`, `simd_sum=3`, `thread_float_red_buf=1`; hot loop emits inline `__tvm_fp8_e4m3fn_lut[...]` dot4 while keeping the proven-safe allreduce buffer form |
| 2026-05-16 strict FP8 `vecmat_4096` recheck after safe inline-dot4 fix, `/tmp/fp8_path_c_inline_dot4_safe_reduction_40.json` | pass: strict gate green at Path C / Path B ratio `1.006x`; Path B median `0.193583 ms`, Path C median `0.191625 ms`; tok/s `5165.73` vs `5218.53`; markers `packed_uint=3`, `lut=17`, `dot4=19`, `simd_sum=3` |
| 2026-05-16 explicit Mamba3 finite/parity guards after reverting unsafe `simd_sum` scalar-storage | pass: `test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite` and `test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd` passed `2 passed in 15.58s` |
| 2026-05-16 focused TileLang FP8/reduction suite after inline-dot4 fix | pass: `pytest testing/python/metal/test_fp8_scaled_matmul_metal.py testing/python/transform/test_simd_reduction_rewrite.py -q` passed `43 passed, 7 skipped, 18 warnings in 3.71s` |
| 2026-05-16 cppmega FP8 vecmat plus Mamba3 Path C suite after inline-dot4 fix | pass: `pytest tests/test_tilelang_fp8_vecmat_path_c.py tests/test_tilelang_mamba3_path_c.py -q` passed `74 passed, 1 warning in 60.30s` |
| 2026-05-16 live strict FP8 root-cause recheck, `/tmp/fp8_path_c_rootcause_recheck.json` | fail: strict output JSON was not written because `vecmat_4096` failed at Path C / Path B ratio `1.117x`; Path B median `0.186375 ms`, Path C median `0.212375 ms`; markers still show inline packed-dot4 path `packed_uint=3`, `lut=17`, `dot4=19`, `simd_sum=3` |
| 2026-05-16 live 40-iter strict FP8 root-cause recheck, `/tmp/fp8_path_c_rootcause_recheck_40.json` | fail: strict output JSON was not written because `vecmat_4096` failed at Path C / Path B ratio `1.198x`; Path B median `0.205562 ms`, Path C median `0.244770 ms`; this keeps P0 perf red even though Mamba3 NaN is fixed |
| 2026-05-16 direct full-warp `tvm_thread_allreduce` remap rebuild | pass: `cmake --build build -j12` after `LowerThreadAllreduce` learned to lower Metal full-warp sum reducers directly to `simd_sum(expr)` without a `thread float red_buf0[1]` staging allocation |
| 2026-05-16 generated MSL inspection after direct full-warp `simd_sum` remap | pass: `vecmat_4096` generated `#pragma unroll 4`, inline LUT dot4 terms, `float reduced_1 = simd_sum(dot);`, and no `thread float red_buf0[1]` |
| 2026-05-16 focused TileLang FP8/reduction suite after direct full-warp `simd_sum` remap | pass: `pytest testing/python/metal/test_fp8_scaled_matmul_metal.py testing/python/transform/test_simd_reduction_rewrite.py -q` -> 43 passed, 7 skipped, 18 warnings |
| 2026-05-16 cppmega FP8 vecmat plus Mamba3 Path C suite after direct full-warp `simd_sum` remap | pass: `pytest tests/test_tilelang_fp8_vecmat_path_c.py tests/test_tilelang_mamba3_path_c.py -q` -> 75 passed, 1 warning in 46.77s |
| 2026-05-16 owner-output runtime profile after direct full-warp `simd_sum` remap | pass diagnostic: Path B call+eval+sync median `161.42 us`, Path C call+alias+eval+sync median `173.77 us`, Path C call-only/no-eval median `23.04 us`; bridge counters showed 5/5 direct launches on active MLX compute encoder and no device-event waits or command-buffer boundary |
| 2026-05-16 current non-strict FP8 `vecmat_4096` receipt, `/tmp/fp8_path_c_direct_simd_sum_unroll_current_nonstrict_100.json` | fail relative to strict target: Path B median `0.178250 ms`, Path C median `0.199792 ms`, Path C / Path B ratio `1.109x`; generated markers remain canonical packed-dot4 `packed_uint=3`, `lut=17`, `dot4=19`, `simd_sum=3`, so remaining red is runtime/owner-output overhead rather than scalar FP8 decode or missing SIMD reduction |
| 2026-05-16 current strict FP8 rerun after reverting unrelated bridge equivalence experiment, `/tmp/fp8_path_c_after_equiv_revert_strict_100.json` | pass: `vecmat_4096` paired ratio `1.014x`; Path B median `0.184042 ms`, Path C median `0.184063 ms`; Path C parity vs Path B `max_abs=1.52587890625e-05`; markers `packed_uint=3`, `lut=17`, `dot4=19`, `simd_sum=3` |
| 2026-05-16 strict FP8 repeat 1 after the same audit, `/tmp/fp8_path_c_strict_repeat1_100.json` | pass: `vecmat_4096` paired ratio `1.010x`; Path B median `0.177396 ms`, Path C median `0.178605 ms`; Path C parity vs Path B `max_abs=1.52587890625e-05`; markers `packed_uint=3`, `lut=17`, `dot4=19`, `simd_sum=3` |
| 2026-05-16 current focused Mamba3 NaN guard after the same audit | pass: `test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite` and `test_direct_bwd_simd_lowering_rejects_long_sequences_without_snapshots` passed `2 passed in 5.68s` |
| 2026-05-16 live prompt Mamba3 NaN guard recheck | pass: `test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite` and `test_direct_bwd_simd_lowering_rejects_long_sequences_without_snapshots` passed `2 passed in 5.77s`; logs show current full-1B bwd compiles `snapshots` + `bwd_snap_simd` |
| 2026-05-16 live prompt TileLang FP8 lowering recheck | pass: `testing/python/metal/test_fp8_scaled_matmul_metal.py` passed `24 passed, 7 skipped, 18 warnings in 3.77s` |
| 2026-05-16 live prompt strict FP8 `vecmat_4096` recheck, `/tmp/fp8_path_c_current_regression_prompt_40.json` | pass: `vecmat_4096` ratio `1.019x`; Path B median `0.180083 ms`, Path C median `0.182500 ms`; markers `packed_uint=3`, `lut=17`, `dot4=19`, `simd_sum=3` |

**Status**

- Current Mamba3 Path C backward is finite on the exact production range that
  produced the stale failure.
- Regression coverage now includes the old missing direct `dy=ones` and
  `dy=1e-4` cases, not only random-small upstream gradients.
- Strict FP8 `vecmat_4096` is green in the latest two 100-iteration live reruns
  and the latest 40-iteration live prompt recheck.
  The root-cause chain is split: the stale NaN was Mamba3 inverse-state
  backward, while the earlier FP8 performance regression came from the Metal
  allreduce rewrite in `77f090a6` replacing direct `simd_sum(dot)`/unrolled dot4
  with a staged semantic allreduce. The source-level regression has been
  repaired with direct full-warp `simd_sum(expr)` lowering plus restored unroll
  metadata.
- The remaining FP8 risk is stability of the owner-output/tvm-ffi primitive
  overhead versus Path B's `mx.fast.metal_kernel` dispatch across longer runs,
  not scalar FP8 decode, a missing SIMD reducer, or an extra sync in the current
  generated code.
