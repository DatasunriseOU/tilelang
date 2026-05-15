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
export TILELANG_DISABLE_CACHE=1
export TILELANG_DEV_BUILD_ROOT="$TL_ROOT"
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
