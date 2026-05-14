# ML Optimization Plan

Date: 2026-05-14

Scope: make TileLang scheduler and IR builder emit semantic optimization IR,
then let TileLang/TVM/TVM-FFI/MLX lower, prove, schedule, and tune it. Model
repos such as `cppmega_mlx` may contain regression fixtures and benchmarks, but
they must not own backend-specific lowering decisions.

## Goals

- TileLang model code describes reductions, scans, dependencies, aliases, and
  materialization requirements as IR metadata, not as hand-written backend
  partial buffers or Metal/CUDA intrinsics.
- Scheduler chooses a lowering from semantic IR using rule checks, Z3 legality
  proofs, and profiling feedback.
- Synchronization is inserted only for proven hazards. If Z3 proves no
  inter-buffer or intra-buffer hazard, no barrier/event is emitted.
- Supported shapes do not return `*_partial` tensors to Python/MLX unless the
  requested public API explicitly asks for debug partials.
- Path C reaches Path B performance class through scheduler/codegen
  improvements, not by reintroducing model-specific MSL or monkeypatches.

## Non-Goals

- No direct cppmega-to-TVM-FFI bypass as a public model path.
- No `mx.fast.metal_kernel` or handwritten MSL for production Path C kernels.
- No Python monkeypatches for production functionality.
- No hidden large allocations or host-side reductions in model training paths.

## Current Baseline

Already landed:

- TileLang exposes `T.thread_allreduce_sum(...)` and emits
  `tir.tvm_thread_allreduce`.
- Mamba3 Path C bwd uses semantic P-axis allreduce for aligned P dimensions.
- Regression coverage verifies P=32, P=64, P=96, P=128, and P=256 lower without
  `dA_partial`, `dB_partial`, `dC_partial`, `dD_partial`, or `ddt_partial`.

Remaining gap: this is still one explicit IR helper callsite in a model
generator. The broader scheduler must learn to detect and produce reduction IR
and dependency metadata automatically.

## Workstreams and Tests

### 1. Reduction IR Surface

Plan:

- Generalize `T.thread_allreduce_sum` into a reduction API family:
  - `T.thread_reduce(value, op, axis, predicate=True, dtype=None)`
  - `T.row_reduce(buffer/value, op, axis, output, policy=None)`
  - `T.block_reduce(value, op, axis, output, policy=None)`
  - `T.reduction_axis(name, extent, map_expr, role="lane")`
- Keep backend-specific operations out of user/model code.
- Preserve existing `T.reduce_sum`, `T.reduce_max`, etc. by routing compatible
  cases through the same semantic reduction representation where possible.

Test:

- Unit: TileLang language tests assert generated TIR contains
  `tir.tvm_thread_allreduce` or semantic reduction op, not backend intrinsics.
- Lowering: Metal/CUDA/CPU smoke tests lower the same semantic IR.
- Negative: unsupported malformed axes fail with a clear scheduler error.
- Command:

```bash
python -m pytest testing/python/metal/test_metal_reduce.py -q
python -m pytest testing/python/language/ -q -k "reduce"
```

Acceptance:

- No new model-level `tir.metal.*` or `tir.cuda.*` reduction intrinsics.
- Existing reduction tests continue to pass.

### 2. IR Builder Reduction Detection

Plan:

- Add a pass that recognizes common manual reduction idioms:
  - per-lane accumulation followed by `if lane == 0: store`
  - `*_partial` write patterns over a reduce axis
  - host-side sum of device partial tensors in adapter code
  - direct `simd_sum` or shuffle intrinsic calls
- Rewrite matched idioms to semantic reduction IR plus explicit output shape.
- Emit diagnostics for patterns that look reducible but fail legality checks.

Test:

- Golden TIR tests for pattern rewrite.
- cppmega Mamba3 bwd no longer needs model-specific reduction helper calls
  after this pass is enabled.
- Static check: search for production `*_partial` outputs and backend reduction
  intrinsics.
- Commands:

```bash
rg -n "simd_sum|simd_shuffle|d[A-Z]?_partial|ddt_partial" tilelang cppmega_mlx
python -m pytest testing/python/transform/ -q -k "reduction"
```

Acceptance:

- Reducible patterns become semantic reduction IR before backend lowering.
- Non-reducible cases explain exactly which legality proof failed.

### 3. Scheduler ReductionPlan

Plan:

- Introduce a scheduler-level `ReductionPlan` metadata object:
  - input buffer regions
  - output buffer regions
  - reduction axes and extents
  - thread/block mapping
  - accumulator dtype
  - candidate strategies
  - required memory visibility scope
  - alias/in-place constraints
- Candidate strategies:
  - same-simdgroup
  - split-simdgroup
  - shared/threadgroup reduction
  - row-reduce helper
  - two-pass global reduction
  - vectorized CPU fallback

Test:

- Unit tests construct ReductionPlan from representative TIR.
- Plan serialization snapshots are stable and reviewable.
- Backend lowerers reject only the strategy, not the semantic reduction itself.
- Commands:

```bash
python -m pytest testing/python/scheduler/ -q -k "reduction_plan"
python -m pytest testing/python/metal/test_metal_reduce.py -q
```

Acceptance:

- P=32/64/96/128/256 choose single-kernel allreduce strategies.
- P greater than one threadgroup cap chooses a generated two-pass strategy
  instead of exposing partial tensors to Python.

### 4. Z3 Legality Proofs

Plan:

- Add proof obligations for each plan:
  - exact coverage: every element reduced once
  - no out-of-bounds access
  - no write-write race
  - no read-after-write hazard without event/barrier
  - in-place legality
  - tail-dim and broadcast legality
  - int64/index-width safety
- Proof result becomes scheduler metadata:
  - `proved_no_sync`
  - `requires_threadgroup_barrier`
  - `requires_device_event`
  - `requires_two_pass`
  - `cannot_parallelize(reason)`

Test:

- Property tests generate small shapes and compare proof result against a
  reference enumerator.
- Regression tests cover tail dims, broadcast, int64 indexing, and aliasing.
- Negative tests deliberately create overlapping writes and require rejection.
- Commands:

```bash
python -m pytest testing/python/scheduler/ -q -k "z3 or legality"
python -m pytest testing/python/metal/ -q -k "hazard or sync or reduce"
```

Acceptance:

- A plan with `proved_no_sync=True` emits no barrier/event.
- A proven hazard emits the minimal required sync primitive.
- Impossible plans fail before codegen, not at runtime.

### 5. Sync and Event Planner

Plan:

- Build a dependency graph from buffer regions, streams, owner-output handles,
  and DLPack/TVM-FFI boundaries.
- Insert:
  - no sync when proof says no hazard
  - threadgroup barrier for local shared-memory hazards
  - device event for cross-buffer/cross-stream hazards
  - materialization only when a real external pointer boundary requires storage
- Make sync decisions inspectable in lowered metadata.

Test:

- IR tests assert barrier absence/presence by hazard class.
- MLX graph transform tests assert no eager eval/materialization unless an
  external pointer boundary is proven.
- Runtime race tests run kernels repeatedly with deterministic debug guards.
- Commands:

```bash
python -m pytest testing/python/metal/ -q -k "sync or hazard"
python -m pytest /Volumes/external/sources/cppmega.mlx/tests -q -k "dlpack or tvm_ffi or path_c"
```

Acceptance:

- No unconditional sync in generated Path C kernels.
- Debug guard catches stale/null pointer races deterministically.

### 6. Backend Lowerer Registry

Plan:

- Route semantic reduction/scan/dependency IR through backend registries:
  - Metal lowerers
  - CUDA lowerers
  - ROCm lowerers
  - CPU fallback lowerers
- Backend registry decides implementation, not model code.
- Existing Metal lowerer handles:
  - same-simdgroup
  - split-simdgroup
  - threadgroup staging
  - row-reduce helpers
  - generated two-pass reductions

Test:

- Same TIR lowers on multiple backends where available.
- Metal source does not contain CUDA tokens.
- CUDA source does not contain Metal tokens.
- Commands:

```bash
python -m pytest testing/python/metal/ -q
python -m pytest testing/python/language/ -q -k "reduce"
```

Acceptance:

- Model generator contains no backend-specific reduction code.
- Lowerer choice is visible in diagnostics and stable under cache.

### 7. Two-Pass Reductions for Large Axes

Plan:

- For reduction extents beyond one threadgroup cap, scheduler generates:
  - pass 1: per-block partials in internal scratch/owner output
  - pass 2: final reduction into public output
- Scratch is internal to TileLang runtime/lowering, not a public model output.
- Cache and reuse scratch buffers when lifetime analysis allows it.

Test:

- P=512, P=1024, and model-real larger rows lower and run.
- Public outputs do not include `*_partial`.
- Runtime parity against Path B/reference.
- Commands:

```bash
python -m pytest /Volumes/external/sources/cppmega.mlx/tests/test_tilelang_mamba3_path_c.py -q -k "headdim or bwd"
python -m pytest testing/python/metal/ -q -k "two_pass or reduce"
```

Acceptance:

- No Python host reduction for large P.
- No public partial tensors for generated two-pass reductions.

### 8. Reverse Recurrence and Scan Planning

Plan:

- Represent reverse recurrence as scheduler IR:
  - state dependencies
  - rematerialization policy
  - snapshot/cache policy
  - parallel chunks
  - legal in-place updates
- Use Z3 to prove chunk independence and minimal sync.
- Fuse post-recurrence residual/gate reductions into the recurrence kernel when
  legality and register pressure allow.

Test:

- Unit tests for scan dependency metadata.
- Runtime parity for Mamba3/M2RNN/hybrid RNN.
- Performance profiles show reduced serial-over-T bottleneck.
- Commands:

```bash
python -m pytest /Volumes/external/sources/cppmega.mlx/tests -q -k "mamba3 or m2rnn or hybrid"
```

Acceptance:

- No forced serial path when chunk parallelization is proven legal.
- In-place bwd route is chosen only when alias proof passes.

### 9. Register Pressure, Split/Inline, and Hoisting

Plan:

- Scheduler attaches cost metadata:
  - estimated registers
  - local memory pressure
  - threadgroup memory pressure
  - index math cost
  - occupancy estimate
- Apply:
  - hoist repeated index math
  - split kernels only when register pressure exceeds threshold
  - inline only when it improves occupancy or reduces memory traffic
  - choose SIMD reductions for P multiples of 32

Test:

- Codegen text tests assert repeated expensive index expressions are hoisted.
- Profiler tests compare selected plan against baseline candidates.
- Commands:

```bash
python -m pytest testing/python/scheduler/ -q -k "cost or register or hoist"
python -m pytest /Volumes/external/sources/cppmega.mlx/tests -q -k "path_c"
```

Acceptance:

- Path C generated code has no obvious repeated hot index math in inner loops.
- Plan selection is reproducible from recorded cost metadata.

### 10. Autotune, Profiling, and Memoization

Plan:

- Store tuned schedule records keyed by:
  - op signature
  - shape
  - dtype
  - backend target
  - TileLang/TVM/MLX ABI hashes
  - proof hash
- On first run, benchmark legal candidates.
- On later runs, reuse best known schedule unless ABI/proof/codegen hash
  changes.

Test:

- Cache hit/miss tests.
- ABI/hash invalidation tests.
- Profiler smoke tests for Path B vs Path C.
- Commands:

```bash
python -m pytest testing/python/scheduler/ -q -k "autotune or cache"
python -m pytest /Volumes/external/sources/cppmega.mlx/tests -q -k "profile or bench"
```

Acceptance:

- Compile-time slow path is paid once per signature.
- Cached Path C uses the tuned schedule and records the proof/cost reason.

### 11. Production Lint: No Monkeypatch, No Direct Backend Hacks

Plan:

- Add lint checks for production paths:
  - no monkeypatch for MLX/TileLang/TVM-FFI production behavior
  - no `mx.fast.metal_kernel` in Path C production route
  - no direct cppmega public TVM-FFI bypass
  - no direct backend intrinsic calls from model generators
- Allow explicit tests and debug tools to opt in by path/name.

Test:

- Static lint fails on forbidden tokens outside allowlisted test/debug paths.
- Commands:

```bash
python -m pytest /Volumes/external/sources/cppmega.mlx/tests -q -k "lint or monkeypatch"
rg -n "monkeypatch|mx\\.fast\\.metal_kernel|tir\\.metal|simd_sum" /Volumes/external/sources/cppmega.mlx/cppmega_mlx
```

Acceptance:

- Production code path is framework-owned, not monkeypatched.

### 12. Full Model Performance Matrix

Plan:

- Run the real 1B model, block size 2048, batch size 1 first.
- Matrix:
  - dtype: bf16, fp8, int8 where supported
  - optimizer: adamw, lion, muon, muon+adamw, other configured mixes
  - path: Path B, Path C cold, Path C warm/cache hit
  - steps: 20
  - metrics: tok/sec, step/sec, compile time, peak memory, cache hit, selected
    schedule, proof result
- Free memory after every run; each case runs in a fresh subprocess.

Test:

- Benchmark script emits a CSV/Markdown table with no failed cells hidden.
- Path C warm should be within the agreed threshold of Path B, initially
  targeting single-digit percent gap, then tighter after scheduler work.
- Commands:

```bash
cd /Volumes/external/sources/cppmega.mlx
TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap \
TVM_LIBRARY_PATH=/private/tmp/tl_apache_tvm_swap/build/lib \
PYTHONPATH=/private/tmp/tl_apache_tvm_swap:/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python \
.venv/bin/python -m pytest tests -q -k "path_c"
```

Acceptance:

- Full table includes pass/fail reason for every dtype/optimizer/path.
- No benchmark result is reported without exact command, git SHAs, and cache
  state.

## Execution Order

1. Reduction IR API generalization.
2. ReductionPlan metadata in scheduler.
3. Z3 proof module for reductions and hazards.
4. Backend lowerer registry integration.
5. IR pattern rewrite from hand partials to semantic reduction IR.
6. Two-pass generated reductions for axes larger than one threadgroup.
7. Reverse recurrence scan metadata and chunk planner.
8. Sync/event planner connected to dependency metadata.
9. Cost model and register/index hoisting.
10. Autotune/cache and benchmark matrix.
11. Production lint and CI gates.
12. Full 1B model performance table.

## Definition of Done

- Production model code contains semantic TileLang operations, not backend
  reduction intrinsics.
- Scheduler output includes machine-readable proof, sync, and strategy metadata.
- Z3 tests cover both allowed and rejected plans.
- Metal generated code for supported reductions has no public partial outputs.
- Full cppmega 1B matrix runs with Path B and Path C results reported side by
  side.
- No monkeypatch is required for the production path.
