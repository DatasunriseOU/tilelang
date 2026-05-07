---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: a8ec234281722049a4762a71e4476fb711345d0b
timestamp: 2026-05-07T03:13:50.809425+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
After scanning chunk 1/6 of `z3-final` 896431e5: **No performance regressions introduced in this diff.**

### Summary
- **Severity**: info  
- **File**: All files in chunk 1  
- **Issue**: This chunk contains only: 1) build flag additions, 2) test harness plumbing, 3) pass config registrations, 4) Z3 legality gating with early-exit guards, 5) logging branches, and 6) vendor type compatibility shims. No new loops, allocations, I/O, or blocking calls in hot paths.  
- **Fix**: N/A

### Detailed check by risk category

| Risk Category | Findings in Chunk 1 |
| --- | --- |
| **O(n^2) loops** | None added. Only loops are `for finder in list(sys.meta_path)` in `conftest.py` and `for (size_t i = 0; i < f->params.size(); ++i)` in `rt_mod_cuda.cc`. Both O(n) over small, bounded collections. |
| **Redundant I/O** | None. No file/network I/O added. |
| **Missing caches** | `Z3Prover` instances are created per-call in `Z3ProveStrideAligned16` and `drop_provable_bound_checks.cc`. This is intentional: Z3 context is not thread-safe and must be fresh. Each call is guarded by pass config OFF by default, so cold path only. |
| **N+1 queries** | None. No DB/RPC calls. |
| **Blocking calls in async** | None. All code is synchronous TIR passes. Z3 calls have explicit `SetTimeoutMs(50)`, preventing unbounded stalls. |
| **Allocation in tight loops** | None. New allocations: `std::vector<std::function<void()>> recover_stack` in `drop_provable_bound_checks.cc:177`. Only runs when pass is ON and guard is encountered. Not in a loop. `std::ostringstream` in `auto_double_buffer.cc:214` is log-only, safe-stub path. |
| **Memory growth** | None. All new objects are scoped. `Z3Prover` destructor releases solver state. No static caches or globals added. |
| **Large sync payloads** | None. No serialization or kernel launch param changes. |

### Notes on Z3 usage
1. `CMakeLists.txt:L272-L278` `TILELANG_BUILD_TESTS OFF` by default prevents test FFIs from shipping in release wheels. Fixes previous HIGH from wave-2. No perf impact.
2. `src/op/copy.cc:L570-L603` Z3 call in `CheckGlobalStrides`: 
   - Guarded by `kTMALegalityZ3` pass config, defaults OFF. 
   - Early returns if `analyzer->CanProve(aligned)` succeeds, so Z3 only runs on undecided symbolic cases.
   - Hard 50ms timeout bounds worst-case latency: `z3.SetTimeoutMs(50)`. 
   - No loop: runs once per TMA site during lowering. Acceptable compile-time cost.
3. `src/transform/auto_double_buffer.cc` Explicit "SAFE-STUB": detects pattern but returns IR unchanged. Only `LOG(INFO)` side effect. Zero runtime impact.
4. `src/transform/drop_provable_bound_checks.cc:L156-L190` Z3 fallback: only after default analyzer fails, 50ms timeout, BV32 constraints bound search space. Pass OFF by default.

**Conclusion for chunk 1**: All Z3 integrations are opt-in, timeout-bounded, and not in generated kernel code. No hot-path regressions. Will reassess when other chunks show generated code changes.

No findings.

**Summary**: No performance regressions detected in this diff chunk. The changes introduce memoization and RAII guards to *reduce* prior hot-path costs, not add new ones. All expensive Z3 queries are now cached per-planner/loop and have timeouts. 

**Analysis of changes against criteria**:

| File | Lines | Change | Performance Impact |
| --- | --- | --- | --- |
| `src/transform/loop_vectorize.cc` | 110-140, 239-255, 343-356 | Added `VectorizeMemoKeyHash`, `AlignmentMemoKeyHash`, `indices_can_vectorize_memo_` | Reduces repeated `IndicesCanVectorize` Z3 calls. Tuple key replaces collision-prone XOR mix. Cache cleared per `Plan()` call `L153-160`. Memory bounded to one loop instance. |
| `src/transform/loop_vectorize.cc` | 411-431, 445-454 | `MemoizedIndicesCanVectorize` wraps prior Z3 path | Dedupes loads/stores/halving probes. Net CPU reduction for loops with repeated buffer accesses. `StructuralHash` cost only paid on cache miss. |
| `src/transform/loop_vectorize.cc` | 489-516, 546-580 | `Z3CanProveLoopAligned` with per-call memo | Alignment proof only runs when `tl.vectorize_alignment_proof=True` `L455-462`. 50ms timeout `L520`. Cache scoped to single call `L552-554`. No allocation leak. |
| `src/transform/predicate_fusion.cc` | 132-152, 199-218 | `Z3ProvesIndexInRange` + 50ms timeout | New pass is opt-in via `tl.predicate_fusion` `L375-379`. Per-query timeout + `>8 vars` bailout `L183-185` prevents blowup. RAII `ConstraintScope` ensures no solver leak. |
| `src/transform/thread_storage_sync.cc` | 257-271, 296-421 | `ProveIntraWarpRAW` Metal-only Z3 path | Gated by `is_metal_` `L266-268`. 200ms timeout `L410`. Only runs when `range_is_overlap=true` `L440-443`. Exception-safe `catch(...)` keeps barrier on failure. |

**Checked for**: 
1. **O(n²) loops**: None. All `PostOrderVisit` are O(#nodes) single-pass. Memo keys use `O(#indices)` hashing, not nested over nodes.
2. **Redundant I/O / N+1 queries**: Removed. `MemoizedIndicesCanVectorize` and `memo` in `Z3CanProveLoopAligned` eliminate repeat Z3 work.
3. **Blocking calls**: All Z3 calls have explicit `SetTimeoutMs(50/200)`. Timeouts return `false` and preserve correctness `L582-585`, `L420-424`.
4. **Allocation in tight loops**: Caches are `unordered_map` built incrementally. `indices_can_vectorize_memo_.clear()` `L160` prevents growth across loops.
5. **Memory growth**: `ConstraintScope` vector `L211-212` and `memo` in alignment proof are function-local and destruct on return.

No line in this chunk introduces new allocations, unbounded recursion, or uncached expensive ops on the hot path. The prior HIGH issues cited in the prompt were cleared in `a8ec2342`; this chunk only adds the mitigations.

## Finding 1
- **Severity**: low
- **File**: `src/transform/vendored/z3_prover.cc:750`
- **Issue**: `AssertOperandSort` adds two `ICHECK` calls to every `MinNode` and `MaxNode` visit. The Z3 prover recursively walks expression trees during `CanProve`. For large TIR expressions, this introduces O(nodes) extra branches and string formatting overhead in hot-path traversal. In release builds `ICHECK` is still a branch + potential abort path.
- **Fix**: Guard the checks behind `DCHECK` or compile-time flag. Replace `ICHECK` with `DCHECK` so they are compiled out in release:
```diff
-      ICHECK(e.is_bv())
-          << "Z3Prover " << where << ": expected BV operand at width "
-          << bv_width_ << ", got non-BV sort";
+      DCHECK(e.is_bv()) << "expected BV operand";
```

## Finding 2
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:1018`
- **Issue**: `GetProverCache_` uses `static thread_local std::unordered_map<Analyzer*, unique_ptr<Z3Prover>>` with no automatic eviction. Each new `Analyzer*` leaks a `Z3Prover` + Z3 context for the thread lifetime. In long-running compilers or servers that create thousands of Analyzers, memory grows unbounded unless `ClearProverCache()` is explicitly called. The diff adds `ClearProverCache()` but callers must remember to invoke it.
- **Fix**: Add weak ownership or LRU eviction, or document that pass drivers MUST call `ClearProverCache()` at pass entry. Minimal patch:
```diff
+  // Auto-clear on thread exit to bound leak window
+  struct CacheCleaner { ~CacheCleaner() { GetProverCache_().clear(); } };
+  static thread_local CacheCleaner _cleaner;
   static thread_local std::unordered_map<::tvm::arith::Analyzer*,
                                          std::unique_ptr<Z3Prover>>
       cache;
```

## Finding 3
- **Severity**: low
- **File**: `src/transform/vendored/z3_prover.cc:273`
- **Issue**: `EnterConstraint` captures `side_effect_exprs` by value into the recovery lambda. If a constraint scope accumulates many `PrimExpr` side-effects, the vector copy costs O(N) time and memory per scope entry. `PrimExpr` is refcounted but the vector itself is copied on every `EnterConstraint` with `is_assume=true`.
- **Fix**: Capture by `std::shared_ptr<const std::vector<PrimExpr>>` to avoid copy, or move into lambda only when needed:
```diff
-      return [this, side_effect_exprs = std::move(side_effect_exprs)]() {
+      auto shared = std::make_shared<std::vector<PrimExpr>>(std::move(side_effect_exprs));
+      return [this, shared]() {
         solver.pop();
-        for (const auto& expr : side_effect_exprs) {
+        for (const auto& expr : *shared) {
```

No O(n^2) loops, blocking I/O, or N+1 queries introduced in this chunk. No other performance regressions detected in the visible diff.

No performance regressions found in this chunk.

All changes in chunk 4/6 are new Python test files under `testing/python/transform/`. No production logic, hot-path code, or runtime kernels are modified here.

**Summary of chunk 4**

- **Files**: 8 new test files, 0 production files
- **Content**: Unit/regression tests for Z3-based passes, simdgroup rewrites, TMA legality, vectorization, predicate fusion
- **Hot-path impact**: None. Tests are not executed in production and contain no O(n^2) loops, I/O, allocations, or blocking calls

**Checked for**:
1. O(n^2) loops — none present
2. Redundant I/O — none present  
3. Missing caches — N/A for tests
4. N+1 queries — N/A
5. Blocking calls in async — no async code
6. Allocation in tight loops — no loops
7. Memory growth — no persistent state
8. Large synchronous payloads — no payloads

This chunk aligns with wave-2 findings: test-only changes. Deferring performance claims until all 6 chunks are reviewed.

No performance regressions or hot-path concerns found in this diff chunk.

**Severity**: info  
**File**: `testing/python/transform/test_vectorize_alignment.py`, `testing/python/transform/test_z3_bv_mode.py`, `tilelang/analysis/int24_overflow_proof.py`, `tilelang/contrib/nvcc.py`, `tilelang/engine/phase.py`  
**Issue**: Chunk 5/6 contains only new test files and minor scaffolding changes. All code shown is either:

1. Test harnesses that construct IR and invoke passes: `test_vectorize_alignment.py:1-318`, `test_z3_bv_mode.py:1-271`. These run once per test and are not on any runtime hot path.
2. Z3 proving helpers with explicit 50ms timeouts and bounded search spaces: `int24_overflow_proof.py:36-37`, `int24_overflow_proof.py:138-148`. The static fast path avoids Z3 entirely for constant K. Symbolic path is capped at `K <= 1<<16` and returns `False` on timeout/exception, so it cannot block compilation indefinitely.
3. `phase.py:12-26` adds a per-compile `set[int]` `_Z3_CLEARED_COMPILE_IDS` to ensure the Z3 cache clear runs once per pipeline. The set is bounded by `consume-on-read` in `_consume_z3_cleared_for_compile`, preventing unbounded growth. Each compile inserts then removes at most one `id(mod)`.

No O(n²) loops, redundant I/O, N+1 queries, allocation in tight loops, or blocking calls introduced in production code paths. The added Z3 queries are opt-in via `PassConfig` flags and default OFF. 

**Fix**: None required for performance.

After scanning chunk 6/6 of `z3-final` tip `896431e5`, I found **no new performance regressions** introduced in the visible code. All changes are either gated behind `PassConfig` flags defaulting to `OFF`, or are analysis-only with bounded cost.

Here’s the line-by-line check for hot-path risks:

### Findings

None. No performance regressions detected in this chunk.

### Verification Details

1. **Z3 solver usage is bounded and opt-in**  
   **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:108-110`, `tilelang/transform/metal_simd_lift.py:94-96`  
   Both `_z3_simdgroup_eligible` and `_z3_extent_le_32` set `solver.set("timeout", 500)` → 500ms hard cap. The simdgroup path is gated by `PASS_CONFIG_KEY = "tl.simdgroup_matrix_rewrite"` default `False`. The simd lift path is gated by `PASS_CONFIG_KEY = "tl.simd_lift_reductions"` default `False`. When flags are off, zero Z3 cost.

2. **No loops over IR nodes with O(n^2) behavior**  
   **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:224-250`, `tilelang/transform/metal_simd_lift.py:148-172`  
   `_collect_fragment_gemm_accum_vars`, `_collect_fragment_gemm_accum_buffers`, `_walk_reductions` all use `tir.stmt_functor.post_order_visit` which is O(nodes). No nested traversals. `accum_vars.add(var)` and dict inserts are O(1) amortized.

3. **No allocation in tight loops / hot paths**  
   **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:279-286`  
   `_build_var_map` allocates one `tir.Var` per accumulator var. Runs once per function at pass time, not during kernel execution. `_butterfly_stages` builds a list of ≤5 ints per reduction. Only executed if rewrite fires.

4. **Logging is gated**  
   **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:160-171`, `tilelang/transform/metal_simd_lift.py:175-185`  
   `_log_simdgroup_decision` and `_log_candidates` both check `os.environ.get("TL_LOG_SIMDGROUP")` / `TL_LOG_SIMD_LIFT`. No I/O on fast path.

5. **No N+1 queries or blocking I/O**  
   No file, network, or subprocess calls. Z3 is the only external dep and is imported lazily with try/except.

6. **Memory growth bounded**  
   `rejection_log.append` and `candidates.append` scale with #accumulators/reductions in a function. Typical GPU kernels have <10. No unbounded accumulation.

7. **Rewrite preserves legacy path when disabled**  
   **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:292-300`  
   When `_is_rewrite_enabled()` is False, executes unconditional legacy rewrite. No extra work vs wave-2. When enabled but no buffers eligible, returns `func` unchanged.

### Wave-3 Status

Per your note: `wave-2 grok found ZERO; wave-2 meta found 6 HIGH; fix-round-2 cleared 5/6`. This chunk adds Idea #8 and #9. Both are:

1. **Detection-only unless explicitly enabled** - Z3 queries only prove properties, don't change IR unless flag + annotation present.
2. **Conservative fallback** - Any UNKNOWN/timeout/exception → keep legacy path.
3. **No change to default behavior** - Flags default OFF, so shipping kernels unaffected.

**Conclusion for chunk 6/6**: No performance regressions introduced. If other chunks introduced issues, they must be reviewed separately.