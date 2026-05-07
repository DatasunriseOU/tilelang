---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: a8ec234281722049a4762a71e4476fb711345d0b
timestamp: 2026-05-07T02:49:10.149159+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
Based on this chunk only. No issues found in the files visible here that would be a correctness bug introduced by this diff. The changes are mostly PassConfig registrations, CMake flag default flip, and scaffolding. 

**Notes on verified fixes from wave-1b claims:**

- **CMakeLists.txt:269** `TILELANG_BUILD_TESTS` default is now `OFF`. This addresses the "HIGH design+sec" concern: test-only Z3 FFI probes are not shipped by default in release wheels. Guarded by `if(TILELANG_BUILD_TESTS)` at CMakeLists.txt:306-308. 
- **src/op/builtin.h** registers `kDropProvableBoundChecks`, `kPredicateFusion`, `kVectorizeAlignmentProof`, `kTMALegalityZ3`. All default to disabled via PassConfig, matching "opt-in" requirement.
- **src/op/copy.cc:530-569** adds `Z3ProveStrideAligned16` with try/catch returning `false` on any exception. This prevents Z3 exceptions from propagating and conservatively rejects TMA, matching the "conservative fallback" fix.
- **src/op/copy.cc:623-641** gates Z3 legality behind `kTMALegalityZ3` flag. When disabled, behavior matches pre-diff: only `CanProveNonNegativeDivisibleBy(16)` rejects. When enabled, unprovable alignment now rejects with warning. This fixes the prior HIGH: silent admission of symbolic strides.

**No new correctness bugs detected in chunk 1 of 6**

- No off-by-one errors: loop bounds unchanged in visible code.
- No swallowed exceptions: only Z3 call in `Z3ProveStrideAligned16` has explicit `catch(...)` returning `false`, which is the intended conservative behavior.
- No mismatched types: `kTMALegalityZ3` read as `Bool(false)`, `Z3ProveStrideAligned16` takes `PrimExpr` and `arith::Analyzer*`, matches call site.
- No race conditions: all changes are single-threaded compiler passes.
- No broken null/None handling: `PrimFunc f` checked with `if (!f->body.defined()) return f;` in `drop_provable_bound_checks.cc`. `ExtractFuncInfo` validates `f.defined()` and `f->params[i].defined()`.
- No regressions to existing behavior when flags are off: `kTMALegalityZ3`, `kDropProvableBoundChecks`, `kPredicateFusion`, `kVectorizeAlignmentProof`, `kAutoDoubleBuffer` all default `false`.

If later chunks contain `z3_prover.cc:489`, `z3_prover.cc:436-449`, or `phase.py` double-clear logic, I will review them when visible.

## Finding 1
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:1731`
- **Issue**: `Z3CanProveAlignedAccess` returns `false` when `vector_size <= 1` but the caller `Z3CanProveLoopAligned` only checks `bool aligned = Z3CanProveAlignedAccess(...)` and then uses it to set `tl.vec_aligned` annotation. If `vector_size == 1`, alignment is trivially true for any address. Returning `false` causes the pass to skip the annotation even though alignment at width-1 is always satisfied. This is a correctness regression: valid aligned accesses won’t get the `tl.vec_aligned` attr, blocking downstream `vec.load_aligned` codegen.
- **Fix**: Treat `vector_size <= 1` as trivially aligned. Replace the early return:
```diff
-  if (vector_size <= 1) {
-    return false;
-  }
+  if (vector_size <= 1) {
+    return true;
+  }
```

## Finding 2
- **Severity**: high  
- **File**: `src/transform/loop_vectorize.cc:1883`
- **Issue**: `ProveIntraWarpRAW` requires `tx_w.defined() && tx_r.defined()` after `bind_axis`. However `bind_axis` only sets `*w_out` and `*r_out` if both `p_iv.has_value()` and `c_iv.has_value()`. For a 2-D launch `threads` contains only `[ty, tx]` or `[tz, tx]`. `find_axis(..., "threadIdx.x")` may fail if the canonical tag was rewritten by earlier passes like `LowerThreadAllreduce` which changes to `threadIdx.x_inner`. In that case `tx_w` remains `NullValue<Var>()`, the `.defined()` check fails, and the function returns `false`. The comment claims it handles 2-D launches, but the strict equality check on tag prevents it. This defeats the entire Apple intra-warp elision for 2-D kernels, a regression from intended behavior.
- **Fix**: Fall back to positional mapping when `threadIdx.x` is missing but `threads.size() >= 1`. Or accept tags containing `"threadIdx.x"` as a substring. Example:
```diff
-    if (!tx_p_iv.has_value() || !tx_c_iv.has_value()) {
-      LOG(WARNING) << "ProveIntraWarpRAW: no canonical threadIdx.x found "
-                      "(prev or curr); keeping barrier.";
-      return false;
-    }
+    if (!tx_p_iv.has_value() || !tx_c_iv.has_value()) {
+      if (prev.threads.size() > 0 && curr.threads.size() > 0) {
+        tx_p_iv = prev.threads[prev.threads.size() - 1]; // assume x is last
+        tx_c_iv = curr.threads[curr.threads.size() - 1];
+      } else {
+        return false;
+      }
+    }
```

## Finding 3
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:1766`
- **Issue**: `Z3CanProveUnitStride` has degenerate-extent guard: `if (iv_imm->value <= 1) return false;`. For a loop with extent 1, the access `A[i]` is trivially contiguous because there is only one iteration. Returning `false` forces scalarization even though vector_size=1 would be valid. More importantly, `IndicesCanVectorize` also has the same guard at line 1673. This causes false negatives for unit loops that appear after loop unrolling or in boundary tiles.
- **Fix**: Extent 1 should return `true` because contiguity is vacuous. Change both guards:
```diff
-    if (iv_imm->value <= 1) {
-      return false;
-    }
+    if (iv_imm->value <= 0) {
+      return false;
+    }
+    if (iv_imm->value == 1) {
+      return true;
+    }
```

## Finding 4
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:1838`
- **Issue**: `Z3ProvesIndexInRange` bit-bounds each free var with `BVBoundsForDtype(dt)`. For `uint32` this gives `[0, 1<<32)`. But `make_const(dt, 1<<32)` overflows int32 and is then used in `var < bv_hi`. If `dt` is `int32`, the bound becomes `[INT_MIN, INT_MAX)` which is too wide: it allows negative indices that TIR treats as OOB. The comment says signed int32 is capped at 2^31-1, but the actual constraint built is `var < make_const(int32, 0x7fffffff)` which is correct, yet the code path for `!vt_is_int32` adds `var < 1<<32` for `int64`. If `iter_var_size` itself is `int32` with value > `0x7fffffff`, the `iter_var_size <= bv_hi` check fails and we reject valid loops. The bound should use `min(1<<32, max_val_of_dtype)` but current logic mixes int64 constant with int32 var.
- **Fix**: Cast `bv_hi` to `var.dtype()` before comparison, and clamp 32-bit unsigned to `0xFFFFFFFFu` not `1<<32`:
```diff
-      PrimExpr bv_hi = make_const(vt, int64_t(1) << 32);
-      range_constraint =
-          range_constraint && (var < bv_hi) && (iter_var_size <= bv_hi);
+      int64_t hi_val = dt.is_uint() ? (1LL << dt.bits()) - 1 : (1LL << (dt.bits()-1)) - 1;
+      if (dt.bits() >= 32 && dt.is_uint()) hi_val = 0xFFFFFFFFLL;
+      PrimExpr bv_hi = make_const(dt, hi_val);
+      range_constraint =
+          range_constraint && (var <= bv_hi) && (iter_var_size <= bv_hi + 1);
```

## Finding 5
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:2012`
- **Issue**: `Z3ProvesConditionLoadsWellDefined` enumerates loads in the inner `if` condition but does not recurse into `CallNode` or other side-effect expressions. `ExprBufferLoadCollector` inherits `ExprVisitor` which will visit `CallNode` arguments, so loads inside call args are caught. However if `cond` is `tirx.if_then_else(a, buf[i], 0) > 0`, the `buf[i]` is inside a nested expression that `ExprVisitor` handles, but the function returns `true` if `loads.empty()`. That part is correct. The bug: there is no check that the condition itself is pure. If `cond` is `popcount(buf[i]) > 0` where `popcount` is an impure intrinsic, Z3 proof may succeed on index range but fusion still changes evaluation order. The outer guard check at line 2250 already rejects impure `inner->condition`, so this is redundant. No bug here. 

Actually re-reading: line 2250 does `SideEffect(inner->condition) > kReadState` and bails. So loads in condition are allowed because `BufferLoad` is `kReadState`. That means fusion to `if(a && buf[i] > 0)` will evaluate `buf[i]` even when `!a`. The Z3 proof ensures index is in-range, but not that the load is non-trapping. On CPU/GPU, reading OOB is UB even if address calculation doesn't fault. The proof only checks `0 <= idx < extent`. If `buf` is null or buffer was freed, the load still faults. Wave-1 meta found this and claimed it was fixed by `Z3ProvesConditionLoadsWellDefined`. The fix is present, so no new issue.

## Finding 6
- **Severity**: low
- **File**: `src/transform/loop_vectorize.cc:1813`
- **Issue**: `Z3CanProveUnitStride` logs with `DLOG(INFO)` on UNKNOWN/timeout. `DLOG` is compiled out in release builds, so silent failures won't be diagnosable in production. The comment says this is intentional, but for a performance pass that can silently disable vectorization, an `VLOG(1)` or counter would help triage. Not a correctness bug, but reduces debuggability.
- **Fix**: Change to `VLOG(1)` or add a stats counter exposed via `PassContext`:
```diff
-      DLOG(INFO) << "Z3CanProveUnitStride: could not prove unit stride for "
+      VLOG(1) << "Z3CanProveUnitStride: could not prove unit stride for "
```

## Finding 7
- **Severity**: low
- **File**: `src/transform/loop_vectorize.cc:173`
- **Issue**: `VectorizeMemoKeyHash` uses `static_cast<size_t>(std::get<3>(k))` for `target_vectorized_size`. If `target_vectorized_size` is negative, cast wraps. `IndicesCanVectorize` already `ICHECK(target_vectorized_size >= 1)`, but the memo is called before that ICHECK in some paths like `MemoizedIndicesCanVectorize` which returns early for `<=1`. Still, a rogue caller could insert negative keys. Hash will not collide, but mixing negative via `TupleHashMix` is fine. No correctness bug, just defensive coding.
- **Fix**: Add `ICHECK_GE(target_vectorized_size, 1)` at start of `MemoizedIndicesCanVectorize` to match `IndicesCanVectorize` contract.

No other correctness bugs introduced by this chunk. The fixes from wave-1b appear correctly implemented: `ICHECK` guard mentioned in prompt is not in this diff chunk; `SetBitVectorMode` usage not shown; `ConstraintScope` RAII is used properly at lines 1791, 1807, 1897. `MemoizedIndicesCanVectorize` correctly handles cache per `Plan` call due to explicit `clear()` at line 436-442.

Based on the provided diff chunk, here are the correctness findings. No NEW critical bugs introduced in this chunk, but I found regressions and correctness issues in the BV-mode logic and scope handling.

## Finding 1
- **Severity**: high
- **File**: `src/transform/vendored/z3_prover.cc:L401-L416`
- **Issue**: `Bind` with empty range `[min, max)` where `min >= max` asserts `ctx->bool_val(false)` and memos the var, but does not mark the constraint scope as UNSAT. Later `CanProve` calls will still push a negated goal and ask Z3 to solve, wasting time. The scope is logically dead, yet the solver remains SAT until a check. Prior meta wave-1b likely intended to short-circuit proof results for dead scopes, but this only adds `false` and continues. If multiple empty binds happen, multiple `false` get asserted redundantly.
- **Fix**: After `solver.add(ctx->bool_val(false));`, immediately mark scope as proven-unsat so subsequent `CanProve` returns `true` without solver query. Example:
```diff
+    // Scope is now trivially UNSAT; tag it so CanProve can short-circuit.
+    scope_stack_.back().push_back(Scope{Scope::Assert, /*var*/{}, /*val*/{}, /*min*/{}, /*max*/{}, /*constraint*/ctx->bool_val(false)});
     solver.add(ctx->bool_val(false));
     commit_memo = false;
     return;
```

## Finding 2
- **Severity**: high  
- **File**: `src/transform/vendored/z3_prover.cc:L375-L395`
- **Issue**: BV-mode clamp logic for out-of-range `Bind(var, Range)` over-approximates but fails to preserve exact failure semantics. When caller range `[min_value, max_value)` is entirely outside BV width, code falls back to full BV range `[lo, hi+1)`. This makes `CanProve(var < lo)` return `false` instead of `true` because the prover forgot the bind was impossible. Prior bug was “dropped bind”; current fix introduces “dropped UNSAT”. A bind that should make the scope trivially false now allows models where `var = lo`, violating caller intent that the variable cannot exist.
- **Fix**: If `clamped_min >= clamped_max` after clamping, treat as empty range: memo the var and assert `false` just like L401-L416. Do not assert the full BV range.
```diff
-              // no representable interval. Fall back to the full BV
-              // range to keep the var bound at the right sort.
-              clamped_min = lo;
-              clamped_max = hi + int64_t{1};
+              // Caller's range is unrepresentable in BV. Mark UNSAT.
+              memo_.emplace(var, var_expr);
+              solver.add(ctx->bool_val(false));
+              commit_memo = false;
+              return;
```

## Finding 3
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:L489-L495`
- **Issue**: `SetBitVectorMode` ICHECK guard prevents mode switch with non-empty root scope. However `RebuildSolver_` clears `scope_stack_` and pushes a new root frame, then `ns = Namespace{}`. If a caller had called `Bind` at root scope, then called `SetBitVectorMode`, the ICHECK fires correctly. But if a caller calls `Bind`, then `EnterConstraint`+pop, root scope is empty and ICHECK passes. The subsequent `RebuildSolver_` still drops the memoized vars from that `Bind`. Recovery lambda from `EnterConstraint` may later erase a var from `memo_` that no longer exists, which is harmless, but any cached `z3::expr` handles held outside `memo_` become stale because `ctx` is reused but `solver` is new. No crash, but violates expectation that `Bind` at root survives mode flip. This is a latent API misuse bug.
- **Fix**: Either 1) Forbid `SetBitVectorMode` unless `memo_.empty()`, or 2) Document that `SetBitVectorMode` invalidates all prior `Bind` calls and caller must re-bind. Add:
```cpp
ICHECK(memo_.empty()) << "SetBitVectorMode invalidates all existing bindings; "
                      << "clear them or use a fresh Z3Prover instance.";
```

## Finding 4
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:L178-L184`
- **Issue**: `MakeUIntVal` in BV mode uses `ctx->bv_val(static_cast<uint64_t>(value), bv_width_)`. For `bv_width_=32`, passing `value > UINT32_MAX` will truncate silently with no warning, unlike `MakeIntVal` which logs truncation. `uint64_t` inputs typically come from `dtype.max_value` for `uint64`, so `UINT64_MAX` will wrap to `0xFFFFFFFF` without diagnostic. This hides precision loss when mixing uint64 constants in BV32 mode.
- **Fix**: Add symmetric truncation warning for `MakeUIntVal`:
```cpp
if (!bv_truncation_warned_ && bv_width_ < 64) {
  uint64_t hi = (uint64_t{1} << bv_width_) - 1;
  if (value > hi) {
    LOG(WARNING) << "Z3Prover BV" << bv_width_ << ": MakeUIntVal(" << value 
                 << ") wraps; (further occurrences suppressed)";
    bv_truncation_warned_ = true;
  }
}
```

## Finding 5
- **Severity**: low
- **File**: `src/transform/vendored/z3_prover.cc:L210-L216`
- **Issue**: In BV mode, `VarNode` creation skips dtype range constraints with comment "already bounded by sort width". True for bit-width, but signedness is lost. A `int8` var becomes BV32 with range `[-2^31, 2^31-1]`, not `[-128,127]`. Downstream proofs relying on `int8` range will be unsound: `CanProve(var <= 127)` returns `false` because Z3 can assign 128. This is a known limitation of BV mode but the comment implies it's safe. It’s a silent semantic change vs Int mode.
- **Fix**: Either add explicit range clamp `solver.add(MakeIntVal(min_val) <= e && e <= MakeIntVal(max_val));` even in BV mode when `dtype.bits() < bv_width_`, or update comment to warn: `// NOTE: dtype range not enforced; BV sort over-approximates`. 

## Finding 6
- **Severity**: info
- **File**: `src/transform/vendored/z3_prover.cc:L338-L348`
- **Issue**: `CanProve` catches `std::exception` and `...`, logs, returns false. This is correct for robustness, but swallows `z3::exception` separately from `std::exception`. Z3 exceptions derive from `std::exception`, so first catch handles it. No bug, but the `catch (...)` after `catch(const std::exception&)` is unreachable for Z3 errors. Keep for non-std throws, but note it may hide ABI issues.
- **Fix**: None required, but add comment that `catch(...)` is for FFI boundary safety.

**Verification of prior fixes**: 
1. `ICHECK guard against SetBitVectorMode-during-active-scope` at L489-L495 present and correct.
2. `Bind empty-range memo with UNSAT scope` at L401-L416 partially fixed: memo is bound and `false` asserted, but scope not marked UNSAT for short-circuit, see Finding 1.
3. `phase.py prover-cache double-clear sentinel-gated` not visible in this chunk, cannot verify.

No other correctness bugs found in this chunk.

No critical or high-severity correctness bugs found in this diff chunk. All tests added are regression tests for Z3-related fixes and guard behavior. No production code shown in this chunk, only test files.

### Review Summary
- **Scope**: Only `testing/python/transform/*` test files visible in chunk 4/6
- **Prior issues referenced**: ICHECK guard, empty-range memo, CMake default, prover-cache sentinel. None of these appear in this chunk, so cannot verify fixes here.
- **New issues introduced**: None detected in test code logic

### Low/Info Findings

## Finding 1
- **Severity**: low
- **File**: `testing/python/transform/test_intra_warp_2d_launch.py:215-216`
- **Issue**: Test `test_unusual_thread_tag_keeps_barrier` asserts barrier must be kept because tag `threadIdx.x_inner` is non-canonical. However the comment says TVM's `ThreadScope::Create` accepts any tag starting with `threadIdx.`. If upstream TVM changes tag canonicalization, this test could become a false positive. The test relies on implementation detail of `ProveIntraWarpRAW` strict-equality allowlist, not on public contract.
- **Fix**: Add explicit comment that test will fail if tag matching policy changes. Consider `pytest.skipif` when TVM version < known-good version, or mock the allowlist check.

## Finding 2
- **Severity**: info
- **File**: `testing/python/transform/test_loop_vectorize_z3_contiguity.py:73-76`
- **Issue**: Comment claims `T.vectorized(8)` should not be downgraded when Z3 proves contiguity. Test only asserts `"main" in text` which does not validate the vectorized annotation was preserved. If Z3 path silently removed `T.vectorized`, test would still pass.
- **Fix**: Parse lowered IR and assert `ForKind == ForKind.kVectorized` or check for `T.vectorized` in script. Example:
  ```python
  assert "T.vectorized" in text or "vectorized" in str(mod["main"].body)
  ```

## Finding 3
- **Severity**: info
- **File**: `testing/python/transform/test_predicate_fusion.py:45-56`
- **Issue**: Helper `_has_nested_if_pattern` defines `visit` but never calls it. It instead uses a lambda with `found.__setitem__`. The `visit` function is dead code. Not a correctness bug, but confusing for future maintenance.
- **Fix**: Delete unused `visit` function or use it consistently:
  ```python
  def visit(s):
      if isinstance(s, tir.IfThenElse) and isinstance(s.then_case, tir.IfThenElse):
          found[0] = True
  tir.stmt_functor.post_order_visit(stmt, visit)
  ```

## Finding 4
- **Severity**: low
- **File**: `testing/python/transform/test_z3_bv_mode.py`
- **Issue**: File referenced in "Files touched" list but content not shown in diff chunk. Cannot review. If this chunk is incomplete, orchestrator must merge to check for SetBitVectorMode guard added at `z3_prover.cc:489` per Wave-2 notes.
- **Fix**: None for this chunk. Flag for orchestrator: verify `ICHECK` guard exists and prevents mode switch during active scope.

## Finding 5
- **Severity**: info
- **File**: `testing/python/transform/test_tma_legality.py:36-46`
- **Issue**: Test uses `_z3_skip = pytest.mark.skipif` to xfail when PassConfig not registered. This will hide regressions if CI environment forgets to rebuild C++ extension. Silent skips can mask the feature being disabled.
- **Fix**: If TMA legality is required for correctness, use `pytest.mark.xfail(strict=True)` or fail loudly. If optional, document in CI that skips are expected without Z3 build.

No correctness bugs, race conditions, null handling errors, or off-by-one errors present in the visible test code.

## Finding 1
- **Severity**: high
- **File**: `testing/python/transform/test_vectorize_alignment.py:117`
- **Issue**: Test `test_static_misaligned_addr` claims "annotation must NOT appear" but only asserts `"main" in text`. It never checks absence of `tl.vec_aligned`. A regression where the prover falsely tags misaligned access as aligned would pass this test. The comment says "no false positive" but the assertion doesn't enforce it.
- **Fix**: Replace `assert "main" in text` with `assert "tl.vec_aligned" not in text and "vec_aligned" not in text` to actually verify the contract.

## Finding 2
- **Severity**: medium
- **File**: `testing/python/transform/test_vectorize_alignment.py:45-48`
- **Issue**: Helper `_has_vec_aligned_annotation` exists but is unused in tests. `test_static_aligned_addr` at line 60-64 only asserts `mod_on is not None` despite the test description saying "alignment annotation should appear". This makes the test vacuous: it cannot catch a regression where alignment proof runs but fails to annotate.
- **Fix**: Use the helper: `assert _has_vec_aligned_annotation(mod_on)` when `enable=True` and alignment is provable.

## Finding 3
- **Severity**: medium  
- **File**: `testing/python/transform/test_vectorize_alignment.py:278-281`
- **Issue**: `test_negative_stride_not_vectorized` only asserts `mod is not None`. It does not verify that no `vectorized` annotation appears. The comment says "must NOT be marked as vectorizable" but the test cannot fail if vectorization incorrectly occurs.
- **Fix**: After build, check script text: `assert "T.vectorized" not in mod.script()` or inspect `loop.annotations` to ensure no `kVectorized` flag.

## Finding 4
- **Severity**: low
- **File**: `testing/python/transform/test_z3_bv_mode.py:260-264`
- **Issue**: `test_z3_clear_prover_cache_idempotent` calls `_CLEAR_PROVER_CACHE()` 5 times but doesn't assert anything about state. If the function became non-idempotent and corrupted internal structures, this test would still pass. It only verifies "doesn't crash", not idempotency of observable behavior.
- **Fix**: Surround with a prove call: prove something, clear 5x, prove again and assert same result. Or at minimum add comment that idempotency is defined as "no crash".

## Finding 5
- **Severity**: info
- **File**: `testing/python/transform/test_vectorize_alignment.py:108-110`
- **Issue**: `_SYM_BASE = T.symbolic("base")` is defined but never used in `_symbolic_aligned_main`. Comment claims it's for "symbolic-base alignment proof" but the function uses plain `i` with no symbolic base offset. Test doesn't actually exercise symbolic address arithmetic, so Z3 path for symbolic base is untested.
- **Fix**: Either use `_SYM_BASE` in the access: `B[i + _SYM_BASE] = A[i + _SYM_BASE]` with constraints, or remove the unused symbol and fix the comment.

## Finding 6
- **Severity**: medium
- **File**: `tilelang/contrib/nvcc.py:349-358`
- **Issue**: New logic tries `target.attrs.get("arch")` then falls back to `getattr(target, "arch", None)`. `target.attrs` can be a `MapContainer` where `.get()` raises if key missing instead of returning None, depending on TVM version. This can throw `AttributeError`/`KeyError` instead of hitting the fallback, regressing `get_target_compute_version` on newer TVM.
- **Fix**: Guard the dict access: 
  ```python
  target_arch = None
  if target is not None:
      if hasattr(target, "attrs") and "arch" in target.attrs:
          target_arch = target.attrs["arch"]
      if target_arch is None:
          target_arch = getattr(target, "arch", None)
  ```

## Finding 7
- **Severity**: low
- **File**: `tilelang/engine/phase.py:16-19`
- **Issue**: `_Z3_CLEARED_COMPILE_IDS: set[int] = set()` is a module-global accumulating IRModule ids. If `OptimizeForTarget` is never called, e.g. exception in `LowerAndLegalize`, the id remains in the set forever. Since Python reuses ids after GC, a future unrelated module could get the same id and incorrectly skip its cache clear. Set is unbounded for long-running processes.
- **Fix**: Use `weakref.WeakKeyDictionary()` keyed by the module object instead of `id()`, or add a `try/finally` in the compile driver to guarantee `_consume_z3_cleared_for_compile` runs even on error.

## Finding 8
- **Severity**: info
- **File**: `tilelang/language/fp8_op.py:378-382`
- **Issue**: Static fast path checks `bound < (1 << 23) and -bound > -(1 << 23)`. For int24, the valid range is `[-2^23, 2^23-1]`. The second check `-bound > -2^23` is equivalent to `bound < 2^23`, making it redundant. The actual upper bound check should be `bound <= 2^23-1`. Current code rejects `bound == 2^23-1` which is valid.
- **Fix**: Replace with `return -_INT24_MIN >= bound and bound <= _INT24_MAX` or `return abs(bound) <= _INT24_MAX`.

### Verification of wave-1b fixes
Based on chunk 5 only: 
1. `ICHECK guard against SetBitVectorMode-during-active-scope (z3_prover.cc:489)`: Not visible in this diff chunk.
2. `Bind empty-range memo with UNSAT scope (z3_prover.cc:436-449)`: Not visible in this diff chunk. 
3. `CMakeLists.txt:269 TILELANG_BUILD_TESTS default flipped to OFF`: Not visible in this diff chunk.
4. `phase.py prover-cache double-clear sentinel-gated`: Visible and implemented at `tilelang/engine/phase.py:12-24`, `193-197`, `253-257`. Logic matches description: sentinel set after first clear, consumed before second. Finding 7 notes a potential id-reuse edge case, but core fix is present.

No new correctness bugs introduced by visible code in this chunk beyond the test-quality and robustness issues above.

## Finding 1
- **Severity**: high
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:70`
- **Issue**: `_z3_simdgroup_eligible` constructs a Z3 solver but only adds `z_s0 > 0, z_s1 > 0` as constraints, then checks `Not(query)`. If `shape[0]` or `shape[1]` are symbolic TIR expressions, they are never translated into Z3 constraints. The function only substitutes concrete `int`/`tir.IntImm` values. For symbolic `tir.Var` or arithmetic like `n*8`, the Z3 variables `z_s0`, `z_s1` remain unconstrained. As a result, `z3.Not(query)` can be SAT even when the real expression is always divisible by 8, causing false negatives: Z3 returns `sat` -> `proved=False`, so eligible symbolic buffers are rejected. This regresses the Z3 fallback: it only works for constants.
- **Fix**: Add an `analyzer = tvm.arith.Analyzer()` and convert `tir` expressions to Z3 via `tir_to_z3`. Else log that symbolic shape translation is unsupported and return False. Minimal patch: before line 70, add `if not isinstance(s0, (int, tir.IntImm)) or not isinstance(s1, (int, tir.IntImm)): return False, "symbolic shape not translated to Z3; conservative reject"`.

## Finding 2
- **Severity**: high  
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:111`
- **Issue**: `is_simdgroup_eligible` returns `False` for Z3-proved cases. Line 111-112: `proved, query = _z3_simdgroup_eligible(shape, dtype)` then `return False, f"static-fail; z3-proved={proved}; {query}"`. The docstring at lines 96-100 claims Z3 fallback only logs and doesn't flip IR rewrite, but the actual pass at lines 245-248 uses `eligible, reason = is_simdgroup_eligible(buf)` and only promotes when `eligible==True`. Since Z3-proved always returns `False`, the Z3 path never enables promotion. This contradicts "Idea #8" description and makes the feature a no-op, potentially surprising users who enable `tl.simdgroup_matrix_rewrite`.
- **Fix**: Change line 111 to `return proved, f"static-fail; z3-proved={proved}; {query}"`. Ensure the gated rewrite respects Z3 results as the comment suggests.

## Finding 3
- **Severity**: high
- **File**: `tilelang/transform/metal_simd_lift.py:87`
- **Issue**: `_z3_extent_le_32` creates `z_ext = z3.Int("extent")` but never constrains `z_ext` to equal the actual symbolic `extent_expr`. For non-constant `tir.Var` or `tir.Add`, the solver checks `Not(z_ext <= 32)` with `z_ext` unconstrained, which is always SAT for `z_ext=33`. Thus `proved` is always False for any symbolic extent, even `n*8` where `n>0`. The Z3 query doesn't prove anything about the real expression. This makes Idea #9 detection fail for all symbolic loops.
- **Fix**: Add translation from `tir` to Z3 expressions. If `extent_expr` isn't `int`/`tir.IntImm`, either bail out or use `tvm.arith.Analyzer().simplify` and a `tir_to_z3` converter. Without translation, return `(False, "symbolic extent not translated")`.

## Finding 4
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:230`
- **Issue**: `_butterfly_stages` asserts `1 <= shift < 32` at lines 243-247. For `extent=64`, `proved=True` would fail earlier due to `_z3_extent_le_32`, but if that check were bypassed, `_butterfly_stages(64)` yields `[32,16,8,4,2,1]` and the assert fails on `shift=32` because `32 < 32` is false. The assert message says "must lie strictly inside", but Apple Metal simdgroup width is 32, and `shfl_xor_sync(mask, val, 32, 32)` is a no-op identity. Using `shift=32` is invalid for xor shuffle. The code should cap at 31 or reject extent>32 earlier. The guard at lines 372-374 checks `extent_val > 32`, but if `extent_val==32` the stages include 16,8,4,2,1 which is correct. However `_z3_extent_le_32` allows `<=32`, so `extent=32` passes. `_butterfly_stages(32)` gives `[16,8,4,2,1]` - no 32, so safe. The assert is correct, but the comment at line 240-242 is misleading: "must lie strictly inside the Apple simdgroup width [1, 32)" implies 31 is max, but `shfl_xor_sync(..., 31, 32)` is valid. No bug here, but if `extent=33` slipped through, assert would catch it. Mark as medium because the earlier extent check must be watertight.
- **Fix**: Keep assert. Ensure upstream extent check is `<=32` not `<33`. Already correct at line 372.

## Finding 5
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:282`
- **Issue**: `_classify_reduce_op` treats `tir.Sub` as `"add"` at lines 68-69 with comment "sub-into-acc still uses simd_sum-style lowering". For pattern `acc = acc - x`, this cannot be lowered to a butterfly using `shfl_xor_sync` because subtraction is not associative in the way required: `shfl_xor` returns neighbor's value, but `a - shfl(a)` != `shfl(a - shfl(a))`. The butterfly construction in `_apply_op` will emit `a - b` at line 231, which is wrong for cross-lane reductions. This will produce incorrect results if an annotated subtraction loop is rewritten.
- **Fix**: Remove `tir.Sub` case or map to `None`. At line 68-69, change to `return None  # subtraction not supported for butterfly`.

## Finding 6
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:360`
- **Issue**: `_visit_for` calls `_is_butterfly_annotated(node)` before checking `proved`. If a loop is annotated but `proved=False`, it logs at lines 342-347 but then returns unmodified IR. However, `_walk_reductions` at line 152 still records `annotated=True, proved=False` and stashes it in `tl.simd_lift_candidates` at lines 454-464. Downstream tooling may assume any annotated loop will be rewritten when the pass is enabled. This is a silent no-op and violates least-surprise. The log is only at warning level and gated.
- **Fix**: Either emit a diagnostic when `annotated && !proved`, or document that annotation is necessary but not sufficient. Already logs, so acceptable. Consider raising `logger.error` when `TL_SIMD_LIFT_REDUCTIONS=1` and annotation present but not proved.

## Finding 7
- **Severity**: low
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:139`
- **Issue**: `_log_simdgroup_decision` checks `os.environ.get("TL_LOG_SIMDGROUP")` at line 139, but the rest of the file uses `logger.warning`. If Python logging is not configured, `logger.warning` may not emit. Mixing env-var gating with logger levels can cause inconsistent diagnostics. Not a correctness bug, but may hide Z3 decisions.
- **Fix**: Use `logger.debug` and let users enable via logging config, or keep env var consistently.

## Finding 8
- **Severity**: low
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:198`
- **Issue**: `_collect_fragment_gemm_accum_buffers` only recovers `Buffer` when `region_call.args[0]` is `tir.BufferLoad` at lines 200-203. If the GEMM accumulator uses `tir.BufferRegion` or a `tir.Load` from a pointer, `buf` becomes `None` and the buffer is dropped from `accum`, then treated as ineligible at line 250. This is conservative, but may silently skip valid buffers if TVM's lowering changes pattern.
- **Fix**: Add fallback to handle `BufferRegion` nodes: `elif isinstance(region_call.args[0], tir.BufferRegion): buf = region_call.args[0].buffer`.

## Finding 9
- **Severity**: info
- **File**: `tilelang/transform/metal_simd_lift.py:463`
- **Issue**: `PassConfigKey` doc at line 169-170 says default OFF, which matches implementation at line 440-441. The unconditional rewrite still runs when flag is OFF, preserving behavior. This confirms wave-2 fix addresses prior HIGH where the gated path broke default behavior. No new regression.
- **Fix**: None. Verified fix is correct.

No findings for `phase.py`, `z3_prover.cc`, `CMakeLists.txt` in this chunk; they were not included. Based only on chunk 6 of 6, no other correctness issues detected.