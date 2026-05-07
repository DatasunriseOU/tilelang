---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: dfcb37dcdfb11bcafebb0a3bfe137d02ad86dfd1
timestamp: 2026-05-07T03:34:07.219561+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
## Finding 1
- **Severity**: high
- **File**: `src/transform/drop_provable_bound_checks.cc:185`
- **Issue**: The BV32 emulation pushes `v < IntImm(v.dtype(), 1<<31)` for every free int `Var`. When `v.dtype()` is `int32`, constructing `IntImm(int32, 1<<31)` overflows because `1<<31` = 2147483648 exceeds `int32` max 2147483647. This wraps to `-2147483648`, turning the intended constraint `v < 2^31` into `v < -2^31`, which is unsatisfiable for any non-negative `v`. Z3 will then fail to prove valid bounds and the pass will silently keep guards it should drop.
- **Fix**: Promote the bound to a 64-bit immediate regardless of `v.dtype()`: `IntImm(DataType::Int(64), int64_t{1} << 31)`. Z3 can still reason about 32-bit variables with a 64-bit bound; or explicitly cast: `cast(v.dtype(), IntImm(DataType::Int(64), 1LL<<31))` after range check.

```diff
- recover_stack.push_back(
- z3.EnterConstraint(v < IntImm(v.dtype(), kBitBound)));
+ recover_stack.push_back(
+ z3.EnterConstraint(v < IntImm(DataType::Int(64), kBitBound)));
```

## Finding 2
- **Severity**: high
- **File**: `src/transform/drop_provable_bound_checks.cc:184-196`
- **Issue**: `recover_stack` is populated by calling `z3.EnterConstraint` twice per variable. If the second `EnterConstraint` throws, the first constraint remains pushed but its corresponding recovery lambda is never popped because the exception exits before the `push_back`. On the next prover use the solver state is corrupted, potentially causing false proofs or crashes. Exceptions are swallowed by the outer `catch (...)`, so the leak is silent.
- **Fix**: Ensure each successful push is paired with a pop even on exception. Either push after both constraints succeed, or use RAII guard:

```diff
- recover_stack.push_back(z3.EnterConstraint(v >= IntImm(v.dtype(), 0)));
- recover_stack.push_back(
- z3.EnterConstraint(v < IntImm(v.dtype(), kBitBound)));
+ auto r1 = z3.EnterConstraint(v >= IntImm(DataType::Int(64), 0));
+ auto r2 = z3.EnterConstraint(v < IntImm(DataType::Int(64), kBitBound));
+ recover_stack.push_back(r2);
+ recover_stack.push_back(r1);
```

## Finding 3
- **Severity**: medium
- **File**: `src/transform/auto_double_buffer.cc:142-145`
- **Issue**: `BuildSoundnessObligation` returns `Bool(true)` unconditionally. The comment claims this is a “safe-stub” and that the detector is the safety gate, but the pass still instantiates `Z3Prover` and calls `CanProve(true)`, which always returns true. If `enabled=true`, the pass will log “soundness obligation proved by Z3” for any detected pattern, even when the pattern is actually unsafe. This defeats the purpose of the prover audit trail and gives false confidence that Z3 validated the transform.
- **Fix**: Either return a real obligation encoding `load_addr_k+1` independence, or skip Z3 entirely in stub mode and log that no proof was attempted. Do not claim Z3 proved soundness when it didn’t check anything.

```diff
- return Bool(true);
+ // Stub: no real obligation yet. Return false to ensure we never claim a proof.
+ return Bool(false);
```

## Finding 4
- **Severity**: low
- **File**: `conftest.py:43-47`
- **Issue**: Path rewrite uses `old[idx + 1:]` then `os.path.join(target_root, rel)`. If `old` contains symlinks or a non-canonical `/tmp/tl_apache_tvm_swap//tilelang/`, `rel` can start with `/`, causing `os.path.join` to discard `target_root` and return an absolute path outside the worktree. This will leave the redirecting finder pointing to the wrong tree, breaking imports in CI when the worktree is not under `/tmp/tl_idea10_worktree`.
- **Fix**: Normalize before slicing and strip leading separators:

```diff
- rel = old[idx + 1:]
- v[key] = os.path.join(target_root, rel)
+ rel = os.path.normpath(old[idx + 1:]).lstrip(os.sep)
+ v[key] = os.path.normpath(os.path.join(target_root, rel))
```

No other correctness regressions, off-by-one, or null-handling bugs detected in this chunk.

## Finding 1
- **Severity**: critical
- **File**: `src/transform/loop_vectorize.cc:1043`
- **Issue**: `Z3CanProveUnitStride` treats negative stride `-1` as a proof of unit stride. The comment says negative strides are “just as vectorizable”, but the `VectorizeRewriter` on lines 1259-1266 only emits `Ramp(stride=+1)` and never generates a negative ramp. If Z3 proves stride `-1`, the planner will return `true` for `IndicesCanVectorize`, the rewriter will mark the loop `kVectorized`, and codegen will emit ascending lanes `[0][1][2][3]` while the actual memory pattern is descending. This produces wrong results for any kernel reading `out[N-1-i]`. The code explicitly removed the `stride == -1` branch at 1608-1619 and replaced it with only positive check, yet `Z3CanProveUnitStride` at 1013-1031 probes both directions and returns true for `-1`.
- **Fix**: Make the Z3 probe respect the codegen limitation. After line 1026, change the negative-direction block to also check a config flag or remove it entirely until codegen supports `negative_ramp`:
```diff
- if (!proved) {
- // Negative-direction probe...
- proved = z3.CanProve(stride_goal_neg);
- }
+ // Codegen only supports stride=+1 Ramp. Do not accept -1.
+ // if (!proved) {... } // disabled until negative_ramp support lands
```

## Finding 2
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:961`
- **Issue**: `Z3CanProveAlignedAccess` skips the Z3 query when `vector_size <= 1` and returns false. However callers like `Z3CanProveLoopAligned` at 1135 will then set `all_aligned = false` for any scalar access, preventing `tl.vec_aligned` from being set even when the address is trivially aligned. Scalar width-1 should be considered aligned to any boundary, otherwise the annotation is needlessly pessimistic and later passes can’t remove redundant alignment checks.
- **Fix**: Treat width 1 as aligned by definition:
```diff
- if (vector_size <= 1) {
- return false;
- }
+ if (vector_size <= 1) {
+ return true; // width-1 is aligned to any boundary
+ }
```

## Finding 3
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:1339`
- **Issue**: `IsAffineInVar` allows any `VarNode` in the expression, not just the loop var. This permits expressions like `i + k` where `k` is another loop var. `Z3CanProveUnitStride` then substitutes `var -> var+1` but leaves `k` unchanged, so `delta = (i+1+k) - (i+k) = 1` and Z3 returns true. That is correct for contiguity, but the earlier guard intended to reject non-affine patterns like `B[i]` to avoid unsound Z3 models. Allowing arbitrary free vars means an expression `A[i + j*stride]` with symbolic `stride` could be accepted if `stride` happens to be 1 in the model, yet the substitution trick is only sound when the coefficient of `var` is a constant. If `stride` is not constant, the proof can be false-positive.
- **Fix**: Restrict free vars to loop-invariant constants only, or ensure `Substitute`+`Simplify` removes all other vars before the Z3 query. Simplest: add a check that all non-`var` Vars are constants in `analyzer`:
```diff
   PostOrderVisit(expr, [&](const ObjectRef &obj) {
     if (const auto *v = obj.as<VarNode>()) {
+ if (v!= var.get() &&!analyzer->const_int_bound(v).has_value()) {
+ ok = false; // reject symbolic coefficients
+ }
     }
   });
```

## Finding 4
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:1392`
- **Issue**: `Z3CanProveUnitStride` computes `iter_hi = analyzer->Simplify(iter_var_size - 1)` then builds `range_constraint = (var >= 0) && (var < iter_hi) && (iter_var_size > 0)`. If `iter_var_size` is symbolic, `Simplify(iter_var_size - 1)` may return a non-const expression, e.g. `N-1`. The constraint `var < N-1` excludes the last iteration where `var = N-1`, so the probe only checks deltas up to the penultimate element. For contiguity you must allow `var+1` to be a valid index, i.e. require `var < iter_var_size - 1` but the goal checks `expr(var+1) - expr(var)`. If the last two addresses are not distance 1, Z3 won’t see it. This makes the proof unsound: it can return true even when the last step is non-unit.
- **Fix**: The valid domain for checking `expr(var+1) - expr(var)` is `0 <= var < iter_var_size - 1`. Use that bound directly:
```diff
- PrimExpr iter_hi = analyzer->Simplify(iter_var_size - 1);
- PrimExpr range_constraint =
- (var >= lo) && (var < iter_hi) && (iter_var_size > 0);
+ PrimExpr one = make_const(vt, 1);
+ PrimExpr range_constraint =
+ (var >= lo) && (var + one < iter_var_size) && (iter_var_size > 0);
```
This ensures `var+1` is always in-bounds for the substitution.

## Finding 5
- **Severity**: low
- **File**: `src/transform/predicate_fusion.cc:117`
- **Issue**: `Z3ProvesConditionLoadsWellDefined` creates a new `Z3Prover` and repeats the full bit-bound setup for every load in the condition. `Z3ProvesInnerWellDefined` at 225 does the same for body loads. If a statement has 10 loads in `b` and 10 in the body, Z3 is invoked 20 times with near-identical constraints. This is a performance regression introduced by the diff. The memo map at 1084-1096 in `loop_vectorize.cc` shows you intended to dedupe, but predicate_fusion lacks it.
- **Fix**: Hoist a single `Z3Prover` and reuse it across all loads, and memoize `(buffer, dim, idx_structural_hash)` like `AlignmentMemoKeyHash` does:
```cpp
auto &z3 = arith::Z3Prover(analyzer);
z3.SetTimeoutMs(50);
std::unordered_map<std::tuple<const void*, size_t, size_t>, bool> memo;
auto probe = [&](const BufferLoadNode* ld, size_t d) {
  auto key = std::make_tuple(static_cast<const void*>(ld->buffer.get()), d,
                             static_cast<size_t>(hasher(ld->indices[d])));
  if (auto it = memo.find(key); it!= memo.end()) return it->second;
  bool ok = Z3ProvesIndexInRange(ld->buffer, d, ld->indices[d], z3);
  return memo.emplace(key, ok).first->second;
};
```

## Finding 6
- **Severity**: low
- **File**: `src/transform/loop_vectorize.cc:1414`
- **Issue**: `bv_hi = make_const(vt, int64_t(1) << 32)` is used when `vt` is 64-bit. If `vt` is actually `int32`, the shift overflows and UB occurs before `make_const` is called. The guard `!vt_is_int32` at 1413 prevents this for 32-bit, but if `vt.bits() == 33` or any non-standard width, the shift is still UB. Z3 expects 32-bit unsigned emulation, so the bound should be clamped to `int64` max regardless of `vt`.
- **Fix**: Use `int64_t` arithmetic only and cast after:
```diff
- PrimExpr bv_hi = make_const(vt, int64_t(1) << 32);
+ int64_t bound = int64_t(1) << 32; // 2^32 fits in int64
+ PrimExpr bv_hi = make_const(vt, bound);
```

No other correctness bugs visible in this chunk. Idea-8/9 soundness: `drop unconstrained z3.Int` change is not in this diff, but the remaining code no longer constructs symbolic `z3.Int` paths for `metal_fragment_to_simdgroup`/`metal_simd_lift`, so no new regressions introduced here.

## Finding 1
- **Severity**: high
- **File**: `src/transform/vendored/z3_prover.cc:748`
- **Issue**: `VisitExpr_` for `RampNode` returns only `VisitExpr(op->base)`. For a vector ramp `[base, base+stride, ...]`, any proof that depends on the upper lanes will use only `base`. This can make `CanProve` return true when the full ramp would violate the property, i.e. unsound over-approximation. The comment acknowledges losing stride/lanes info, but the prover is used for bound-check elimination and double-buffer legality where all lanes matter.
- **Fix**: Either reject `RampNode` by returning a fresh unconstrained var in BV mode and `Create(op)` in Int mode, or conservatively model the ramp as `base <= ramp <= base + stride*(lanes-1)` and assert that in the solver. Simplest:
```cpp
::z3::expr VisitExpr_(const ::tvm::tirx::RampNode* op) override {
  // Conservatively unsupported: ramp lanes not modeled -> don't prove
  return Create(op);  // fresh symbolic, prevents false positives
}
```

## Finding 2
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:752`
- **Issue**: `VisitExpr_` for `BroadcastNode` returns `VisitExpr(op->value)`. Same issue as Ramp: the value expression is used but lane count is dropped. If a proof relies on a property failing in lane `i>0`, the prover will not see it. For bound-check elimination this is unsound when the broadcast is indexed, e.g. `select(i < N, broadcast(0,4)[i])`.
- **Fix**: Treat as unsupported for now, same as Ramp:
```cpp
::z3::expr VisitExpr_(const ::tvm::tirx::BroadcastNode* op) override {
  return Create(op);
}
```

## Finding 3
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:405`
- **Issue**: When `Bind(var, Range)` with `min_value >= max_value`, the code does `solver.add(ctx->bool_val(false))`. This correctly makes the scope UNSAT. However, later calls to `CanProve` in this scope will return `true` for any predicate because `unsat => forall P`. That is logically correct, but it means an empty range bind silently makes all subsequent proofs succeed. If a caller accidentally binds an empty range due to upstream bug, optimizations will trigger based on vacuously true proofs. There is no diagnostic.
- **Fix**: Add a one-time warning when an empty range is bound, so users can see a mis-generated range:
```cpp
if (min_value >= max_value) {
  if (!bv_range_warned_) {
    LOG(WARNING) << "Z3Prover: empty range bind for " << var 
                 << " [" << min_value << ", " << max_value << ")";
    bv_range_warned_ = true;
  }
  memo_.emplace(var, var_expr);
  solver.add(ctx->bool_val(false));
  commit_memo = false;
  return;
}
```

## Finding 4
- **Severity**: low
- **File**: `src/transform/vendored/z3_prover.cc:167`
- **Issue**: `MakeUIntVal` is defined but never used. `VisitExpr_(IntImmNode*)` always calls `MakeIntVal(op->value)`. If `op->dtype` is `uint64` and `op->value > INT64_MAX`, TIR stores it as a negative `int64_t`. `MakeIntVal` will wrap it in BV mode, giving the wrong bit pattern. Unsigned constants therefore lose precision even in BV64 mode.
- **Fix**: Dispatch on dtype in `VisitExpr_(IntImmNode*)`:
```cpp
::z3::expr VisitExpr_(const IntImmNode* op) override {
  if (bv_width_ > 0 && op->dtype.is_uint()) {
    return MakeUIntVal(static_cast<uint64_t>(op->value));
  }
  return MakeIntVal(op->value);
}
```

## Finding 5
- **Severity**: info
- **File**: `src/transform/vendored/z3_prover.cc:378`
- **Issue**: `Bind(const Var&, const Range&, ...)` calls `memo_.emplace(var, var_expr)`. `emplace` is a no-op if `var` already exists in `memo_`. If the same `Var` is rebound in a nested scope, the old Z3 expr will be reused and the new range constraints will be applied to the old symbol. `scope_stack_.pop_back()` erases the memo entry on exit, so this only affects rebinds in the same scope, which TIR usually forbids. Still, a defensive `memo_[var] = var_expr` would prevent a stale-symbol bug if upstream violates the invariant.
- **Fix**: Replace `memo_.emplace(var, var_expr);` with `memo_[var] = var_expr;` in all Bind paths.

No other correctness regressions visible in this chunk. The `EnterConstraint` side-effect snapshot and `SetBitVectorMode` scope checks correctly prevent use-after-scope and mode-leak bugs addressed by CPPMEGA fix-A4/A6.

## Finding 1
- **Severity**: medium
- **File**: `testing/python/transform/test_loop_vectorize_z3_contiguity.py:79`
- **Issue**: `test_symbolic_contiguous_vectorizes` claims to test the Z3 fallback for symbolic-extent loops, but the loop is `for i in T.vectorized(8):`. Extent 8 is static, so the vectorizer never hits the symbolic-extent path. The Z3 fallback is never exercised, making the test a false positive for the behavior it documents.
- **Fix**: Change the loop to iterate over the symbolic bound so the pass must prove unit-stride for `_SYMBOLIC_N`:
```diff
- for i in T.vectorized(8): # noqa: F821
+ for i in T.serial(_SYMBOLIC_N):
             B[i] = A[i]
```
If the intent is to keep `vectorized`, use `for i in T.vectorized(_SYMBOLIC_N):`.

## Finding 2
- **Severity**: low
- **File**: `testing/python/transform/test_predicate_fusion.py:40`
- **Issue**: `_has_nested_if_pattern` defines an inner `visit(s)` function that is never called. It also contains `tir.stmt_functor.post_order_visit(s, lambda _: None)` where `s` is undefined in that scope. The unreachable code does nothing, but it obscures intent and could hide future logic errors if someone assumes `visit` is used.
- **Fix**: Delete the dead `visit` definition and keep only the lambda that actually runs:
```diff
- def visit(s):
- if isinstance(s, tir.IfThenElse):
- inner = s.then_case
- if isinstance(inner, tir.IfThenElse):
- found[0] = True
- return
- try:
- tir.stmt_functor.post_order_visit(s, lambda _: None)
- except Exception:
- pass
-
     tir.stmt_functor.post_order_visit(stmt, lambda s: (
         found.__setitem__(0, True) if (
             isinstance(s, tir.IfThenElse) and isinstance(s.then_case, tir.IfThenElse)
         ) else None
     ))
```

No other correctness bugs, race conditions, off-by-one errors, or mishandled null/edge cases were found in this diff chunk. The remaining tests correctly pin the behaviors described in their docstrings.

No critical correctness bugs found in this diff chunk. All findings are either test gaps or minor issues that don’t introduce regressions to existing behavior.

## Finding 1
- **Severity**: medium
- **File**: `testing/python/transform/test_vectorize_alignment.py:115`
- **Issue**: `test_static_misaligned_addr` asserts `"main" in text` but never checks that `tl.vec_aligned` is absent. Comment says “annotation should NOT be added”, yet the test only verifies the script renders. With `tl.vectorize_alignment_proof=True`, a buggy prover that emits `tl.vec_aligned=True` for misaligned access would still pass this test. This defeats the purpose of the regression test for fix-B1.
- **Fix**: Add explicit negative assertion:
```python
text = mod_on.script()
assert "tl.vec_aligned" not in text
assert "vec_aligned" not in text
assert "main" in text
```

## Finding 2
- **Severity**: low
- **File**: `testing/python/transform/test_vectorize_alignment.py:110`
- **Issue**: `test_static_aligned_addr` only asserts `mod_on is not None`. The docstring claims “alignment annotation should appear”, but the test never checks it. If the Z3 proof path silently stops working, CI stays green. The helper `_has_vec_aligned_annotation` exists but isn’t used.
- **Fix**: Replace `assert mod_on is not None` with `assert _has_vec_aligned_annotation(mod_on)` or at minimum `assert "tl.vec_aligned" in mod_on.script()`.

## Finding 3
- **Severity**: low
- **File**: `testing/python/transform/test_vectorize_alignment.py:258`
- **Issue**: `test_negative_stride_not_vectorized` checks `assert mod is not None` but never validates that the loop remains scalar or that no vectorized annotation appears. If a future change makes `IndicesCanVectorize` accept stride=-1, this test won’t catch the regression.
- **Fix**: After build, scan for `T.vectorized` in script and fail if present:
```python
text = mod.script()
assert "T.vectorized" not in text or "for i in T.vectorized" not in text
```

## Finding 4
- **Severity**: info
- **File**: `testing/python/transform/test_z3_bv_mode.py:250`
- **Issue**: `test_z3_prover_cross_pass_isolation` depends on `tl.z3.clear_prover_cache` FFI. If a future refactor removes/renames that symbol, the test is skipped rather than failing. Cache-bleed regressions would become silent. Given this tests a critical isolation invariant, skipif is too weak.
- **Fix**: Remove `@pytest.mark.skipif` and make the test hard-fail if FFI missing:
```python
_CLEAR_PROVER_CACHE = tvm.ffi.get_global_func("tl.z3.clear_prover_cache")
assert _CLEAR_PROVER_CACHE is not None, "tl.z3.clear_prover_cache FFI must exist for isolation guarantee"
```

## Finding 5
- **Severity**: info
- **File**: `testing/python/transform/test_vectorize_alignment.py:130`
- **Issue**: `_SYM_BASE = T.symbolic("base")` is defined but never used in `_symbolic_aligned_main`. Comment says it’s for module-scope symbolic cooperation, but the function doesn’t reference it. If the intent was to test symbolic base alignment, the test doesn’t actually exercise that path.
- **Fix**: Either use `_SYM_BASE` in the indexing: `B[i + _SYM_BASE] = A[i + _SYM_BASE]`, or remove the unused symbol and update the comment.

## Finding 6
- **Severity**: info
- **File**: `tilelang/engine/phase.py:8`
- **Issue**: `_Z3_CLEARED_COMPILE_IDS: set[int] = set()` is a module-global that persists across compilations. `OptimizeForTarget` calls `_consume_z3_cleared_for_compile` which discards the key, so normal flow is safe. However, if `LowerAndLegalize` runs and marks the ID, but `OptimizeForTarget` is never called due to an exception in later passes, the ID leaks in the set forever. For long-running processes this is a memory leak of 8 bytes per aborted compile.
- **Fix**: Add a `finally` cleanup or use `weakref.WeakKeyDictionary` keyed by the `IRModule` object instead of `id()`. Simpler: rely on GC of the module to naturally bound growth since `id()` can be reused after GC, but add a comment documenting the tradeoff.

No new regressions introduced by e0402d77 + dfcb37dc are visible in this chunk. Idea-8/9 soundness changes are not in this diff, so cannot be verified here.

## Finding 1
- **Severity**: high
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:L99-L101`
- **Issue**: `_z3_simdgroup_eligible` docstring claims to run a Z3 proof for `shape[0] % 8 == 0 /\ shape[1] % 8 == 0 /\ addr % 16 == 0`, but the implementation immediately returns `False` for any non-`IntImm` shape and never calls Z3. The function name and log output `static-prove ... (no z3 needed)` mislead callers and tests into believing a solver-backed proof occurred. Downstream `is_simdgroup_eligible` uses this to set `proved=True` and logs `z3-proved=True`, so tooling will report "Z3 proved" when only a static check ran. This violates the stated Idea #8 behavior and makes the rewrite unsound if future code trusts the `proved` bit for symbolic shapes.
- **Fix**: Rename to `_static_simdgroup_eligible_with_addr` and stop returning `proved=True` for the non-Z3 path. Either: 1) Actually invoke Z3 with proper symbolic variable binding, or 2) Make `is_simdgroup_eligible` return `proved=False, "static-pass"` and never claim Z3. Update `_log_simdgroup_decision` to not pass `decided_z3=True` when static check passes.

## Finding 2
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:L87-L92`
- **Issue**: `_static_simdgroup_eligible` checks `shape[-2:]` but does not verify rank >= 2 before `len(shape) < 2`. For a 1-D buffer, this returns `False` silently. However `GEMM` accumulators should be 2D. The gated rewrite path will silently skip 1D `local.fragment` buffers even if they are valid 8x1 simdgroup tiles, causing a regression: with `tl.simdgroup_matrix_rewrite=True`, formerly promoted buffers stay scalar. This is a behavior change vs unconditional path.
- **Fix**: Document that 1D buffers are intentionally ineligible, or adjust check to allow `shape[-1] % 8 == 0` for rank-1 and match Metal spec. Add explicit log when rank<2.

## Finding 3
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:L217-L224`
- **Issue**: `_collect_fragment_gemm_accum_buffers` extracts the buffer only if `region_call.args[0]` is `tir.BufferLoad`. If the GEMM accumulator is passed via `tir.BufferRegion` with `region` field, or via `tir.decl_buffer`, `buf` becomes `None` and the var is dropped from `accum_with_buf`. With `tl.simdgroup_matrix_rewrite=True`, such buffers are never considered eligible, so they won't be promoted even when static shape passes. This is a regression vs the unconditional path which only needed the var, not the buffer.
- **Fix**: Fall back to `region_call.buffer` if `region_call` is `tir.BufferRegion`. Handle `decl_buffer` case by mapping var->buffer via `func.buffer_map`. If buffer cannot be recovered, log and treat as ineligible explicitly.

## Finding 4
- **Severity**: low
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:L459-L465`
- **Issue**: `apply_simdgroup_matrix_rewrite` catches all `Exception` when writing attributes, then silently returns `new_func` without attrs. If `EMITTED_ATTR_KEY` fails to write due to `tir.StringImm` construction on large `rejection_log`, the caller has no indication. Tests relying on `tl.simdgroup_matrix_rewrite_emitted` attr will flake. Swallowed exception hides bugs.
- **Fix**: Remove broad `except Exception: pass`. Let attribute errors propagate or log at `ERROR` level. If join string is too long, truncate with `...`.

## Finding 5
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:L121-L122`
- **Issue**: `_z3_extent_le_32` claims Z3 fallback but immediately returns `False, "symbolic extent rejected"` for any non-`IntImm`. Docstring says "UNKNOWN/timeout → False" implying Z3 was attempted. Like Finding 1, this misreports detection. `detect_candidates` will mark `proved=False` with message `symbolic extent rejected`, which is correct, but `_log_candidates` at L240 triggers only if `os.environ.get("TL_LOG_SIMD_LIFT") or any(c.proved for c in candidates)`. So rejected symbolic loops won't log unless env var set, hiding missed optimizations.
- **Fix**: Either implement real Z3 binding for symbolic extents, or rename function to `_static_extent_le_32` and update docstring. Ensure logging path covers rejected symbolic cases when `tl.simd_lift_reductions=True`.

## Finding 6
- **Severity**: high
- **File**: `tilelang/transform/metal_simd_lift.py:L262-L270`
- **Issue**: `_butterfly_stages` asserts `1 <= shift < 32` to guard shuffle mask. However the caller `_build_butterfly` passes `extent` directly from loop extent without checking if `extent` is a power of two. For `extent=30`, stages become `[16,8,4,2,1]` which sum to 31, not 30. The butterfly assumes full power-of-2 participation. If lane id >= extent participates, `shfl_xor_sync` will XOR with out-of-range lanes, producing undefined values. The later guard at L364-L373 only rejects non-pow2, but `_z3_extent_le_32` may prove `extent<=32` for symbolic `extent=30` and annotation present, then `_visit_for` still runs because it doesn't re-check pow2 before calling `_build_butterfly`. Missing guard between Z3 proof and rewrite.
- **Fix**: Add explicit `assert (extent & (extent-1)) == 0` at start of `_build_butterfly`, or filter candidates in `_walk_reductions` to require pow2. Ensure `_visit_for` re-checks pow2 after Z3 proof.

## Finding 7
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:L400-L403`
- **Issue**: `_ButterflyRewriter` only handles `tir.For`, `SeqStmt`, `IfThenElse`, `LetStmt`, `AttrStmt`, `AllocateConst`, `Allocate`, `DeclBuffer`, `Block`, `BlockRealize`. If the reduction loop is inside `tir.AssertStmt` or `tir.Evaluate` of a side-effecting call, recursion stops and returns node unchanged. Nested annotated loops inside asserts will be skipped silently, leaving scalar loop in IR. This is a correctness gap: rewrite is partial.
- **Fix**: Add cases for `AssertStmt` and generic fallback: `return node` only for leaf stmts; otherwise reconstruct with mutated fields. Or use `tir.stmt_functor.mutate` to avoid manual cases.

## Finding 8
- **Severity**: low
- **File**: `tilelang/transform/metal_simd_lift.py:L114-L119`
- **Issue**: `_is_butterfly_annotated` converts annotation value via `bool(int(v))` which throws `ValueError` for non-int strings like `"true"`. The outer `except Exception: return bool(v)` catches it, but that returns `True` for any non-empty string, including `"false"`. Annotation `"tl.simd_butterfly_lane": "false"` will be treated as `True`, causing incorrect rewrite.
- **Fix**: Explicitly check for string `"true"/"false"` case-insensitive before `int` cast. Example: `if isinstance(v, str): return v.lower() == "true"`.

No other correctness bugs found in this chunk. The changes correctly make Idea-8/9 conservative-by-default: Z3 paths now reject symbolic inputs instead of using unconstrained `z3.Int`, preventing unsound proofs.