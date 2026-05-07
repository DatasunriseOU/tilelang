---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: a8ec234281722049a4762a71e4476fb711345d0b
timestamp: 2026-05-07T03:08:11.843322+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
I reviewed chunk 1/6 of diff `main..896431e5`. 

**No correctness bugs found in this chunk.**

### Summary of chunk 1

This chunk covers:
1. `CMakeLists.txt`: Adds `TILELANG_BUILD_TESTS` flag, default OFF. This fixes the HIGH issue from wave-2 about test-only Z3 FFIs leaking into release wheels. The guard `if(TILELANG_BUILD_TESTS)` before `target_compile_definitions` is correct.
2. `conftest.py`: Test-only file for worktree pytest redirection. No correctness risk to production.
3. `src/op/builtin.cc/.h`: Registers new pass config options: `kTMALegalityZ3`, `kPredicateFusion`, `kVectorizeAlignmentProof`, `kDropProvableBoundChecks`. All default `false/OFF`. Safe.
4. `src/op/copy.cc`: Adds `Z3ProveStrideAligned16` + gating via `tl.tma_legality_z3`. 
   - Lines 566-577: `addr_bytes` and `stride_bytes` both cast to `Int(64)` before Z3. Prevents type mismatch.
   - Lines 633-647: Try/catch around Z3. On exception returns `false` = conservative fallback to cp.async. Correct.
   - Lines 649-657: `elem_bytes` uses `int64_t` to avoid overflow on `dtype.bytes()*dtype.lanes()` before cast to `IntImm`. No overflow for realistic types.
   - Flag default OFF, so no behavior change unless user opts in. No regression.
5. `src/target/codegen_*.cc/.h`: Compatibility fixes for vendored `AllocateNode` vs apache TIR. Methods no longer marked `final/override` because vendored type isn't in apache's `StmtFunctor` table. Comments explain why. This is correct per apache/tvm-latest migration. No dispatch bugs introduced.
6. `src/target/rt_mod_cuda.cc`: Migrates from `runtime::CUDAModuleCreate` to `target::CUDAModuleCreateWithFallback`. Adapts `FunctionInfo` from struct to ObjectRef with `ffi::Array`. 
   - Lines 57-62: Added explicit check for undefined `PrimFunc` and undefined params. Prevents crash on malformed IR.
   - Lines 74-82: `arg_extra_tags` correctly sets `kTensorMap` when param is tensormap pointer. Handles dropped `kDLGridConstant` correctly.
   - Lines 167-172: `source_map` constructed correctly. No API misuse.
7. `src/transform/auto_double_buffer.cc`: Safe-stub pass. Default OFF via `kAutoDoubleBuffer`. Logs only, never modifies IR even when enabled. Z3 call wrapped and result ignored except for log. No risk.
8. `src/transform/drop_provable_bound_checks.cc`: Pass default OFF. 
   - Lines 200-212: BV32 emulation with `0 <= v < 2^31` pushed before Z3 query. Prevents unsound overflow proofs.
   - Lines 218-223: `recover_stack` popped in reverse order. Correct scope management.
   - Lines 191-193: `try/catch` around Z3. Any exception -> `proved=false`, guard kept. Conservative.

All error paths are conservative. No swallowed exceptions that change behavior silently. No off-by-one, no null deref, no race conditions visible. Flag defaults preserve existing behavior.

Note: This review is limited to chunk 1/6. New regressions could exist in chunks 2-6.

After reviewing chunk 2/6 of `z3-final` diff: no correctness bugs introduced in this chunk. All changes are fixes to previously identified issues or conservative safeguards.

### Summary
This chunk only touches `src/transform/loop_vectorize.cc`, `src/transform/predicate_fusion.cc`, `src/transform/thread_storage_sync.cc`, and adds `src/transform/vendored/z3_constraint_scope.h`. Changes address wave-2 meta findings and add defensive guards.

No new regressions found in the visible code.

## Finding 1
- **Severity**: info
- **File**: `src/transform/loop_vectorize.cc:L57-L83`
- **Issue**: `TupleHashMix` and the tuple-keyed memoization for `IndicesCanVectorize` and alignment probes eliminate the XOR-cancellation collision class identified in wave-2. The mix uses `+=` on rotated seed to prevent two equal values from canceling. This is a correctness fix, not a bug. Cache invalidation at `L418-L425` also scopes memo per `Plan` call, preventing stale entries from prior nested loops.
- **Fix**: None needed. Change is correct and addresses prior HIGH finding.

## Finding 2
- **Severity**: info
- **File**: `src/transform/loop_vectorize.cc:L1001-L1011`
- **Issue**: Added `vector_size <= 1` early-return in `Z3CanProveAlignedAccess` and `iter_var_size <= 1` guard in `Z3CanProveUnitStride` and `IndicesCanVectorize`. This prevents meaningless `FloorMod(var, 1)` and `var+1` out-of-domain substitutions that could make Z3 prove false positives on degenerate loops. Conservative guard.
- **Fix**: None needed.

## Finding 3
- **Severity**: info
- **File**: `src/transform/predicate_fusion.cc:L221-L230`
- **Issue**: `Z3ProvesIndexInRange` now null-guards `BufferLoadNode* ld` and checks `ld->indices.empty() || !ld->buffer.defined()` before indexing `buf->shape`. This prevents null deref on malformed TIR where a `BufferLoad` has no indices. Returns `false` to keep fusion disabled, which is safe.
- **Fix**: None needed.

## Finding 4
- **Severity**: info
- **File**: `src/transform/thread_storage_sync.cc:L1596-L1613`
- **Issue**: `ProveIntraWarpRAW` tag mapping switched from positional to `thread_tag` equality. Also added `is_canonical_thread_tag` allowlist rejecting `__wmma_x`, `threadIdx.x_outer`, etc. This prevents false positives when custom thread tags are present. Early return with warning if `threadIdx.x` missing. Conservative and correct.
- **Fix**: None needed.

## Finding 5
- **Severity**: info
- **File**: `src/transform/vendored/z3_constraint_scope.h:L102-L111`
- **Issue**: `ConstraintScope` destructor wraps `recover_()` in `try/catch` to guarantee `noexcept` even if Z3 `solver.pop()` throws. Prevents `std::terminate` from RAII unwind. This fixes a latent crash path noted in fix-B2.
- **Fix**: None needed.

No off-by-one, type mismatch, null deref, race, or regression introduced by this chunk. All modified logic paths either tighten preconditions, add memoization with proper keys, or add RAII/exception safety.

I reviewed chunk 3/6 only. No correctness bugs were introduced in the visible changes.

### Summary
- **No findings**: All changes in `src/transform/vendored/z3_prover.cc`, `z3_prover.h`, and new tests in this chunk are safe and do not introduce regressions.

### Details checked in this chunk
1. **BV mode sort safety** `src/transform/vendored/z3_prover.cc:L197-L206`  
   `Create` skips dtype range constraints when `bv_width_ > 0`. Correct: prevents mixing `Int` and `BV` sorts in solver assertions. Matches the comment.

2. **Side-effect snapshot + lambda capture** `L265-L296`  
   `side_effect_exprs_` is moved to a local, cleared, then captured by value in the `is_assume_in` lambda. This fixes the use-after-scope bug from wave-2. The non-assume path erases memo immediately, matching pre-fix behavior. No lifetime issue.

3. **Z3 exception handling** `L342-L352`  
   `CanProve` now catches `std::exception` and `...`, logs, returns `false`. Prevents unhandled Z3 exceptions from crashing the compiler. Conservative default is correct.

4. **BV range bind clamping** `L383-L422`  
   Out-of-range `Bind` with BV mode now clamps to `[lo, hi+1)` and memoizes `var_expr`. This fixes A7: previously the var was left unbound and later `Visit(var)` minted a fresh free symbol. Clamped range is a sound over-approximation. Empty range case asserts `false` at `L429-L436`, also sound.

5. **Empty range handling** `L428-L436`  
   `min_value >= max_value` commits memo and adds `ctx->bool_val(false)`. Prevents the old bug where the var became unconstrained. Correctly makes scope UNSAT.

6. **Solver rebuild on mode change** `L474-L499`  
   `SetBitVectorMode` checks `scope_stack_.size()==1` and root frame empty before rebuild. Prevents corrupting an active `EnterConstraint` scope. Fast-path when `width == bv_width_` avoids unnecessary rebuilds.

7. **Reset() lifecycle checks** `L542-L562`  
   `Reset` requires root scope and empty frame, same invariant as `SetBitVectorMode`. Clears `memo_`, rebuilds solver, keeps `is_assume=false`. Prevents stale state leaking across passes.

8. **Ramp/Broadcast visitors** `L748-L754`  
   Added to handle vector nodes. Conservatively returns `base` or `value`, which is the prior implicit behavior via `Create(op)` fallback. No correctness regression.

9. **AssertOperandSort** `L760-L774`  
   Enforces sort consistency in BV mode before arithmetic. Prevents opaque Z3 sort-mismatch errors. `ICHECK` is correct for internal invariant.

10. **FloorMod BV dispatch** `L815-L834`  
    Uses `a % b` directly for BV, which is `bvsmod` = sign-of-divisor per SMT-LIB. `floormod` helper only for Int mode where `operator%` is Euclidean. Fixes double-correction bug for negative divisors in BV32/64.

11. **Bitwise ops in BV mode** `L901-L941`  
    Early return for `bv_width_ > 0` to use native BV ops instead of `int2bv/bv2int` roundtrip. Bounds check on shift amount `b_bv < MakeIntVal(bv_width_)` is correct.

12. **Test additions** `testing/python/analysis/test_int24_overflow_proof.py`, `testing/python/language/test_fp8_dot4_packed_legality.py`, `testing/python/transform/test_auto_double_buffer.py`, `test_drop_bound_checks.py`  
    All tests assert expected behavior of new paths. `test_symbolic_negative_addr_returns_false` at `L164-L181` correctly pins the BV32 invariant. `test_overflow_near_intmax_keeps_guard` at `L116-L143` ensures unconstrained Int no longer over-proves. No test logic bugs.

No off-by-one, swallowed exception, type mismatch, race, null deref, or edge-case mishandling found in this chunk.

After reviewing chunk 4 of 6 for `z3-final` tip `896431e5`, here are the correctness findings visible in this diff. I checked for regressions, edge cases, and correctness bugs introduced by the changes.

## Finding 1
- **Severity**: high
- **File**: `testing/python/transform/test_intra_warp_2d_launch.py:157`
- **Issue**: `test_unusual_thread_tag_keeps_barrier` constructs a `tir.Allocate` with `tir.const(1, "bool")` as the `condition` arg. `tir.Allocate` signature is `Allocate(buffer_var, dtype, extents, condition, body, annotations=None)`. Passing `tir.const(1, "bool")` makes the allocation unconditional, which is fine. But then line 160 sets `body = tir.AttrStmt(A.data, "storage_scope", tvm.runtime.convert("shared"), body)`. `tir.Allocate` already has `annotations={"storage_scope": "shared"}` on line 159, so line 160 wraps the allocate with a redundant `AttrStmt` for the same scope. This duplicates metadata and can confuse passes that expect a single storage_scope annotation. If a pass keys off the outermost scope attr, the inner `Allocate` annotation could be shadowed, leading to incorrect scope analysis in downstream transforms like `ThreadSync`.
- **Fix**: Remove the redundant `AttrStmt` on line 160. The `Allocate` annotations already set storage_scope:
```diff
- body = tir.Allocate(A.data, "float32", [16], tir.const(1, "bool"),
- body, annotations={"storage_scope": "shared"})
- body = tir.AttrStmt(A.data, "storage_scope",
- tvm.runtime.convert("shared"), body)
+ body = tir.Allocate(A.data, "float32", [16], tir.const(1, "bool"),
+ body, annotations={"storage_scope": "shared"})
```

## Finding 2
- **Severity**: medium
- **File**: `testing/python/transform/test_intra_warp_2d_launch.py:77`
- **Issue**: `test_2d_launch_intra_simdgroup_elides` and other tests use `_count_storage_sync(mod)` which counts `'T.tvm_storage_sync("shared")'` in `mod.script()`. This is brittle. `tvm.tir.transform.ThreadSync("shared")` can lower to `tir.AttrStmt` with attr key `"storage_scope"` and value `"shared"` + `"storage_sync"` calls, but the script printer may render it as `T.tvm_storage_sync("shared")` or as an intrinsic call depending on TVM version. If the printer changes, the test will pass/fail incorrectly without exercising the actual barrier elision. This is a test correctness issue: the test asserts a codegen property but only checks string presence, not IR semantics. A regression in `ProveIntraWarpRAW` could be masked.
- **Fix**: Replace string counting with structural IR inspection. Walk the IR and count `tir.Call` nodes with op == `tir.op.tvm_storage_sync` and arg == "shared":
```python
def _count_storage_sync(mod: tvm.IRModule) -> int:
    count = [0]
    def visit(node):
        if isinstance(node, tir.Call) and node.op.name == "tir.tvm_storage_sync":
            if len(node.args) > 0 and str(node.args[0]) == "shared":
                count[0] += 1
    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    return count[0]
```

## Finding 3
- **Severity**: medium
- **File**: `testing/python/transform/test_loop_vectorize_z3_contiguity.py:167`
- **Issue**: `test_indirect_indexing_no_vectorize` verifies that indirect indexing `A[B[i]]` does not trigger Z3 fallback and does not crash. The test only asserts `"main" in text`, which only checks that lowering completed. It does not assert that no vectorization happened. If the Z3 guard in `IsAffineInVar` regressed and allowed `B[i]` as affine, the loop could be incorrectly vectorized with wrong semantics. The test name implies it checks "no vectorize", but the assertion doesn't verify it. This weakens the regression coverage for the HIGH audit finding #1 it references.
- **Fix**: Add an explicit check that the lowered loop lacks `T.vectorized` annotation or that the vector length is 1:
```python
def test_indirect_indexing_no_vectorize():
    text = _stringified_ir(_indirect_indexing_main)
    assert "main" in text
    assert "T.vectorized" not in text or "T.vectorized(1)" in text
```

## Finding 4
- **Severity**: low
- **File**: `testing/python/transform/test_predicate_fusion.py:50`
- **Issue**: `_has_nested_if_pattern` defines a local `visit(s)` on line 42 but never calls it. Instead it calls `tir.stmt_functor.post_order_visit` with a lambda on line 52-56. The `visit` function is dead code. Same pattern in `_has_anded_condition` on line 64. Dead code in test helpers increases maintenance burden and can mask logic bugs if someone modifies `visit` thinking it's used.
- **Fix**: Remove the unused `visit` definitions:
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
```

## Finding 5
- **Severity**: info
- **File**: `testing/python/transform/test_simd_reduction_lift.py:41`
- **Issue**: Worktree module loader in `_load_worktree_module` uses `os.path.join(here, "..", "..", "..", "tilelang", "transform", "metal_simd_lift.py")`. If the repo layout changes or tests are run from an installed wheel, the relative path breaks. This is test infrastructure fragility, not a correctness bug in the transform. However, it can cause false negatives in CI if the path resolves to the parent clone instead of the worktree, masking regressions.
- **Fix**: Guard the path and fall back to standard import with a warning if `candidate` doesn't exist, or use `importlib.metadata` to locate the package root.

No critical regressions detected in this chunk. The changes appear to be test additions that pin behaviors for Z3 roadmap items #1, #6, #7, #8, #9, #11. The logic in tests matches the described fixes from wave-2: tag-based axis fix, strict-equality allowlist, 200ms timeout fallback, dtype-aware BV bounds, affine guard for BufferLoad, negative-stride handling.

## Finding 1
- **Severity**: high
- **File**: `testing/python/transform/test_vectorize_alignment.py:73`
- **Issue**: Test `test_static_misaligned_addr` asserts alignment must NOT appear, but only checks `"main" in text`. The comment says "no crash, no false positive", yet the test does nothing to verify the annotation is absent. This means a regression that incorrectly adds `tl.vec_aligned=True` for `B[i + 1]` would pass silently. The test is ineffective at catching false positives from the alignment prover.
- **Fix**: Replace the weak assertion with an explicit negative check:
```python
text = mod_on.script()
assert "tl.vec_aligned" not in text
assert "vec_aligned" not in text
```

## Finding 2
- **Severity**: medium
- **File**: `testing/python/transform/test_vectorize_alignment.py:245`
- **Issue**: `test_negative_stride_not_vectorized` comment claims the loop must NOT be marked as vectorizable, but the test only asserts `mod is not None`. It never inspects the IR to verify `T.vectorized` is absent or that no negative-stride ramp is emitted. A buggy planner that vectorizes `B[127 - i]` with stride -1 would still pass this test.
- **Fix**: Add an IR check that no `For` has kind `kVectorized` or that the lowered loop does not contain `Ramp(..., -1)`. Example:
```python
mod = _build_with_alignment_proof(main, enable=False)
text = mod.script()
assert "T.vectorized" not in text  # or parse IRModule and check For.kind
```

## Finding 3
- **Severity**: medium
- **File**: `testing/python/transform/test_vectorize_alignment.py:51`
- **Issue**: `_has_vec_aligned_annotation` uses substring search `"tl.vec_aligned" in text or "vec_aligned" in text`. TIR script printer may output attributes as `tir.annotation("tl.vec_aligned", True)` or as dict keys with quotes/spacing. The loose check could false-positive on variable names like `not_vec_aligned` or comments. More importantly, the function is unused except in tests, and other tests reimplement the check inline, risking inconsistency.
- **Fix**: Parse the IR properly or use a stricter regex. Better: use `mod.get_global_var().func.body` and `tvm.tir.stmt_functor.post_order_visit` to look for `For` nodes with `anno.tl_vec_aligned`. If keeping text search: `assert re.search(r'\bvec_aligned\s*=\s*True\b', text)`.

## Finding 4
- **Severity**: low
- **File**: `testing/python/transform/test_vectorize_alignment.py:95-96`
- **Issue**: `test_symbolic_aligned_via_z3` creates `_SYM_BASE = T.symbolic("base")` but never uses it in `_symbolic_aligned_main`. The test claims to exercise "symbolic-base alignment proof" yet the loop uses plain `i` with no symbolic offset. This makes the test misleading and it doesn't actually cover the code path where base is symbolic. A regression in symbolic base handling would not be caught.
- **Fix**: Either use `_SYM_BASE` in the access: `B[_SYM_BASE + i] = A[_SYM_BASE + i]`, or remove the unused symbolic and update the docstring to reflect that this tests loop-var-only symbolic reasoning.

## Finding 5
- **Severity**: low
- **File**: `tilelang/contrib/nvcc.py:353-357`
- **Issue**: `get_target_compute_version` wraps `target.attrs.get("arch")` in `try/except Exception`. If `target.attrs` is not a dict or raises something other than `AttributeError`, the blanket catch will hide unrelated errors. The fallback to `getattr(target, "arch", None)` is fine, but swallowing all exceptions can mask bugs in `target` object invariants.
- **Fix**: Narrow the exception: `except (AttributeError, KeyError):` or check `hasattr(target, "attrs")` before access:
```python
target_arch = None
if target is not None:
    if hasattr(target, "attrs") and "arch" in target.attrs:
        target_arch = target.attrs["arch"]
    else:
        target_arch = getattr(target, "arch", None)
```

## Finding 6
- **Severity**: info
- **File**: `testing/python/transform/test_vectorize_alignment.py:19-20`
- **Issue**: `mod.script()` is used throughout tests to check annotations. TIR `script()` output is not guaranteed to be stable across TVM versions and may omit annotations depending on printer flags. If the printer changes, tests like `test_default_off_preserves` could fail even when the IR is correct. This is a maintenance risk, not a correctness bug in product code, but affects test reliability.
- **Fix**: Prefer IR structural checks: iterate over `mod.functions` and inspect `For.annotations` directly instead of relying on text rendering.

No other correctness bugs, off-by-ones, null-handling issues, or regressions are visible in this diff chunk.

## Finding 1
- **Severity**: critical
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:403`
- **Issue**: Line 403 has a stray `)` causing a syntax error. The tuple in the `logger.warning` call is malformed: `tuple(str(d) for d in getattr(buf, "shape", ())),` is closed with an extra `)` before the next argument. This will crash on import/execution when `TL_LOG_SIMDGROUP=1` and an ineligible buffer is logged. 
- **Fix**: Remove the extra `)`:
```diff
-                    tuple(str(d) for d in getattr(buf, "shape", ())),
+                    tuple(str(d) for d in getattr(buf, "shape", ())),
                     getattr(buf, "dtype", "?"),
```

## Finding 2
- **Severity**: high
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:149`
- **Issue**: Z3 query incorrectly asserts `shape[0] % 8 == 0 /\ shape[1] % 8 == 0` for *all* concrete instantiations, but only adds constraints for constant dimensions. For symbolic `s0`, `s1`, the solver creates free `Int("shape0")`, `Int("shape1")` with only `> 0` constraint. The `z3.Not(query)` check then returns `sat` for any symbolic shape because `z_s0=1` violates `% 8 == 0`. Thus `proved = (res == z3.unsat)` is always `False` for non-constant shapes. The fallback never returns True, making Z3 detection dead code. This is a logic bug: you cannot prove a property of unconstrained symbolic integers.
- **Fix**: Constrain symbolic shapes using `z3.Int` variables bound to the actual TIR expressions via `z3_substitute` or use TVM's `arith.Analyzer` to get bound info. If shape is symbolic and not constant-foldable, return False early instead of querying Z3 with unconstrained vars.

## Finding 3
- **Severity**: high
- **File**: `tilelang/transform/metal_simd_lift.py:118`
- **Issue**: `_z3_extent_le_32` creates `z_ext = z3.Int("extent")` but never constrains it to equal `extent_expr`. For symbolic `extent_expr`, the solver only has `z_ext > 0` and checks `Not(z_ext <= 32)`. This is always `sat` with `z_ext=33`, so `proved` is always `False`. The Z3 check for symbolic loop bounds never succeeds, so butterfly rewrite will never trigger for non-constant extents even when the bound is provably `<=32`. 
- **Fix**: Need to model `extent_expr` in Z3. If it's a TIR Var or expression, either skip Z3 or use `tvm.arith.ExtractZ3Vars` to bind it. Otherwise return `(False, "extent symbolic; no model")` explicitly and document the limitation.

## Finding 4
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:73`
- **Issue**: `_is_simdgroup_dtype` treats any string starting with `"e4m3"` or `"e5m2"` as eligible, but FP8 variants like `e4m3fnuz` are not supported by Metal simdgroup matrix. Metal spec only supports `half`, `float`, `bfloat16` in `simdgroup_matrix`, and `int8/uint8` for `simdgroup_multiply_accumulate`. Packed FP8 is not in MSL simdgroup spec. This will misclassify buffers and attempt illegal codegen.
- **Fix**: Remove FP8 from `_SIMDGROUP_DTYPES` and tighten the check:
```diff
-    if s.startswith("e4m3") or s.startswith("e5m2"):
-        return True
     return any(s.startswith(d) for d in _SIMDGROUP_DTYPES)
```
Update `_SIMDGROUP_DTYPES = {"float16", "fp16", "bfloat16", "uint8", "int8"}`.

## Finding 5
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:249`
- **Issue**: `_butterfly_stages` asserts `1 <= shift < 32` but `_SIMD_LANES = 32`. For `extent=32`, `top = 16` and shifts are `[16,8,4,2,1]` which is correct. However, if `extent=64` somehow passes the earlier guard, `top=32` and shift `32` violates the assert. The earlier check `if (extent_val < 2 or extent_val > 32 ...)` should prevent this, but if `extent_val=32` the assert `shift < 32` fails for `shift=32`. The assert is off-by-one: Apple simdgroup supports shuffle with mask `32` for width=32. The comment says `[1, 32)` but actual Metal supports `shift` in `[0,31]` for 32-wide.
- **Fix**: Change assert to `assert 1 <= shift <= 31` or `assert 1 <= shift < _SIMD_LANES`. Also update comment to match Metal spec.

## Finding 6
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:235`
- **Issue**: In `_collect_fragment_gemm_accum_buffers`, `buf = region_call.args[0].buffer` assumes `region_call.args[0]` is a `tir.BufferLoad`. No check before access, but line 234 already checks `isinstance(region_call.args[0], tir.BufferLoad)`. However if `region_call.args` is empty, `region_call.args[0]` raises IndexError before the isinstance check. The guard `len(region_call.args) > 0` is present, so this is safe. But later `accum_vars` path at line 48 uses `call.args[2].args[0]` without guarding `len(call.args[2].args) > 0` in the `if` condition, only in the ternary. If `call.args[2].args` is empty, the ternary still evaluates the condition first, so it's safe. No bug here, but brittle. 
- **Fix**: No functional bug, but for robustness, add early continue if `not region_call.args`.

## Finding 7
- **Severity**: low
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:446`
- **Issue**: `new_attrs["tl.simdgroup_matrix_rewrite_rejected"] = tir.StringImm(";".join(rejection_log))` writes potentially unbounded string attribute to PrimFunc. For large kernels with many ineligible buffers, the attribute string can exceed TVM's attribute size limits and bloat IR. This is debug-only but risky.
- **Fix**: Cap the number of logged rejections or only store count: `new_attrs["tl.simdgroup_matrix_rewrite_rejected_count"] = tir.IntImm("int64", len(rejection_log))`.

## Finding 8
- **Severity**: low
- **File**: `tilelang/transform/metal_simd_lift.py:131`
- **Issue**: `_classify_reduce_op` returns `"add"` for `tir.Sub`. Comment says "sub-into-acc still uses simd_sum-style lowering", but `simd_sum` cannot express `a - b` across lanes. Butterfly using `shfl_xor_sync` with `add` will compute sum, not subtraction. This will miscompile `acc = acc - x` to `acc = acc + x`. 
- **Fix**: Remove `tir.Sub` case or return `None`. Subtraction reductions are not supported by `simd_reduce_sum`.

No other correctness bugs visible in this chunk.