---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: dfcb37dcdfb11bcafebb0a3bfe137d02ad86dfd1
timestamp: 2026-05-07T03:40:16.340227+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
Based on chunk 1/6 only. No runtime hot-path regressions found in this chunk — changes are compile-time only. There are compile-time performance risks from new Z3 prover usage.

## Finding 1
- **Severity**: medium
- **File**: `src/op/copy.cc:550`
- **Issue**: `Z3ProveStrideAligned16` constructs a new `arith::Z3Prover(analyzer)` on every call, then pushes/pops constraints. `CheckGlobalStrides` invokes it once per non-innermost stride. For a tensor with `d` dims, this creates `d-1` independent Z3 contexts. Z3 context setup is heavy: it allocates a solver, registers bitvector sorts, and does native interop. At 50ms timeout each, worst-case compile time grows linearly with rank.
- **Fix**: Instantiate `Z3Prover` once per `CheckGlobalStrides` call and reuse it. Push all per-stride constraints into one solver scope, or batch the alignment goals into a single `CanProve(And(...))` call. 
```cpp
arith::Z3Prover z3(analyzer);
z3.SetTimeoutMs(50);
for (size_t i = 1; i < strides.size(); ++i) {
  // build constraints...
  if (!z3.CanProve(goal)) return false;
}
```

## Finding 2
- **Severity**: low
- **File**: `src/transform/drop_provable_bound_checks.cc:175`
- **Issue**: For each `IfThenElse` candidate, the pass collects all free `Var`s, then pushes two constraints per var into a `recover_stack` of lambdas, calls `CanProve`, then pops in reverse. This allocates `2*V` `std::function` objects and does `2*V` solver push/pop operations per bound check. In functions with many guards, compile time scales with `num_checks * num_free_vars`.
- **Fix**: Push a single conjunctive constraint `And(0<=v1<2^31, 0<=v2<2^31, ...)` once instead of `2*V` separate calls. Reuse a single `Z3Prover` instance across all `IfThenElse` nodes in the pass, not one per node.

## Finding 3
- **Severity**: low  
- **File**: `src/transform/auto_double_buffer.cc:216`
- **Issue**: When `tl.auto_double_buffer=True`, the stub pass still constructs `arith::Z3Prover(analyzer).CanProve(...)` for every detected candidate loop, even though it never transforms IR. Detector runs on all `For` nodes. This adds unnecessary solver overhead to compile time when the feature is enabled for experimentation.
- **Fix**: Gate the `Z3Prover` construction behind a second flag or only run after heuristics strongly suggest profitability. For the stub, remove the `CanProve` call entirely and just log detection.

No runtime regressions, O(n^2) loops, allocations in tight loops, or N+1 I/O patterns visible in this chunk. The metal_fragment_to_simdgroup / metal_simd_lift changes that drop symbolic `z3.Int` paths are not in this diff slice, so cannot verify Idea-8/9 soundness yet.

## Finding 1
- **Severity**: critical
- **File**: `src/transform/loop_vectorize.cc:45-78`
- **Issue**: New `TupleHashMix` and `VectorizeMemoKeyHash`/`AlignmentMemoKeyHash` use custom bit mixing with rotates and shifts. The implementation writes `TupleHashMix(seed, std::hash<const void *>{}(std::get<1>(k)));` with an extra `)` at the end of the line. That syntax error will break compilation for any build that instantiates `VectorizeMemoKeyHash`. Also `TupleHashMix` mutates `seed` with both `^=` and `+=` operations but is marked `inline` without `noexcept`, and is called in hash functors that must be noexcept. If any exception occurs during hash, unordered_map behavior is undefined.
- **Fix**: Remove the stray `)` and mark hash functors `noexcept`. 
```diff
-    TupleHashMix(seed, std::hash<const void *>{}(std::get<1>(k)));
+    TupleHashMix(seed, std::hash<const void *>{}(std::get<1>(k)));
```

## Finding 2
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:180-189`
- **Issue**: `indices_can_vectorize_memo_` is an `unordered_map` keyed by `std::tuple<size_t, const void*, size_t, int>` with custom hash. Clearing it at the end of `Plan` is correct, but the memo is never bounded during `Plan`. For a loop body with `M` distinct `BufferLoad/Store` expressions and a halving probe that tries up to `log2(vector_size_)` sizes, worst case inserts `M * log2(V)` entries. For `M=1000`, `V=128`, that’s 7k entries. Each key holds a `size_t`, pointer, `size_t`, `int` = 32 bytes, plus node overhead ~64 bytes. 7k * 96B ≈ 670KB transient allocation per `Plan` call. If `VectorizePlanner` is reused across many top-level loops, memory spikes. No LRU/eviction.
- **Fix**: Bound the cache size or use `arith::IRDeepHash` with a single `size_t` key instead of tuple. Add:
```cpp
if (indices_can_vectorize_memo_.size() > 10000) indices_can_vectorize_memo_.clear();
```
before `emplace`.

## Finding 3
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:310-330`
- **Issue**: `Z3CanProveLoopAligned` builds a per-call `memo` map with key `tuple<const void*, int, vector<size_t>>`. For kernels with many loads to the same buffer but different index expressions, `vector<size_t>` in the key causes O(K) hash cost where K = number of indices. Hashing the vector each probe is O(K). If a loop body has 200 accesses each with 4 indices, you pay 800 `StructuralHash` calls and 800 vector hashes per `Plan`. No dedup across `Plan` calls because cache is local. This is allocation in a hot path.
- **Fix**: Compute a single structural hash of the full `Array<PrimExpr>` once, not per-element. Replace:
```cpp
std::vector<size_t> idx_hashes;
for (const PrimExpr &idx : indices) {
  idx_hashes.push_back(static_cast<size_t>(hasher(idx)));
}
auto key = std::make_tuple(static_cast<const void *>(buf.get()), vector_size, std::move(idx_hashes));
```
with:
```cpp
size_t idx_hash = hasher(indices); // hash the whole Array node
auto key = std::make_tuple(static_cast<const void *>(buf.get()), vector_size, idx_hash);
```
and update `AlignmentMemoKeyHash` accordingly.

## Finding 4
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:234-267`
- **Issue**: `Z3CanProveAlignedAccess` creates a fresh `Z3Prover` and pushes `EnterConstraint` scopes for every free var on every call. For a loop with 50 distinct global loads, each call spins up Z3, allocates a context, and runs `CanProve`. With `SetTimeoutMs(50)`, worst case adds 50 * 50ms = 2.5s compile time per loop. No sharing of Z3 context across calls, so solver initialization cost repeats. `ConstraintScope` destructors do `solver.pop()` but Z3 context is still torn down/rebuilt each call.
- **Fix**: Hoist `Z3Prover` to `VectorizePlanner` member and reuse. Batch constraints. Add early exit if `!PassContext::Current()->GetConfig<Bool>("tl.vectorize_alignment_proof")` before collecting free vars.

## Finding 5
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:402-415`
- **Issue**: `IsAffineInVar` does `PostOrderVisit` on every `expr` to check for `BufferLoadNode`. For complex index expressions like `((i*32 + j)*128 + k)*4 + base`, this walks the entire AST. Called from `Z3CanProveUnitStride` which is itself called inside the halving loop of `IndicesCanVectorize`. So you get O(N * H * L) AST walks where N=expr size, H=halving steps, L=number of loads. No memoization of affine-check result.
- **Fix**: Cache `IsAffineInVar` result per `StructuralHash(expr)` in a `DenseMap`. Return cached bool before `PostOrderVisit`.

## Finding 6
- **Severity**: medium
- **File**: `src/transform/predicate_fusion.cc:120-148`
- **Issue**: `Z3ProvesIndexInRange` builds a new `Z3Prover` and `EnterConstraint` for each index dimension of each load. For a store `C[i,j,k] = ...` you launch 3 Z3 queries. With timeout 50ms, a 3-D tensor access costs 150ms compile time. `Z3ProvesInnerWellDefined` calls this for every load/store in the inner body. No batching of goals: you could conjoin all `idx >= 0 && idx < extent` into one `CanProve` call. Current design causes N*D solver calls.
- **Fix**: Accumulate all `idx` goals into a single `PrimExpr` using `&&` and call `CanProve` once per `PredicateFuser` invocation. Reduces Z3 context switches.

## Finding 7
- **Severity**: low
- **File**: `src/transform/loop_vectorize.cc:166-173`
- **Issue**: `indices_can_vectorize_memo_.clear()` at end of `Plan` is correct for soundness, but it prevents reuse across multiple `Plan` calls on different loops in the same function. If a function has 10 vectorizable loops with identical access patterns, you redo all Z3 work 10 times. Comment says cache is "per-planner-instance", but planner is constructed per function in `VectorizeLoop`. So cache lifetime is too short.
- **Fix**: Move `indices_can_vectorize_memo_` to `VectorizeLoop` and pass by reference, or keep it and only clear when `analyzer_` pointer changes. Document tradeoff.

## Finding 8
- **Severity**: info
- **File**: `src/transform/thread_storage_sync.cc:184-200`
- **Issue**: `ProveIntraWarpRAW` runs Z3 with `SetTimeoutMs(200)` for every shared-memory RAW/WAR pair. For reductions with many partial sums, pairwise checks are O(P^2). Each check spins Z3. No early exit when `!is_metal_`, which is good, but Metal kernels with 100 shared accesses could add 20s compile time. No metrics to see if this path triggers.
- **Fix**: Add `PassContext` counter `tl.apple_intra_warp_elisions_` already exists. Gate with `PassContext::GetConfig<Bool>("tl.metal_intra_warp_elision")` default false, to avoid surprise compile time hits.

**Verification of e0402d77 + dfcb37dc changes**: 
Diff shows `metal_fragment_to_simdgroup` and `metal_simd_lift` not in this chunk, so cannot verify "drop unconstrained z3.Int symbolic paths". 

**Verification of Idea-8/9 soundness**: 
`Z3CanProveUnitStride` now rejects non-affine expr via `IsAffineInVar` before Z3, preventing false positive on `A[B[i]]`. `Z3ProvesIndexInRange` uses `BVBoundsForDtype` instead of flat `[0,2^31)`, fixing negative index unsoundness. When flags ON, these paths are sound per diff.

## Finding 1
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:189`
- **Issue**: `CreateSolver` hard-codes `solver.set("random_seed", (unsigned)42)` and is called from both `RebuildSolver_()` and `Reset()`. Every `SetBitVectorMode()` change or `Reset()` call rebuilds a new Z3 solver with a fixed seed. For hot passes that flip BV mode per function or per CanProve, this forces Z3 to discard its learned clauses and start cold on each rebuild. With the new `ClearProverCache()` called at pass entry, plus the `SetBitVectorMode` fast-path, rebuilds should be rare. But if a pass toggles modes, e.g. Int→BV32→Int, each toggle incurs full solver recreation + re-seeding. That’s a context switch + allocation spike in tight proof loops. 
- **Fix**: Cache `::z3::solver` objects per `bv_width_` inside `Z3ProverImpl` and reuse them instead of calling `CreateSolver` on every mode switch. Only call `solver.reset()` and re-assert global bounds. Preserve the seed but don’t re-instantiate the solver unless `ctx` changes. Example:
  ```cpp
  std::unordered_map<int, ::z3::solver> solvers_;
  ::z3::solver& GetSolver() {
    auto& s = solvers_[bv_width_];
    if (s.ctx() != ctx.get()) s = CreateSolver(*ctx); // first use
    else s.reset(); // clear assertions, keep learned clauses if desired
    s.set("timeout", timeout_ms);
    s.set("rlimit", rlimit);
    return s;
  }
  ```

## Finding 2
- **Severity**: low
- **File**: `src/transform/vendored/z3_prover.cc:339`
- **Issue**: `CanProve` wraps each query in `try { solver.check(constr) } catch(...)`. Z3 can throw on resource exhaustion or internal errors. Catching and logging per-query adds exception-unwind overhead in the hot path. `LOG(WARNING)` also allocates. For passes like `DropProvableBoundChecks` that call `CanProve` on every bound-check site, this adds measurable cost when Z3 is compiled with exceptions enabled. No NEW regression vs prior stub, but the new exception path is now exercised by Idea-8/9.
- **Fix**: Move the `try/catch` to a single outer wrapper in the pass, or set a global Z3 error handler via `Z3_set_error_handler` and avoid C++ exceptions. For per-query, check `solver.reason_unknown()` after `check()` instead of relying on exceptions. Remove logging in release builds.

## Finding 3
- **Severity**: info
- **File**: `src/transform/vendored/z3_prover.cc:401`
- **Issue**: `Bind(const Var&, const Range&)` does `memo_.emplace(var, var_expr)` on every range bind, then may `solver.add()` 2 constraints. For BV32 mode with many induction vars, this duplicates memory: `memo_` holds the Z3 expr, `solver` holds assertions. In the old Int-only path this was already true. The new `clamped_min/clamped_max` path adds an extra `MakeIntVal` call per bound, creating temporary Z3 AST nodes each time. If a pass binds 100s of loop vars, AST node churn grows linearly. Not O(n^2), but allocation in a tight loop.
- **Fix**: Cache `MakeIntVal(lo)` and `MakeIntVal(hi)` per `bv_width_` as members, since they are constants for BV32/BV64. Reuse instead of constructing new `bv_val` nodes each bind.

## Finding 4
- **Severity**: info
- **File**: `src/transform/vendored/z3_prover.cc:528`
- **Issue**: `CountSatisfyingValues` loops with `while (count < max_count)` and on each iteration does `solver.add(z3_var != MakeIntVal(val))`. Each `add` clones the assertion stack. For `max_count` large, this is O(m^2) in solver size, not just iterations. The old code had the same pattern, but now it’s called by more passes after Idea-8/9 enablement. Risk of blowup if a pass probes with `max_count=1000`.
- **Fix**: Use `solver.push()` before the loop and `solver.pop()` after. Inside loop, assert the disequality, check, then `pop(1)` to remove only the last disequality. That keeps assertion stack depth O(1) instead of O(m). 

### Verification requested

1. **No NEW regressions introduced by e0402d77 + dfcb37dc**: This chunk only touches `z3_prover.cc/h` and tests. No loops, I/O, or allocations in hot paths beyond the solver-rebuild and exception issues noted above. No N+1 queries or blocking calls added. The BV helpers `MakeIntVal/MakeUIntVal` are inline and branch-predictable. 

2. **Idea-8/9 sound when flags ON**: 
   - Idea-8 `DropProvableBoundChecks`: Now uses `Z3Prover` with BV32 guards. The new `Bind` logic clamps out-of-range binds and asserts `false` for empty ranges `src/transform/vendored/z3_prover.cc:425`. That prevents the unsound “prove i < ceildiv(N,K)*K + 1 near INT_MAX” case. Sound.
   - Idea-9 `metal_simd_lift` path: Diff chunk doesn’t show `metal_fragment_to_simdgroup.py` or `metal_simd_lift.py` changes, only z3 infra. From this chunk, the prover now rejects unconstrained `z3.Int` when BV mode is on because `MakeIntConst` returns BV sort. Combined with the commit message “drop unconstrained z3.Int symbolic paths”, Idea-9’s requirement that symbolic paths be gated by `static IntImm` is satisfied at the prover level. Sound, assuming the Python side no longer feeds `z3.Int` when flags are on.

No other performance-impacting changes visible in chunk 3 of 6.

No findings.

This chunk contains only new test files:
- `testing/python/transform/test_intra_warp_2d_launch.py`
- `testing/python/transform/test_loop_vectorize_z3_contiguity.py`
- `testing/python/transform/test_predicate_fusion.py`
- `testing/python/transform/test_simd_reduction_lift.py`
- `testing/python/transform/test_simd_reduction_rewrite.py`
- `testing/python/transform/test_simdgroup_matrix_detection.py`
- `testing/python/transform/test_simdgroup_matrix_rewrite.py`
- `testing/python/transform/test_tma_legality.py`

There are no production code changes, no loops, I/O, allocations, or hot-path logic introduced in this diff chunk. No performance regressions present in the visible code.

No performance regressions or hot-path concerns in this diff chunk.

**Summary**  
Chunk 5 of 6 only adds/modifies test files, analysis utilities, and pass plumbing. No new runtime loops, I/O, allocations, or blocking calls introduced in hot paths. All changes are additive behind PassConfig flags defaulting to OFF.

## Finding 1
- **Severity**: info
- **File**: `testing/python/transform/test_vectorize_alignment.py:L1-L318`
- **Issue**: New test file adds extensive alignment-proof tests. Test code is not executed in production kernels. No runtime cost.
- **Fix**: None needed. This is test-only code.

## Finding 2
- **Severity**: info
- **File**: `testing/python/transform/test_z3_bv_mode.py:L1-L271`
- **Issue**: Z3 prover tests invoke solver with 50ms timeout. The timeout bounds worst-case per-call cost. Calls are test-only and not in compiler hot path unless `tl.z3.bv_can_prove` is used in a pass. Passes here gate Z3 behind config flags.
- **Fix**: None. Existing timeout already prevents runaway solves. Verify production passes keep `tl.vectorize_alignment_proof`, `tl.drop_provable_bound_checks`, etc OFF by default.

## Finding 3
- **Severity**: info
- **File**: `tilelang/analysis/int24_overflow_proof.py:L105-L139`
- **Issue**: Static fast path for `prove_dot4_int24_safe` uses O(1) integer multiply: `bound = k_int * x_max * y_max`. No loops or allocations. Symbolic path falls back to Z3 with 50ms timeout and `K <= 1<<16` bound, preventing unbounded solve time.
- **Fix**: None. Timeout + K upper bound quantifies worst-case: ~50ms per unique symbolic K. Call site must still memoize to avoid N queries.

## Finding 4
- **Severity**: info
- **File**: `tilelang/engine/phase.py:L13-L26`
- **Issue**: Adds `_Z3_CLEARED_COMPILE_IDS: set[int]` to track per-compile Z3 cache clears. Set grows with number of distinct `id(mod)` seen during a process lifetime. Keys are integers, discarded after `OptimizeForTarget` consumes them. Memory impact: 28 bytes per compile, negligible unless compiling millions of modules in one process without GC.
- **Fix**: None required. If long-lived daemons compile >>1M modules, add periodic `clear()` of stale ids, but current design discards on consume.

## Finding 5
- **Severity**: info
- **File**: `tilelang/language/fp8_op.py:L244-L277`
- **Issue**: `_z3_prove_dot4_legal` static path evaluates predicate with only scalar Python ops. Symbolic path constructs one Z3 solver per call with timeout 50ms. No caching shown in this chunk, but `test_indices_can_vectorize_memoized_halving` and `test_alignment_proof_repeated_access` indicate memoization exists elsewhere. Quantified impact: without memo, repeated `(buffer, indices)` pairs would pay 50ms each. With memo per fix-B5/B6, repeated access is O(1) hash lookup.
- **Fix**: Ensure callers use the memoized planner entry points. This chunk only adds the proof function; no regression.

## Finding 6
- **Severity**: info
- **File**: `tilelang/transform/__init__.py:L179-L229`
- **Issue**: New passes `DropProvableBoundChecks`, `AutoDoubleBuffer`, `PredicateFusion` are registered. Per docstrings, all are gated by PassConfig default OFF. `AutoDoubleBuffer` is explicitly a stub that logs but does not transform IR, so no compile-time or runtime cost added yet.
- **Fix**: None. Keep defaults OFF until memoized Z3 + benchmarking lands.

**Verified per prompt**  
1. No NEW regressions in visible code: All additions are tests, config plumbing, or guarded by OFF flags.  
2. Idea-8/9: `tilelang/transform/metal_fragment_to_simdgroup.py`, `metal_simd_lift.py` not in this chunk. Per note, commits e0402d77 + dfcb37dc drop unconstrained `z3.Int` paths there. No unconstrained symbolic Z3 usage visible here.  
3. No O(n^2) loops, redundant I/O, N+1 queries, blocking async calls, tight-loop allocations, or large sync payloads introduced.

## Finding 1
- **Severity**: high
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:115`
- **Issue**: `_z3_simdgroup_eligible` claims to be a Z3 fallback but performs only static Python checks and never invokes Z3. The function returns `True` for static shapes without proving anything about symbolic inputs, and explicitly returns `False, "symbolic shape rejected"` for any non-IntImm. This contradicts the docstring and Idea #8 description. If the flag `tl.simdgroup_matrix_rewrite` is ON, legitimate symbolic buffers that *could* be proven `% 8 == 0` via Z3 are conservatively skipped. That means the simdgroup path is disabled and code stays on `local.fragment` scalar lowering, which is a performance regression vs the intended Z3-gated optimization.
- **Fix**: Either implement the Z3 query using `tvm.tir.analysis.z3`/`tilelang.transform.vendored.z3_prover` to bind `s0/s1` expressions, or update the docstring and logging to reflect that only static IntImm shapes are supported. If Z3 is unavailable, keep the static check but rename function to `_static_simdgroup_eligible_with_addr` to avoid misleading callers.

## Finding 2
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:97`
- **Issue**: `_z3_extent_le_32` similarly advertises a Z3 fallback but immediately returns `False` for any symbolic extent without calling Z3. Comment says “Reject symbolic extents conservatively without spinning up z3”. This makes `tl.simd_lift_reductions` unable to rewrite reductions where `tile_extent` is a TIR Var but provably `<= 32` via range analysis or Z3. When `tl.simd_lift_reductions=ON`, performance wins from butterfly `shfl_xor_sync` are missed, leaving threadgroup-memory reductions in the hot path.
- **Fix**: Use `tir.analysis.estimate_extent` + Z3 prover to handle symbolic `node.extent`. If extent has known upper bound `<= 32`, return `proved=True`. Otherwise keep conservative False. Or update docs to state static-only support so users don’t expect symbolic handling.

## Finding 3
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:211`
- **Issue**: `_collect_fragment_gemm_accum_buffers` walks the IR with `post_order_visit` and for each GEMM call inspects `call.args[2].args[0]` to extract `BufferLoad.buffer`. This is O(number of stmt nodes) with no caching. For large Metal PrimFuncs with many GEMM ops, the visitor runs on every pass invocation. Since this pass is `opt_level=0` and runs unconditionally when target=metal, repeated IR walks can add measurable compile-time overhead, especially when `tl.simdgroup_matrix_rewrite` is OFF and the result is discarded.
- **Fix**: Gate the collection behind `if rewrite_gated:` before the call on line 262. When the flag is OFF, skip the walk entirely since `_collect_fragment_gemm_accum_vars` is already called for the legacy path. Also consider caching the accumulator map in a pass context attribute.

## Finding 4
- **Severity**: low
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:180`
- **Issue**: Logging inside `_collect_fragment_gemm_accum_vars` at line 224-230 checks `os.environ.get("TL_LOG_SIMDGROUP")` for every GEMM accumulator found. The env lookup happens in the tight IR walk. On large IR, repeated `os.environ.get` and string formatting for `shape`/`dtype` tuples adds CPU overhead even when logging is disabled. This is a hot-path compile-time cost, not runtime, but still a regression vs shipping pass.
- **Fix**: Lift the env check out of the visitor: `log_enabled = os.environ.get("TL_LOG_SIMDGROUP")` once before `post_order_visit`. Or remove detection-time logging entirely and rely on the attribute `tl.simdgroup_matrix_rewrite_emitted` for post-hoc inspection.

## Finding 5
- **Severity**: info
- **File**: `tilelang/transform/metal_simd_lift.py:375`
- **Issue**: `_ButterflyRewriter._mutate` recursively rebuilds the entire TIR Stmt tree even when no annotated loops exist. For PrimFuncs without `tl.simd_butterfly_lane`, the pass does O(#nodes) copy work with no IR change. When `tl.simd_lift_reductions=ON`, this runs on every Metal func and increases compile time linearly in IR size.
- **Fix**: Early-exit if `any(c.annotated and c.proved for c in candidates)` is False after `_walk_reductions`. Skip `_ButterflyRewriter` instantiation. The existing check at line 472 already does this, so no regression. Marking as info to confirm the guard is present.

No new N+1 queries, blocking I/O, allocation in tight loops, or memory growth introduced in this chunk. The simdgroup and simd_lift passes are compile-time only and gated by `PassConfig` flags defaulting OFF, so no runtime regression for existing workloads. Idea-8/9 are sound when flags ON because both now require static `IntImm` and explicit annotations, preventing unsound Z3-free “proofs” from triggering rewrites.