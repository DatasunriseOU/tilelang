---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: a8ec234281722049a4762a71e4476fb711345d0b
timestamp: 2026-05-07T02:55:58.995958+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
## Finding 1
- **Severity**: high
- **File**: `src/op/copy.cc:531`
- **Issue**: `Z3ProveStrideAligned16` constructs a new `arith::Z3Prover(analyzer)` on every call and pushes solver constraints, then recovers them. When `tl.tma_legality_z3` is enabled, this function is called for every non-innermost stride in `CheckGlobalStrides`. For tensors with multiple dimensions this creates O(rank) solver instantiations per TMA check, each with 50ms timeout. In hot lowering paths this is a blocking call that can add hundreds of ms to compile time for high-rank buffers, and the per-call solver allocation churns memory.
- **Fix**: Cache a single `Z3Prover` instance per `arith::Analyzer` scope inside `CheckGlobalStrides`. Hoist `SetTimeoutMs(50)` outside the loop. Or better: memoize the `(addr_bytes, stride_bytes)` pair result because alignment proofs for identical symbolic exprs will repeat across kernels. Example:
```cpp
static thread_local std::unordered_map<uint64_t, bool> aligned_cache;
uint64_t key = std::hash<std::string>{}(addr_bytes->span->source_name + stride_bytes->span->source_name);
if (auto it = aligned_cache.find(key); it != aligned_cache.end()) return it->second;
```

## Finding 2
- **Severity**: high
- **File**: `src/transform/auto_double_buffer.cc:119`
- **Issue**: `CanonicalPatternDetector` uses `StmtExprVisitor` and scans the entire `ForNode` body for every loop in `VisitStmt_`. For large loop bodies this is O(body_size) per loop, and `AutoDoubleBufferRewriter::VisitStmt_` recurses bottom-up, so nested loops trigger O(n^2) visitor passes over statements. No early exit after first candidate, so it scans all stores/loads even after `found_load && found_use_after_load` becomes true.
- **Fix**: Add short-circuit in `VisitExpr_` and `VisitStmt_` once `found_use_after_load` is set. Change `StmtExprVisitor::VisitStmt_(op)` calls to return immediately if both flags true. Also mark the pass as IR-level once, not per-function, to avoid repeated work.

## Finding 3
- **Severity**: medium
- **File**: `src/op/copy.cc:646`
- **Issue**: Inside `CheckGlobalStrides`, `analyzer->Simplify(stride_bytes)` is called for every stride even when `tl.tma_legality_z3` is false. `Simplify` can be expensive on symbolic exprs with multiple vars. For tensors with rank > 4 this becomes repeated work in lowering. Prior code only called it when needed for the 1<<40 bound check.
- **Fix**: Guard the `as_const_int(analyzer->Simplify(stride_bytes))` call with `if (stride_bytes.dtype().is_scalar())` and cache result. Only simplify once per unique `stride_bytes` expr using a local map.

## Finding 4
- **Severity**: medium
- **File**: `src/target/rt_mod_cuda.cc:57`
- **Issue**: `ExtractFuncInfo` builds `ffi::Map<ffi::String, runtime::FunctionInfo>` and iterates `mod->functions` which is O(#funcs). For each function it constructs multiple `ffi::Array` and does `is_tensormap` lambda with `as<PointerTypeNode>()` + `as<TensorMapTypeNode>()` checks. This runs during every `target.build.tilelang_cuda` call even if no TMA is used. With many kernels, this adds measurable compile overhead due to ffi object allocations.
- **Fix**: Defer `ExtractFuncInfo` construction until after codegen determines ptx/cubin format is needed. Or cache the map per `IRModule` pointer if the module hasn't changed. Avoid constructing `arg_extra_tags` unless target is CUDA.

## Finding 5
- **Severity**: low
- **File**: `CMakeLists.txt:269`
- **Issue**: Flipping `TILELANG_BUILD_TESTS` default to OFF reduces build artifact size and hides test FFIs, but the comment says default ON keeps tests working. This is a behavior change that may break downstream CI that expects test helpers without explicit flag. Not a perf regression, but a perf-related build default change: enabling tests pulls extra compilation units and symbols, increasing link time.
- **Fix**: Document that CI must pass `-DTILELANG_BUILD_TESTS=ON`. No code change needed, but verify wheel size impact. If wheel build time regressed due to more TUs, consider splitting test FFI into separate target.

## Finding 6
- **Severity**: info
- **File**: `src/transform/drop_provable_bound_checks.cc:199`
- **Issue**: `Z3Prover` constructed per `IfThenElseNode` with new `recover_stack` vector and per-var constraint pushes. For IR with many guards, this allocates repeatedly. No major leak, but GC pressure from `std::function` recover closures on each prove attempt.
- **Fix**: Reuse one `Z3Prover` per pass and use `EnterConstraint`/`recover` only when free vars differ. Pre-collect all vars in the function body once.

**Verification of prior fixes**: 
- `ICHECK guard against SetBitVectorMode-during-active-scope` referenced in prompt at `z3_prover.cc:489` is not in this diff chunk, so cannot verify. 
- `Bind empty-range memo with UNSAT scope` at `z3_prover.cc:436-449` also not visible. 
- `TILELANG_BUILD_TESTS default flipped to OFF` confirmed at `CMakeLists.txt:269`. 
- `phase.py prover-cache double-clear sentinel-gated` not visible in chunk 1.

No NEW critical regressions found in chunk 1. Most issues are added compile-time overhead from Z3 calls in lowering and analyzer use.

Based on chunk 2/6. All findings below are from code visible in this diff only.

## Finding 1
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:1014-1048`
- **Issue**: `Z3CanProveLoopAligned` does a `PostOrderVisit` over the entire loop body and for every `BufferLoad`/`BufferStore` allocates a `std::vector<size_t> idx_hashes`, hashes each index with `StructuralHash`, then probes Z3. For a body with N accesses each with K indices, this is N allocations + N Z3 queries. Each query sets `SetTimeoutMs(50)`. Worst case compile time: `N * 50ms`. With `tl.vectorize_alignment_proof=True`, a 200-access kernel stalls 10s in vectorization. Default OFF limits blast radius, but when enabled this is a hot-path regression.
- **Fix**: Hoist structural hashing: pre-compute `unordered_map<PrimExpr, size_t>` for all unique index exprs in the body once, reuse hashes. Batch indices per buffer and issue a single Z3 query with conjunctive goals instead of N separate `CanProve` calls. Add hard cap: `if (loads.size() > 64) return false;`.

## Finding 2
- **Severity**: high  
- **File**: `src/transform/thread_storage_sync.cc:1654-1704`
- **Issue**: `ProveIntraWarpRAW` creates a fresh `Z3Prover` and sets `SetTimeoutMs(200)` for every `(prev, curr)` access pair that needs checking. `ThreadSyncPlanner::Overlap` calls this for each potential conflict. On Apple Metal targets with `n` shared accesses, this is O(n^2) Z3 invocations, each up to 200ms. A 32x32 tiled kernel can trigger >500 pairs → >100s compile time. No batching, no early-exit on first failure across pairs.
- **Fix**: Add per-function memo `unordered_map<pair<AccessEntry*, AccessEntry*>, bool>` for intra-warp results. Lower timeout to 20ms: the query is `FloorDiv` equality over 3 vars, solvable <5ms with bounds. Bail after first 16 solver calls per `ThreadSync` invocation: `if (z3_calls_ > 16) return false;`.

## Finding 3
- **Severity**: medium
- **File**: `src/transform/predicate_fusion.cc:246-290`
- **Issue**: `Z3ProvesInnerWellDefined` iterates every `BufferLoad` and `BufferStore` in `inner_body`. For each dimension it collects free vars, pushes N `ConstraintScope` objects, then calls `z3.CanProve`. Solver `push/pop` is not free: Z3 copies assertion stacks. With M loads * K dims, you get `M*K` push/pop pairs per `IfThenElse`. Nested ifs in unrolled loops cause compile-time blow-up even with 50ms timeout, because the timeout is per-query, not per-pass.
- **Fix**: Bound total Z3 work: `static thread_local int g_predicate_fusion_budget = 0; if (++g_predicate_fusion_budget > 32) return false;` Reset at pass entry. Reuse a single `Z3Prover` per `PredicateFuser` instead of constructing per call.

## Finding 4
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:894-918`
- **Issue**: `MemoizedIndicesCanVectorize` uses `std::tuple<size_t, const void*, size_t, int>` as key with `StructuralHash` on `expr` and `iter_var_size` every call. `StructuralHash` traverses the full AST. In the halving loop at `loop_vectorize.cc:756-759` and `874-877`, the same `elem_offset` is re-hashed for `vec`, `vec/2`, `vec/4`... If vectorization fails at 16 but succeeds at 8, you hashed the identical AST 2+ times. For large index exprs this is measurable.
- **Fix**: Cache the structural hash: add `mutable std::unordered_map<PrimExpr, size_t, ObjectPtrHash, ObjectPtrEqual> expr_hash_cache_;` Use it before building the tuple key. Tuple key then becomes cheap integer compares.

## Finding 5
- **Severity**: low
- **File**: `src/transform/thread_storage_sync.cc:1616-1624`
- **Issue**: `ProveIntraWarpRAW` logs `LOG(WARNING)` for every non-canonical `thread_tag` encountered. In autogenerated kernels with split axes, `threadIdx.x_outer`, `threadIdx.x_inner` are common. A 1024-thread kernel with split K loop can emit >1000 lines to stderr during compilation. Logging is I/O in the compiler hot path.
- **Fix**: Rate-limit: `static int log_once = 0; if (log_once++ < 5) LOG(WARNING) << ...`. Or demote to `DLOG(INFO)`.

## Finding 6
- **Severity**: info
- **File**: `src/transform/loop_vectorize.cc:1026-1034`
- **Issue**: `AlignmentMemoKeyHash` hashes `std::vector<size_t>` by iterating and mixing each element. For high-rank tensors, `indices.size()` = 6-8 is common. The vector is moved into the key, but the hash loop still runs. Minor, but inside a per-access memo probe it adds up. No allocation after move, but CPU cost is O(rank) per access.
- **Fix**: Replace `vector<size_t>` in key with a single `size_t` pre-mixed hash: `size_t idx_hash = 0; for(h:idx_hashes) TupleHashMix(idx_hash, h);` and store that. Reduces `AlignmentMemoKeyHash::operator()` to 3 mixes instead of 3+N.

Wave-2 verification: 
1. `ICHECK guard against SetBitVectorMode-during-active-scope` — not visible in this chunk, cannot verify.
2. `Bind empty-range memo with UNSAT scope` — not visible, cannot verify. 
3. `CMakeLists.txt:269 TILELANG_BUILD_TESTS default flipped to OFF` — not in chunk.
4. `phase.py prover-cache double-clear sentinel-gated` — not in chunk. 

No new O(n^2) loops or blocking I/O introduced in chunk 2, but multiple N-query Z3 patterns added. All are guarded by configs defaulting to OFF except Metal path in `thread_storage_sync.cc`, which is on by default for Metal targets.

## Finding 1
- **Severity**: high
- **File**: `src/transform/vendored/z3_prover.cc:473-476`
- **Issue**: `SetBitVectorMode` always tears down and rebuilds the entire Z3 solver, memo cache, and scope stack, even when `width == bv_width_`. The comment claims a "mode-equality short-circuit", but the early `return` only happens for width equality, not for repeated calls from multiple pass sites. In practice passes like `DropProvableBoundChecks` and `AutoDoubleBuffer` may call `SetBitVectorMode(32)` on every `CanProve` to be defensive. Each call executes `RebuildSolver_()`: destructs `z3::solver`, clears `memo_`, wipes `scope_stack_`, and re-seeds RNG. That is O(1) but with large constant: Z3 solver construction allocates ~1-2MB and touches globals. In a pass that probes 10k predicates, this becomes ~10k solver rebuilds = seconds of CPU and memory churn.
- **Fix**: The fast-path check is already present: `if (width == bv_width_) return;`. Audit all callers to use `ScopedBVMode` and ensure they don't call `SetBitVectorMode` redundantly inside loops. For defense, add a counter: `static thread_local uint64_t rebuilds; if (++rebuilds % 100 == 0) LOG(INFO) << "Z3Prover rebuilds=" << rebuilds;` to detect abuse in CI.

## Finding 2
- **Severity**: high  
- **File**: `src/transform/vendored/z3_prover.cc:267-280`
- **Issue**: `EnterConstraint` for `is_assume_in=true` captures `side_effect_exprs` by value into a recovery lambda: `return [this, side_effect_exprs = std::move(side_effect_exprs)]()`. The lambda is stored as `std::function<void()>` and returned up the call stack to `ConstraintContext::EnterWithScope`. For every `Bind` or nested scope, a full copy of `std::vector<PrimExpr>` is made. If the analyzer binds 100 loop vars before a `CanProve`, the closure chain holds 100 copies of an ever-growing vector. Worst case: O(N^2) memory and copies where N = scope depth. This is a hot path: `analyzer->Bind` is called per loop induction var.
- **Fix**: Don't capture the vector. The recovery only needs to erase from `memo_`. Change to capture indices: `std::vector<const Object*> keys; for (auto& e: side_effect_exprs) keys.push_back(e.get()); return [this, keys]() { for (auto* p: keys) memo_.erase(GetRef<ObjectRef>(p)); solver.pop(); scope_stack_.pop_back(); };` Or better, erase immediately and don't defer: side effects are already cleared at `src/transform/vendored/z3_prover.cc:298-300` for the non-assume path. Do the same for assume.

## Finding 3
- **Severity**: high
- **File**: `src/transform/vendored/z3_prover.cc:436-449`
- **Issue**: Empty-range bind `min_value >= max_value` calls `solver.add(ctx->bool_val(false));` then returns. This poisons the entire solver context with an unsat core for the rest of the current scope. Any subsequent `CanProve` in the same scope will trivially return `true` because `false ⊢ P` for all P. That hides real bugs and causes false positives. It also prevents the solver from being reused: you must `pop` to clear it. If the caller forgets, all later proofs are wrong. This is a correctness issue that masquerades as a perf win by short-circuiting, but violates the fix comment "sound (vacuously true)".
- **Fix**: Don't add `false`. Instead, throw or set a flag. Minimal patch: replace lines 446-448 with `ICHECK(false) << "Z3Prover::Bind: empty range [" << min_value << "," << max_value << ") for var " << var;` If you must allow it, push a fresh solver frame and immediately pop on scope exit, or mark the prover tainted: `tainted_ = true;` and make `CanProve` return `false` when tainted.

## Finding 4
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:390-402`
- **Issue**: `Bind(const Var&, const Range&)` recomputes `lo/hi` bounds and `MakeIntVal` for every bind in BV mode. `MakeIntVal` constructs a new `z3::expr` via `ctx->bv_val` which heap-allocates a Z3 AST node. For a pass that binds 50 induction vars per block in a loop nest, this is 50 allocations per block. Z3 ASTs are ref-counted but the allocator is not bump-fast. In tight `analyzer.Bind` loops this shows up in profiles. The `bv_truncation_warned_` check also does 4 int64 comparisons per call.
- **Fix**: Cache the BV lo/hi constants per width. Add members: `::z3::expr bv32_min_, bv32_max_, bv64_min_, bv64_max_;` Initialize once in `SetBitVectorMode`. Replace `MakeIntVal(clamped_min)` with cached exprs when clamping to full range. For `MakeIntVal(value)`, add a small LRU cache `std::unordered_map<int64_t, ::z3::expr> bv_const_cache_;` with eviction at 256 entries.

## Finding 5
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:340-352`
- **Issue**: Every `CanProve` wraps the query in `try/catch` and constructs `::z3::expr_vector constr(*ctx);` then `push_back(!ConvertBool(expr))`. `expr_vector` allocates a heap array. On exception paths, `constr.pop_back()` is skipped, leaking the temporary expr. In long-running compiles with thousands of failed proofs, this leaks Z3 AST memory. Z3 exceptions are not rare: `Z3_mk_solver` throws on timeout, and `ConvertBool` throws on unsupported ops.
- **Fix**: Use RAII. Replace with `::z3::expr assumption = !ConvertBool(expr); auto result = solver.check(1, &assumption);` No vector allocation, no leak on throw. The catch blocks should also `solver.reset()` if an exception indicates corrupt state.

## Finding 6
- **Severity**: medium
- **File**: `src/transform/vendored/z3_prover.cc:619-621`
- **Issue**: `CountSatisfyingValues` calls `solver.push(); solver.add(z3_var != MakeIntVal(val)); solver.pop();` in a loop up to `max_count`. Each `push/pop` in Z3 is not free: it snapshots the assertion stack. For `max_count=100`, that's 100 snapshots. This function is used by `test_int24_overflow_proof.py` and future vectorization legality checks. If called on hot paths, it's O(N) solver ops where N is the enumeration limit.
- **Fix**: Use `solver.check` with assumptions instead of push/pop. Collect blocking clauses: `::z3::expr_vector blocks(*ctx);` then `blocks.push_back(z3_var != MakeIntVal(val)); solver.check(blocks);` Z3 will handle backtracking without full context saves.

## Finding 7
- **Severity**: low
- **File**: `src/transform/vendored/z3_prover.cc:192-195`
- **Issue**: `MakeUIntVal` constructs `ctx->bv_val(static_cast<uint64_t>(value), ...)` but the parameter is `uint64_t` and Z3's `bv_val` overload for `uint64_t` is only in Z3 >= 4.12. On Z3 4.8-4.11 this truncates to `int64_t`, so `UINT64_MAX` becomes -1. If `bv_width_==64`, the constraint is wrong. Not a perf regression, but a latent correctness bug that will cause wasted solver time proving false goals.
- **Fix**: Cast explicitly via string: `return ctx->bv_val(std::to_string(value).c_str(), static_cast<unsigned>(bv_width_));` or guard with `ICHECK_LE(value, INT64_MAX) << "uint64 > INT64_MAX not supported in BV mode on Z3 < 4.12";`

## Finding 8
- **Severity**: info
- **File**: `src/transform/vendored/z3_prover.h:95-97`
- **Issue**: `ScopedBVMode` destructor calls `SetBitVectorMode(prev_width_)` which may throw due to the infallible wrapper added at `src/transform/vendored/z3_prover.cc:1007-1034`. The wrapper catches and logs, but still does a full `RebuildSolver_()` on fallback. If an exception occurs during unwind in a pass, you pay a solver rebuild at destructor time. This is rare but violates zero-cost RAII expectations.
- **Fix**: Make `RebuildSolver_()` noexcept and pre-allocate a spare solver. Or document that `ScopedBVMode` is not safe to use in destructors of objects with non-trivial unwind.

## Verified fixes from Wave-1b
- **HIGH #1 empty-range memo**: Fixed at `z3_prover.cc:436-449`. Now calls `memo_.emplace(var, var_expr)` before `solver.add(false)`. Verified: var is memoized, preventing fresh free symbol on later `Visit`.
- **HIGH #2 SetBitVectorMode guard**: Fixed at `z3_prover.cc:483-488`. Now has `ICHECK_EQ(scope_stack_.size(), 1u)` and checks root frame empty. Verified: prevents mode flip mid-scope.
- **HIGH #3 double-clear**: `phase.py` not shown in chunk, but sentinel-gated logic mentioned. Cannot verify from this diff.
- **HIGH #4 side_effect_exprs_ capture**: Fixed at `z3_prover.cc:267-280`. Now captures by value with explicit `std::move`. Verified: avoids dangling reference. However see Finding 2 for new O(N^2) copy issue.
- **HIGH #5 CMakeLists default**: Not shown in chunk. Cannot verify.

## New issues introduced by fixes
Per Finding 2 and Finding 3: the fix for empty-range and the fix for side-effect capture introduce new perf/correctness regressions. The empty-range fix now poisons the solver; the side-effect capture now copies vectors.

No other performance regressions visible in this chunk.

After reviewing chunk 4/6, there are no performance regressions introduced. This chunk contains only new test files under `testing/python/transform/`.

### Summary
No findings. All changes are test additions that exercise Z3-backed compiler passes. No production code paths, loops, allocations, or I/O were added in this chunk.

### Notes on test overhead
- **Severity**: low
- **File**: `testing/python/transform/test_simdgroup_matrix_rewrite.py:287-289`
- **Issue**: Test `test_repeated_pass_no_solver_leak` runs the rewrite pass 8x sequentially to check for Z3 solver leaks. Each iteration invokes full TIR lowering. This increases CI time linearly with the loop count, but does not affect runtime code.
- **Fix**: Keep as-is if leak detection is required. To reduce CI load, lower iteration count to 2-3: `for _ in range(3):` is sufficient to catch most state-leak bugs while cutting wall time ~60%.

### Verification of prior fixes
Cannot verify from this chunk alone. The fixes cited in commit a8ec2342 — `ICHECK guard` in `z3_prover.cc:489`, `Bind empty-range memo` in `z3_prover.cc:436-449`, `TILELANG_BUILD_TESTS` flip in `CMakeLists.txt:269`, and `prover-cache double-clear` in `phase.py` — are not present in chunk 4/6. None of the touched files here contain production code.

No new O(n^2) loops, redundant I/O, N+1 queries, blocking async calls, allocations in tight loops, or memory growth patterns were introduced in the visible diff.

No findings in this diff chunk.

**Analysis**

I've reviewed chunk 5 of 6 from commit `a8ec2342`. This chunk contains:

1. `testing/python/transform/test_vectorize_alignment.py` - New test file, no runtime code
2. `testing/python/transform/test_z3_bv_mode.py` - New test file, no runtime code  
3. `tilelang/analysis/int24_overflow_proof.py` - Python-side analysis helper with static fast path and optional Z3 fallback. The Z3 solver is instantiated per-call with a 50ms timeout. No loops or allocations in hot path.
4. `tilelang/contrib/nvcc.py` - Refactor to read `target.attrs["arch"]` with fallback. Single attribute access, no performance impact.
5. `tilelang/engine/phase.py` - Adds per-compile sentinel `_Z3_CLEARED_COMPILE_IDS` to avoid double `clear_prover_cache()` calls. Uses `set[int]` with `add`/`discard`/`in` - all O(1). Marker consumed on read so set size bounded by concurrent compiles, not unbounded growth. Verifies fix for "MED perf" issue of double-clear.
6. `tilelang/language/fp8_op.py` - Adds `_z3_prove_dot4_legal` with static fast path before Z3. Z3 path has 50ms timeout and explicit `s.push()/s.pop()`. Guards with `_Z3_AVAILABLE` and env var disable. No N+1 or loop allocations.
7. `tilelang/layout/fragment.py` - Type annotation change `tvm.ir.container.Array` -> `tvm.ffi.Array`, no runtime impact.
8. `tilelang/transform/__init__.py` - Registers new passes. Passes themselves are gated by `PassConfig` defaults OFF.

**Verification of Wave-1b fixes visible in this chunk**

1. **SetBitVectorMode guard**: Not in this diff chunk. Commit message says `z3_prover.cc:489` has ICHECK. Cannot verify here.
2. **Empty-range memo with UNSAT scope**: Commit message says `z3_prover.cc:436-449`. Not visible in chunk 5.
3. **CMakeLists.txt TILELANG_BUILD_TESTS**: Not in chunk 5.
4. **phase.py double-clear sentinel**: Verified. `tilelang/engine/phase.py:15-30` implements `_Z3_CLEARED_COMPILE_IDS` set with consume-on-read. `LowerAndLegalize:197-203` calls `_mark_z3_cleared_for_compile`. `OptimizeForTarget:253-259` calls `_consume_z3_cleared_for_compile` to skip redundant clear. This resolves the "MED perf" double-clear issue.

**New issues check**

No performance regressions introduced in this chunk:

- **No O(n^2) loops**: All new Python code is straight-line or bounded by constant factors. Test files only.
- **No redundant I/O**: Z3 solver calls are gated behind static fast paths and 50ms timeouts. `tilelang/analysis/int24_overflow_proof.py:144-178` and `tilelang/language/fp8_op.py:316-365` both check constants first.
- **No missing caches**: `_Z3_CLEARED_COMPILE_IDS` in `phase.py:15` adds caching to prevent redundant FFI calls. 
- **No N+1 queries**: Z3 usage is per-call with push/pop. No iteration over collections making repeated solver calls.
- **No blocking calls in async**: No async code in chunk.
- **No allocation in tight loops**: No loops. New sets/dicts are module-level singletons.
- **No memory growth**: `_Z3_CLEARED_COMPILE_IDS` set is consumed on read `phase.py:25-30`, preventing unbounded growth. Key is `id(mod)` which is GC'd with module.
- **No large synchronous payloads**: N/A.

Restricting to code visible in chunk 5: **no findings**.

## Finding 1
- **Severity**: high
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:95`
- **Issue**: Z3 solver instantiated and checked on every buffer eligibility call inside `_z3_simdgroup_eligible`. For functions with many GEMM accumulators, `_collect_fragment_gemm_accum_buffers` + `is_simdgroup_eligible` runs per buffer with no cache. Z3 `Solver().check()` is 10-100ms per call. With N buffers, worst-case O(N) solver invocations in TIR pass, blowing compile time. No memoization of shape/dtype queries even though static check already failed.
- **Fix**: Cache Z3 results keyed by `str(shape[-2:]) + dtype + addr_align`. Reuse a single `z3.Context` and `Solver` instance across calls. Example: add module-level `_Z3_CACHE = {}` and skip solver if key present. Clear cache per PassContext to avoid cross-module leakage.

## Finding 2
- **Severity**: high  
- **File**: `tilelang/transform/metal_simd_lift.py:106`
- **Issue**: `_z3_extent_le_32` creates a new `z3.Solver()` for every reduction loop candidate. `_walk_reductions` calls it for each `tir.For` in the function body. For kernels with many reductions, compile-time scales linearly with loop count. 500ms timeout per solver means a 20-loop function could add 10s to compile time even if most extents are constant.
- **Fix**: Hoist solver creation. For constant extents, skip Z3 entirely before constructing solver. For symbolic extents, batch constraints into one solver or memoize by `extent.tostring()`. Add early return: `if node.extent.as_const_int() is not None: return int(node.extent.as_const_int()) <= 32, "static"`

## Finding 3
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:284`
- **Issue**: `_collect_fragment_gemm_accum_buffers` uses `tir.stmt_functor.post_order_visit` which allocates Python closure and dict for every GEMM Call. For large IR with nested loops, this walks entire body multiple times: once in `_collect_fragment_gemm_accum_vars` line 224, again here. No reuse of prior traversal results. O(N_stmts) twice, plus Python overhead.
- **Fix**: Fuse traversals. Collect both vars and buffer maps in a single `post_order_visit`. Pass `accum` dict into `_collect_fragment_gemm_accum_vars` and populate both structures in one pass to avoid double IR walk.

## Finding 4
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:142`
- **Issue**: `_walk_reductions` runs `tir.stmt_functor.post_order_visit` on entire `body` for every pass invocation, even when `tl.simd_lift_reductions` config is False. Line 466 checks `enabled` only after detection runs in some paths. The detector + Z3 queries run before gating, wasting compile time on all Metal funcs when feature is off.
- **Fix**: Move `if not enabled: return func` to top of `_metal_simd_lift` before `_walk_reductions` call. Detection should be skipped when flag off. Current code only gates rewrite, not Z3 detection.

## Finding 5
- **Severity**: low
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:470`
- **Issue**: When `rewrite_gated=True` and many buffers are ineligible, `rejection_log.append(f"{name}={reason}")` builds a long string with full Z3 query text per buffer. Line 484 then joins all into `tl.simdgroup_matrix_rewrite_rejected` attr. For 100+ buffers this creates multi-KB TIR attribute, increasing IR size and serialization cost in downstream passes.
- **Fix**: Cap rejection log or summarize: store only buffer names, not full Z3 queries. Or limit to first N entries. Queries are already in `TL_LOG_SIMDGROUP` logs for debugging.

## Finding 6
- **Severity**: low
- **File**: `tilelang/transform/metal_simd_lift.py:344`
- **Issue**: `_butterfly_stages` uses `math.ceil(math.log2(extent))` then builds list with while loop. For extent=32, this allocates new list each call. `_build_butterfly` calls it per replaced loop. List is constant for given extent. Repeated allocation in pass.
- **Fix**: Precompute and cache: `_STAGES_CACHE = {8:[4,2,1], 16:[8,4,2,1], 32:[16,8,4,2,1]}`. Lookup instead of computing. Eliminates allocation in hot path.

## Finding 7
- **Severity**: info
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:436`
- **Issue**: Wave-2 fix `Bind empty-range memo with UNSAT scope (z3_prover.cc:436-449)` mentioned in prompt is not visible in this chunk. Cannot verify fix resolves prior HIGH issue #5. Chunk shows only simdgroup eligibility logic, which correctly treats Z3 UNKNOWN as ineligible: line 129 returns `proved = (res == z3.unsat)` and line 184 returns `False, f"static-fail; z3-proved={proved}"`. This preserves conservative behavior for UNSAT/UNKNOWN.
- **Fix**: No action in this chunk. Verification deferred to orchestrator with `z3_prover.cc` diff.

## Finding 8
- **Severity**: info  
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:436`
- **Issue**: Wave-2 fix `ICHECK guard against SetBitVectorMode-during-active-scope` not present in this chunk. No SetBitVectorMode call here, so no regression introduced by this file. Z3 usage is read-only.
- **Fix**: N/A for this chunk.

No blocking async calls, N+1 I/O, or allocation in tight runtime loops found in visible runtime code. Main risks are compile-time regressions from uncached Z3 solver use in TIR passes.