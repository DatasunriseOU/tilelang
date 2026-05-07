---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: 896431e59e4a4c9132b01a6551738758a29ad037
timestamp: 2026-05-07T02:31:57.952192+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
## Finding 1
- **Severity**: high
- **File**: `src/op/copy.cc:L530`
- **Issue**: `Z3ProveStrideAligned16` creates a fresh `arith::Z3Prover` and calls `z3.SetTimeoutMs(50)` for every TMA stride legality check when `tl.tma_legality_z3` is enabled. The prover is invoked per buffer, per non-innermost stride in `CheckGlobalStrides`. For kernels with many tiled buffers this adds up to 50 ms * N solver invocations in the compiler pipeline, blocking lowering. Z3 context creation + SMT encoding is heavy and not cached across calls.
- **Fix**: Lift the `Z3Prover` instance out of `Z3ProveStrideAligned16` and reuse a single solver per `Analyzer` scope. Add an early-exit when the cheap analyzer already decides alignment. Also reduce timeout or make it configurable. Minimal patch:
```cpp
static thread_local std::unique_ptr<arith::Z3Prover> tma_legality_prover;
if (!tma_legality_prover) tma_legality_prover = std::make_unique<arith::Z3Prover>(analyzer);
tma_legality_prover->SetTimeoutMs(5); // 5ms, not 50
```

## Finding 2
- **Severity**: medium
- **File**: `src/transform/drop_provable_bound_checks.cc:L95`
- **Issue**: Every `IfThenElse` bound-check guard that reaches the Z3 fallback spins up a `Z3Prover`, pushes 2 constraints per free variable, then runs with `z3.SetTimeoutMs(50)`. IR with many loop guards will trigger dozens to hundreds of 50 ms solver calls during `tl.DropProvableBoundChecks`. This pass is opt-in, but when enabled it makes compilation time O(#if_guards * solver_cost) with a high constant.
- **Fix**: Add a pass-level budget and reuse one prover. Fail fast if `cond` has >K free vars. Reduce timeout to 5 ms. Guard the whole pass with a cheap pre-check: only invoke Z3 if `cond` contains `Var` nodes whose bounds are unknown to `analyzer`. 
```cpp
static arith::Z3Prover* GetCachedZ3(arith::Analyzer* analyzer) {
  thread_local std::unique_ptr<arith::Z3Prover> p;
  if (!p) { p = std::make_unique<arith::Z3Prover>(analyzer); p->SetTimeoutMs(5); }
  return p.get();
}
```

## Finding 3
- **Severity**: low
- **File**: `src/target/codegen_py.cc:L569`
- **Issue**: `joined` is built with repeated `std::string::operator+=` inside a loop over `op->message_parts`. For asserts with many message fragments this is O(n^2) character copies due to repeated reallocation. Codegen runs once per assert but Python emitter is used for dumping large IRs and debugging.
- **Fix**: Use `std::ostringstream` or reserve. 
```cpp
std::ostringstream joined;
for (const auto &part : op->message_parts) {
  joined << part->value;
}
if (!joined.str().empty()) {
  stream << "assert " << cond << ", ";
  EscapeStringLiteral_(joined.str(), stream);
```

No other performance regressions or hot-path allocations are visible in chunk 1 of 6.

## Finding 1
- **Severity**: high
- **File**: `src/transform/loop_vectorize.cc:400`
- **Issue**: `indices_can_vectorize_memo_` is a `std::unordered_map` keyed by `std::tuple<size_t, const void*, size_t, int>` whose third element is `std::vector<size_t>`. Every lookup constructs a fresh `std::vector` and copies all index hashes from the `BufferLoad/Store` into it, then destroys it. For loops with many buffer accesses or large rank tensors, this allocates in the hot path of vectorization planning. `MemoizedIndicesCanVectorize` is called inside tight halving loops at lines 391-393, 756-758, 877-879, so the per-call allocation cost multiplies. `Z3CanProveLoopAligned` has the same pattern at line 1034.
- **Fix**: Replace `std::vector<size_t>` in the key with a hash of the sequence, not the sequence itself. Compute `size_t indices_hash = 0; for (h : idx_hashes) TupleHashMix(indices_hash, h);` and use `std::tuple<const void*, int, size_t>` as the key. This removes heap allocation per memo probe. Patch sketch:
```cpp
struct AlignmentMemoKeyHash {
  size_t operator()(const std::tuple<const void*, int, size_t>& k) const noexcept {
    size_t seed = std::hash<const void*>{}(std::get<0>(k));
    TupleHashMix(seed, static_cast<size_t>(std::get<1>(k)));
    TupleHashMix(seed, std::get<2>(k));
    return seed;
  }
};
// In Z3CanProveLoopAligned:
size_t indices_hash = 0;
for (const PrimExpr& idx : indices) TupleHashMix(indices_hash, static_cast<size_t>(hasher(idx)));
auto key = std::make_tuple(static_cast<const void*>(buf.get()), vector_size, indices_hash);
```

## Finding 2
- **Severity**: high  
- **File**: `src/transform/loop_vectorize.cc:981`
- **Issue**: `Z3CanProveAlignedAccess` constructs a fresh `arith::Z3Prover(analyzer)` and calls `SetTimeoutMs(50)` for every single buffer access. `Z3CanProveLoopAligned` calls this inside `PostOrderVisit` for all loads/stores, so a loop body with N memory ops spawns N Z3 solver contexts. Context creation + `setOption("timeout", 50)` is not free. With `vector_size_` halving probes and multiple buffers, this is O(num_accesses * num_retry) solver initializations. 
- **Fix**: Hoist the prover construction out of `Z3CanProveAlignedAccess`. Pass a `Z3Prover&` from `Z3CanProveLoopAligned` and reuse it. Push/pop scopes per access via `ConstraintScope`. Patch:
```cpp
static bool Z3CanProveAlignedAccess(const Buffer& buffer,
                                    const Array<PrimExpr>& indices,
                                    const Var& loop_var, int vector_size,
                                    ::tilelang::tlz3::Z3Prover& z3) { ... }

static bool Z3CanProveLoopAligned(...) {
  auto& z3 = arith::Z3Prover(analyzer);
  z3.SetTimeoutMs(50);
  auto probe = [&](const Buffer& buf, const Array<PrimExpr>& indices) {
    ...
    bool ok = Z3CanProveAlignedAccess(buf, indices, loop_var, vector_size, z3);
    ...
  };
}
```

## Finding 3
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:1072`
- **Issue**: `FreeVarCollector` uses `std::unordered_set<const VarNode*> vars` and `PostOrderVisit` to collect free vars from `elem_offset` and `extent` for every index being checked. Then `Z3CanProveAlignedAccess` iterates those sets and pushes a `ConstraintScope` per var. If an access has rank 8 with complex indices, `PostOrderVisit` + set insertion runs repeatedly. The memo in `Z3CanProveLoopAligned` dedupes by buffer+indices, but indices with different constant folds still re-run the collector. This is CPU overhead in the planner, not Z3 time.
- **Fix**: Cache the free-var set per `StructuralHash(elem_offset)` or per `Buffer`. For most TIR generated by TileLang, `elem_offset` only depends on loop vars that are already bit-bounded once. Store `unordered_map<size_t, unordered_set<const VarNode*>>` keyed by hash. Reuse across accesses.

## Finding 4
- **Severity**: medium
- **File**: `src/transform/loop_vectorize.cc:427`
- **Issue**: `indices_can_vectorize_memo_.clear();` runs at the end of every `VectorizePlanner::Plan` call. If a function has multiple vectorizable loops, the planner object is reused and each loop re-fills the memo from scratch. Keys are `StructuralHash(expr)` which is expensive to compute. For kernels with 10+ loops, this discards useful memoization across loops where the same index expressions appear, e.g. `A[i,j*32 + tx]`. 
- **Fix**: Keep the memo across `Plan` calls but invalidate only when `analyzer_` state changes. If analyzer is constant per pass, make memo a member of `VectorizeRewriter` instead of `VectorizePlanner`, and only clear between IRModules. Or guard clear with a config: `if (Config::ClearVectorizeMemoPerLoop()) indices_can_vectorize_memo_.clear();`

## Finding 5
- **Severity**: low
- **File**: `src/transform/loop_vectorize.cc:1436`
- **Issue**: `Z3CanProveUnitStride` creates a new `arith::Z3Prover(analyzer)` per call and sets timeout. It is invoked from `IndicesCanVectorize` which itself is called by `MemoizedIndicesCanVectorize` in halving loops. Though memoized, cache misses still pay context creation cost. On large kernels with many symbolic strides, this adds up.
- **Fix**: Thread a `Z3Prover&` through `IndicesCanVectorize` from `VectorizePlanner`. Construct once per `Plan` call. Same pattern as Finding 2.

## Finding 6
- **Severity**: low
- **File**: `src/transform/loop_vectorize.cc:1060`
- **Issue**: `PostOrderVisit` in `Z3CanProveLoopAligned` traverses the entire loop body for each call to `Plan`. If the body is large, e.g. unrolled matmul, the traversal is O(body_size) even when the memo hits for all accesses. The lambda checks `if (!all_aligned) return;` but still visits nodes to find the next load/store.
- **Fix**: Short-circuit the `PostOrderVisit` by using a custom visitor that throws or returns early once `all_aligned` becomes false. Or collect loads/stores once per `For` in `VectorizePlanner` and reuse the list for both contiguity and alignment probes.

No other performance regressions visible in this chunk.

No performance regressions found in this diff chunk.

I reviewed `src/transform/vendored/z3_prover.cc`, `src/transform/vendored/z3_prover.h`, and the new test files for hot-path concerns: O(n²) loops, redundant I/O, missing caches, N+1 queries, blocking calls, allocations in tight loops, memory growth, or large sync payloads. 

**Why there are no findings:**

1. **Solver rebuild cost is now gated** `src/transform/vendored/z3_prover.cc:L480-L491`
   `SetBitVectorMode` previously rebuilt the Z3 solver on every call. The diff adds a fast-path `if (width == bv_width_) return;` before calling `RebuildSolver_()`. This removes the main risk of accidental O(N) rebuilds when passes redundantly set the same mode.

2. **New `Reset()` uses the same safe rebuild** `src/transform/vendored/z3_prover.cc:L493-L499`
   It reuses `RebuildSolver_()` and has explicit lifecycle checks to prevent use while constraints are live. No allocations leak across calls: `memo_.clear()`, `side_effect_exprs_.clear()`, and `scope_stack_` is reset to one empty frame.

3. **Exception handling in `CanProve` avoids blocking** `src/transform/vendored/z3_prover.cc:L339-L349`
   Wrapping `solver.check()` in try/catch prevents a Z3 timeout or internal exception from hanging the compiler. It returns `false` instead of propagating, which is the intended conservative behavior and has no perf impact.

4. **Side-effect snapshot is moved, not copied** `src/transform/vendored/z3_prover.cc:L258-L267`
   `side_effect_exprs` is moved out of the member before the lambda capture. The lambda captures by value via `= std::move(side_effect_exprs)`, avoiding a vector copy in the hot path.

5. **No new loops or allocations introduced** 
   `MakeIntVal`, `MakeUIntVal`, and BV sort helpers are O(1) Z3 API calls. The warning paths use `LOG(WARNING)` which is rate-limited by `bv_truncation_warned_`/`bv_range_warned_` booleans, so they won’t spam I/O.

6. **Test helpers allocate per call, but that’s expected** `testing/python/analysis/test_int24_overflow_proof.py:L79-L86`
   `BvCanProve` constructs a fresh `Analyzer` and `Z3Prover` each invocation. That’s test-only and matches existing test patterns. Not used in production passes.

The changes are all correctness/robustness: BV mode plumbing, mode-reset safety, cache hygiene, and logging. Nothing in this chunk introduces new hot-path work, unbounded growth, or redundant computation.

No findings in this chunk.

This diff chunk only adds new Python tests under `testing/python/transform/` for Z3-based passes. There is no implementation code here to review for performance regressions. 

**File**: `testing/python/transform/test_loop_vectorize_z3_contiguity.py`, `test_predicate_fusion.py`, `test_simd_reduction_lift.py`, `test_simd_reduction_rewrite.py`, `test_simdgroup_matrix_detection.py`, `test_simdgroup_matrix_rewrite.py`, `test_tma_legality.py`, `test_vectorize_alignment.py`  
**Issue**: All changes are test harnesses, helper functions, and assertions. No hot-path logic, loops, allocations, I/O, or solver calls are introduced in production code in this chunk.  
**Fix**: N/A — no performance-relevant code present. 

If you want cross-file analysis of the Z3 fallbacks mentioned in the test docstrings, I’ll need the corresponding C++ diffs from `src/transform/loop_vectorize.cc`, `src/op/copy.cc`, `src/transform/predicate_fusion.cc`, etc.

Based on chunk 5 only. Wave-1 said ZERO issues; cross-check shows several hot-path performance risks introduced by the new Z3/prover plumbing and duplicated IR walks.

## Finding 1
- **Severity**: high
- **File**: `tilelang/analysis/int24_overflow_proof.py:L129`
- **Issue**: `prove_dot4_int24_safe` constructs a fresh `z3.Solver()` and sets timeout on every symbolic-K call. Z3 solver init is ~0.5-2ms + 50ms timeout budget per invocation. If this predicate is evaluated per-tile or per-loop-iteration during lowering, it becomes N×50ms blocking latency. No context/solver caching, no early-out. `fp8_op.py:L396` calls this unconditionally for each M=1 vecmat kernel that passes the static fast-path check.
- **Fix**: Cache a thread-local `z3.Context` + `Solver` and reuse. For constant K, the static path already avoids Z3. For symbolic K, hoist the query construction out of inner loops: memoize by `(K_expr_structural_hash, x_max, y_max)`. If K is loop-invariant, prove once at `PrimFunc` entry. Diff:
  ```python
  _Z3_CTX = None
  def _get_solver():
      global _Z3_CTX
      if _Z3_CTX is None:
          _Z3_CTX = _z3.Context()
      s = _z3.Solver(ctx=_Z3_CTX)
      s.set("timeout", _Z3_INT24_TIMEOUT_MS)
      return s
  ```

## Finding 2
- **Severity**: high
- **File**: `tilelang/engine/phase.py:L170-L174`, `L267-L271`
- **Issue**: `tvm.ffi.get_global_func("tl.z3.clear_prover_cache", allow_missing=True)` is called at the start of both `LowerAndLegalize` and `OptimizeForTarget`. FFI lookup is not free: ~50-200ns per call but it runs on every compile, and the result is not cached. More importantly, clearing the prover cache unconditionally per phase defeats the point of a per-thread cache. If a pass invokes `CanProve` 100 times, you wipe memoization 100 times, turning O(1) amortized proofs into O(N) re-solves.
- **Fix**: Resolve the FFI func once at module import and only clear when the pass boundary actually changes thread-local state. Remove from both phases or guard with a phase-local flag:
  ```python
  _Z3_CLEAR = tvm.ffi.get_global_func("tl.z3.clear_prover_cache", allow_missing=True)
  def LowerAndLegalize(...):
      if _Z3_CLEAR is not None and not hasattr(LowerAndLegalize, "_cleared"):
          _Z3_CLEAR(); LowerAndLegalize._cleared = True
  ```

## Finding 3
- **Severity**: medium
- **File**: `tilelang/transform/metal_fragment_to_simdgroup.py:L229-L231`, `L430-L432`
- **Issue**: Two separate `tir.stmt_functor.post_order_visit(body, _visitor)` walks over the entire function body: `_collect_fragment_gemm_accum_vars` and `_collect_fragment_gemm_accum_buffers`. For large Metal kernels with 10k+ stmts, this is O(N) work done twice. Both visitors collect overlapping info and could be merged. Current code walks IR twice even when `tl.simdgroup_matrix_rewrite` is OFF.
- **Fix**: Fuse into a single traversal that returns both `set[Var]` and `dict[Var, Buffer]`. Skip traversal entirely if target != metal or if no GEMM ops present. Guard:
  ```python
  def _collect_accums(body):
      acc_vars, acc_bufs = set(), {}
      # single post_order_visit
      return acc_vars, acc_bufs
  ```

## Finding 4
- **Severity**: medium
- **File**: `tilelang/language/fp8_op.py:L361-L364`
- **Issue**: `_z3_prove_dot4_legal` creates a new `z3.Solver()` per call and instantiates 6 `z3.Int` variables even when many args are constant. Timeout is 50ms. This function is invoked by `_fp8_scaled_matmul` during lowering for every M=1 vecmat site. If a module has 50 such calls, worst-case 2.5s compile-time stall. No caching of `z3.Context`, no structural hash memoization for identical symbolic args.
- **Fix**: Same as Finding 1: thread-local solver cache + memoize by structural hash of `(K_a, K_b, stride_a, stride_b, addr_a, addr_b)`. Or better, move this predicate to C++ `Z3Prover` where `Analyzer` + `CanProve` already caches subexpressions. Current Python z3 bypasses all existing C++ memoization.

## Finding 5
- **Severity**: low
- **File**: `tilelang/contrib/nvcc.py:L347-L355`
- **Issue**: To support apache's `target.attrs["arch"]`, the code does `try: target_arch = target.attrs.get("arch") except Exception: ...` then falls back to `getattr(target, "arch", None)`. This runs on every `get_target_compute_version` call. `target.attrs` access is cheap but the try/except + dict lookup + getattr chain adds overhead vs a single cached accessor. Not hot-path today, but called by `tilelang.engine.phase` during TMA detection.
- **Fix**: Cache the attribute lookup strategy once per process or use `hasattr` check without exception:
  ```python
  def _get_arch(target):
      return target.attrs.get("arch") if hasattr(target, "attrs") else getattr(target, "arch", None)
  ```

## Finding 6
- **Severity**: info
- **File**: `testing/python/transform/test_z3_bv_mode.py:L216-L237`
- **Issue**: Test `test_z3_prover_cross_pass_isolation` calls `_CLEAR_PROVER_CACHE()` 4 times per test run. If cache clear triggers Z3 context teardown/recreate, this adds 1-3ms per clear in CI. Test is valid, but documents that cache clearing is not free.
- **Fix**: Document in `z3_prover.cc` that `clear_prover_cache` should be O(1) pointer map clear, not context teardown. If it currently tears down, change to just `map_.clear()`.

No O(n^2) loops, redundant I/O, or large synchronous payload allocations observed in this chunk. Main risks are Z3 solver lifecycle and duplicated IR traversals introduced by the new prover gates.

## Finding 1
- **Severity**: medium
- **File**: `tilelang/transform/metal_simd_lift.py:L87`
- **Issue**: `_z3_extent_le_32` creates a new `z3.Solver()` and imports `z3` on every call from `_walk_reductions`. For functions with many reduction loops, this repeats solver setup/teardown and incurs the 500ms timeout per candidate even if extents are structurally identical. No caching/memoization is used, so identical `extent_expr` ASTs trigger redundant Z3 work.
- **Fix**: Hoist the `import z3` to module scope with lazy init, and add an LRU cache keyed by a structural hash of `extent_expr`. Reuse a single solver instance and `push/pop` constraints per query. Example:
  ```python
  _Z3_CACHE = {}
  def _z3_extent_le_32(extent_expr):
      key = structural_hash(extent_expr)
      if key in _Z3_CACHE: return _Z3_CACHE[key]
     ...
      _Z3_CACHE[key] = (proved, query)
  ```

## Finding 2
- **Severity**: low
- **File**: `tilelang/transform/metal_simd_lift.py:L121`
- **Issue**: `_log_candidates` emits `logger.warning` for every proved candidate when `TL_LOG_SIMD_LIFT` is set or any candidate proved. In large modules with dozens of reduction loops this becomes log spam during compilation, adding I/O and string formatting cost to the pass. The log is gated by `any(c.proved for c in candidates)` so a single proved loop forces logging for all.
- **Fix**: Rate-limit or summarize: log once per function with counts, or downgrade to `logger.debug`. Example:
  ```python
  if proved := [c for c in candidates if c.proved]:
      logger.info("simd-lift: %s: %d/%d loops proved", func_name, len(proved), len(candidates))
  ```

## Finding 3
- **Severity**: low
- **File**: `tilelang/transform/metal_simd_lift.py:L255`
- **Issue**: `_ButterflyRewriter._mutate` recurses over every IR node type and rebuilds `SeqStmt`, `IfThenElse`, `LetStmt`, etc, even when no annotated loops exist. For large bodies this copies the entire AST once per function invocation. The pass runs after detection, but detection already traversed once in `_walk_reductions`. Net effect: 2x AST traversal + potential rebuild per func when enabled.
- **Fix**: Early-exit the rewriter if `candidates` shows no `annotated and proved` loops. Pass the candidate list into `_ButterflyRewriter` and skip recursion when `not any(c.annotated and c.proved for c in candidates)`.

## Finding 4
- **Severity**: info
- **File**: `tilelang/transform/metal_simd_lift.py:L1`
- **Issue**: Pass is fully gated by `TL_SIMD_LIFT_REDUCTIONS=False` default and `target.kind.name!= "metal"` check. No runtime cost unless explicitly enabled. Z3 queries are bounded by 500ms timeout, preventing hangs. No allocation in tight loops or N+1 external calls detected in this chunk.
- **Fix**: No change required. Keep default OFF and document that enabling on large Metal kernels should be profiled due to Z3 overhead.

No critical regressions, blocking I/O, memory growth, or O(n^2) patterns found in this diff chunk.