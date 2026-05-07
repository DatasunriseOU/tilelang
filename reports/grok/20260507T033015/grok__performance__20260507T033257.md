---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: dfcb37dcdfb11bcafebb0a3bfe137d02ad86dfd1
timestamp: 2026-05-07T03:32:57.746807+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
**No findings.**

This chunk (1/6) of the `z3-final` diff introduces no performance regressions or hot-path concerns.

### Summary of changes reviewed (restricted to visible code):
- Build/CI plumbing (`CMakeLists.txt`, `conftest.py`): test-only Z3 FFI gating and worktree redirector — no runtime impact.
- Pass config registration (`src/op/builtin.cc/h`): new opt-in flags (`kTMALegalityZ3`, `kDropProvableBoundChecks`, `kPredicateFusion`, `kVectorizeAlignmentProof`) — all default **OFF**.
- TMA legality Z3 path in `src/op/copy.cc`:
  - New helper `Z3ProveStrideAligned16` with 50 ms timeout.
  - Only invoked when `tl.tma_legality_z3=True` **and** cheap analyzer cannot decide.
  - Conservative fallback on any Z3 error/timeout/unknown.
  - No allocation, no loops, no new hot-path work on the default (OFF) configuration.
- Minor compatibility shims for Apache TVM API changes (`codegen_*.cc/h`, `rt_mod_*.cc`): vendored `AllocateNode`/`LetStmtNode` handling, `FunctionInfo` migration — no behavioral change on hot paths.
- New stub passes (`auto_double_buffer.cc`, `drop_provable_bound_checks.cc`):
  - Both **default OFF**.
  - `AutoDoubleBuffer`: pure detection + logging in safe-stub mode; no IR mutation.
  - `DropProvableBoundChecks`: pattern matching + cheap analyzer first, then guarded Z3 (50 ms timeout, BV32 bounds) — only when explicitly enabled. Drops `IfThenElse` only on conclusive proof.

All Z3 usage is:
- Opt-in via `PassContext` flags (default OFF).
- Time-boxed (50 ms).
- Conservative on failure/timeout.
- Limited to symbolic cases where the cheap analyzer already fails.

No O(n²) constructs, no redundant I/O, no N+1 queries, no blocking calls in async contexts, no new allocations in tight loops, no visible memory growth, and no large synchronous payloads introduced in this chunk. The new Z3 paths are narrowly scoped and do not affect default hot-path performance.

Idea-8/9 soundness claims cannot be fully verified from chunk 1 alone (metal-related changes live in later chunks). No **new** regressions introduced here.

**No findings.**

This chunk (2/6) of the z3-final diff introduces no visible performance regressions or hot-path concerns in the provided code. All changes are either:

- **Memoization improvements** in `loop_vectorize.cc` (new `MemoizedIndicesCanVectorize`, tuple-based hash mixers with `TupleHashMix`, cache clear on `Plan` boundary). These address prior collision pathologies and redundant Z3 calls in the vectorization planner. The memo tables are small (per-planner or per-`Z3CanProveLoopAligned`), use cheap structural hashes, and the added work (hashing + lookup) is negligible compared to the avoided Z3 invocations. No allocation in tight loops, no N+1, no redundant I/O.

- **New Z3-based proofs** (`Z3CanProveLoopAligned`, `Z3CanProveUnitStride`, `Z3ProvesInnerWellDefined` etc. in `loop_vectorize.cc` and the new `predicate_fusion.cc`). These are gated behind `PassContext` flags (`tl.vectorize_alignment_proof`, `tl.predicate_fusion` — both default **OFF** per the comments). When disabled, zero runtime cost. When enabled, they use tight 50–200 ms timeouts, conservative fallbacks on timeout/unknown/exception, and small per-query memoization or RAII scopes. No unbounded work, no hot-path blocking calls.

- **Intra-warp elision logic** in `thread_storage_sync.cc` (`ProveIntraWarpRAW`, `is_metal_` flag). Again gated by target==metal (rare for many workloads) and only consulted on shared-memory conflict detection. The Z3 query is bounded (200 ms timeout) and conservative (keep barrier on any doubt). The added diagnostic counter and tag logging are cold-path.

- **RAII helper** (`z3_constraint_scope.h`) and supporting utilities (`BVBoundsForDtype`). Pure infrastructure with no runtime impact on non-Z3 paths.

No O(n²) loops, no large synchronous payloads, no obvious memory growth vectors, no allocation in hot loops, no missing caches (new memos improve the situation), no N+1 query patterns, and no new blocking calls in async contexts visible in this chunk. The changes appear performance-neutral to positive on the Z3-heavy planning paths when flags are ON, and zero-cost otherwise. The "drop unconstrained z3.Int symbolic paths" note from the commit message is not visible in this specific chunk (likely in the other 5 chunks, e.g. Metal transforms).

All new Z3 usage follows the established conservative pattern (timeout + exception safety + RAII). Idea-8/9 soundness claims cannot be fully verified from chunk 2 alone, but no regressions are introduced here.