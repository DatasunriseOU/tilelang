---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: dfcb37dcdfb11bcafebb0a3bfe137d02ad86dfd1
timestamp: 2026-05-07T03:31:05.960876+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
**No findings.**

This chunk (chunk 2/6) of the z3-final diff introduces no new correctness bugs, off-by-one errors, swallowed exceptions with behavioral change, mismatched types, race conditions, incorrect defaults, broken null/None handling, or regressions to existing behavior in the visible code.

Key changes reviewed (all appear sound given the stated intent and cross-references to prior fixes):

- **loop_vectorize.cc**: 
  - Replacement of collision-prone FNV-xor memoization with `std::tuple` + custom `TupleHashMix` (with rotation/add instead of xor) for `MemoizedIndicesCanVectorize` and `AlignmentMemoKeyHash`. The new hash avoids documented cancellation pathology while preserving per-planner-instance lifetime. Cache clear in `Plan` is conservative and prevents stale cross-`Plan` reuse.
  - New Z3-based alignment proof (`Z3CanProveAlignedAccess`/`Z3CanProveLoopAligned`) and unit-stride fallback (`Z3CanProveUnitStride` + `IsAffineInVar` guard). All paths are conservative on timeout/unknown/exception (return false). New `vector_size <= 1` and extent `<= 1` short-circuits are correct. Negative-stride probe is present but positive-stride-only acceptance in `IndicesCanVectorize` prevents codegen mismatch (explicitly documented). RAII `ConstraintScope` usage (fix-B2) and dtype-aware bounds (fix-B4) improve exception safety and soundness.
  - `ProveIntraWarpRAW` in the later `thread_storage_sync.cc` portion is not in scope for this chunk.

- **predicate_fusion.cc** (new file):
  - Conservative predicate fusion guarded by Z3 well-definedness proofs for inner condition + body loads/stores. Bailout on side effects, loops, calls, malformed loads, too many free vars, or any solver failure. `fix-B3` for condition loads and null guards (`ld == nullptr`) are defensive. LetStmt handling mirrors other TileLang passes. Pass is off-by-default.

- **thread_storage_sync.cc**:
  - Minor constructor extension for `is_metal_` flag and diagnostic counter. No behavioral change to existing non-Metal path. `ProveIntraWarpRAW` and tag-based axis mapping (fix for 2-D launches) appear in this chunk but the core logic is gated behind `is_metal_` and remains conservative (keep barrier on any uncertainty/non-canonical tags). No regression to prior shared-memory sync behavior.

- **z3_constraint_scope.h** (new):
  - Pure RAII helper + `BVBoundsForDtype`. No executable logic beyond the documented contract; exception safety in destructor is intentional and conservative.

All Z3 interactions use tight timeouts, catch-all handlers returning conservative results (usually `false`), and RAII where introduced. New features (alignment proof, predicate fusion, intra-warp elision) are gated behind PassConfig flags (default OFF) or target kind. Memoization uses structural hashing + tuple keys to avoid prior collision classes. No off-by-one, null dereferences, or type mismatches observed in the visible diff. Existing behavior for non-Z3/non-Metal paths is preserved.

The changes are consistent with the commit message (drop unconstrained `z3.Int` paths elsewhere; verify Idea-8/9 soundness when flags ON). No correctness regressions introduced in this chunk.

**No findings.**

This chunk (4/6) consists exclusively of new test files exercising the Z3-based passes added in the `z3-final` branch (intra-warp barrier elision, loop vectorize contiguity, predicate fusion, simdgroup matrix detection/rewrite, simd reduction lift/rewrite, and TMA legality).

All tests are written conservatively:
- They pin expected conservative-by-default behavior on Z3 `UNKNOWN`/timeout/exception paths.
- They explicitly verify that the new gated paths (`tl.simd_lift_reductions`, `tl.simdgroup_matrix_rewrite`, `tl.tma_legality_z3`, etc.) are **off by default** and preserve prior behavior when disabled.
- They include regression tests for previously identified audit issues (strict thread tag matching, signed int32 bounds, buffer loads in conditions, indirect indexing, etc.).
- No modifications to production lowering logic appear in this chunk; only test scaffolding and helper exercising.

No off-by-one errors, incorrect error handling, swallowed exceptions, type mismatches, race conditions, bad defaults, null/None issues, or behavioral regressions are introduced in the visible code.

The changes are clean from a correctness perspective. The orchestrator can proceed with the remaining chunks.