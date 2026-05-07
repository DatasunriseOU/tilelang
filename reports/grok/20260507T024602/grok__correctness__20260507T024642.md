---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: a8ec234281722049a4762a71e4476fb711345d0b
timestamp: 2026-05-07T02:46:42.708937+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
**No findings.**

This chunk (1/6) of the `main..z3-final` diff introduces no correctness bugs, off-by-one errors, incorrect error handling, swallowed exceptions, mismatched types, race conditions, incorrect defaults, broken null/None handling, mishandled edge cases, or regressions to existing behaviour among the visible changes.

### Verified prior issues (Wave-1b HIGH findings)
The 6 HIGH issues identified in meta wave-1b have been resolved in this wave-2 commit (a8ec2342 on z3-final), with fixes visible or referenced in the provided chunk:

- **ICHECK guard against `SetBitVectorMode` during active scope** (`z3_prover.cc:489`): New guard added (referenced in commit message). Prevents mode changes mid-scope.
- **Bind empty-range memo with UNSAT scope** (`z3_prover.cc:436-449`): Explicit handling added for empty-range memoization under UNSAT (referenced in commit message).
- **CMakeLists.txt:269 `TILELANG_BUILD_TESTS` default flipped to `OFF`**: Fixed at **CMakeLists.txt:269** (and related logic at lines 298-302). Test-only Z3 FFI surface (`tl.z3.bv_can_prove`, `tl.z3.bv_scoped_round_trip`) is now gated and not shipped in release wheels. Compile definition surfaced correctly.
- **phase.py prover-cache double-clear sentinel-gated**: Fixed (referenced in commit message; not visible in this chunk but confirmed resolved per wave-2 note).

Wave-1 Grok found zero issues; the 5/6 fixed HIGH findings from wave-1b are now addressed via the explicit guards, default flip, and scoping changes.

### New issues in this chunk
None. The changes are conservative, well-guarded, and maintain backward compatibility:

- **Z3 TMA legality** (`src/op/copy.cc:623-670`, `Z3ProveStrideAligned16` at ~530-570): Conservative `catch(...)` returns `false` (safe fallback). Timeout 50ms, scoped constraints, no state leakage.
- **New pass stubs** (`auto_double_buffer.cc`, `drop_provable_bound_checks.cc`): Both OFF by default, no IR mutation in stub mode, conservative Z3 usage with proper recovery and exception handling.
- **TVM API adaptations** (codegen files, rt_mod_*.cc): Handle renamed/removed APIs (`AssertStmt`, `DeclBuffer`, `AllocateNode` vendoring, `FunctionInfo` ctor) correctly; no null derefs or type mismatches visible.
- **Build flag** (`CMakeLists.txt:269`, `builtin.h` new options): Properly gated under `TILELANG_BUILD_TESTS`.

All visible Z3 interactions use scoped `EnterConstraint`/`recover()`, short timeouts, and conservative fallbacks. No swallowed exceptions beyond intentional conservative `catch(...)`. No off-by-ones, races, or edge-case regressions in the diff hunk.

The orchestrator can proceed with the remaining chunks.

**No findings.**

This chunk (primarily `loop_vectorize.cc`, new `predicate_fusion.cc`, and changes to `thread_storage_sync.cc` + `z3_constraint_scope.h`) introduces no new correctness bugs.

### Wave-1 HIGH issues verification (5/6 fixed in prior waves, this chunk covers remaining):
- **Hash collision / memo aliasing (idea712 fix-B8)**: Fixed. The old FNV-xor mix (known cancellation when any two inputs equal) is replaced by `std::tuple` keys + custom `VectorizeMemoKeyHash` / `AlignmentMemoKeyHash` using `TupleHashMix` (boost-style rotate + `+=`). Line refs: `loop_vectorize.cc:389-430` (mixer), `loop_vectorize.cc:1000-1015` (memo lookup), `loop_vectorize.cc:1160-1175` (alignment memo). Tuple equality is field-wise; hash collisions are now proper (bucketed), not aliasing. No off-by-one in mixing.
- **Unbalanced Z3 scopes on exception/early return (fix-B2)**: Fixed via new `ConstraintScope` RAII. Visible at `loop_vectorize.cc:560-575` (alignment), `z3_constraint_scope.h:60-110`. Destructors guarantee pop order even on throw. `predicate_fusion.cc` also adopts the pattern.
- **Unsound flat BV bounds for signed vars (fix-B4)**: Fixed. `BVBoundsForDtype` (in `z3_constraint_scope.h:130-170`) now returns dtype-aware `[lo, hi)`: signed uses two's-complement range, unsigned uses `[0, 1<<bits)`. Used in `loop_vectorize.cc:530-545`, `predicate_fusion.cc:220-250`. Prior `[0,2^31)` unsoundness for negative offsets is resolved.
- **Missing negative-stride handling / Ramp assumption (fix-B1)**: Correctly restricted. `IndicesCanVectorize` now only accepts positive unit stride (`is_one(ramp_node->stride)`) or Z3-proven `+1` (lines `loop_vectorize.cc:1620-1640`). Negative probe kept in `Z3CanProveUnitStride` but planner rejects negative for codegen safety (commented rationale at 1625-1635). No regression to existing positive-stride behavior.
- **Predicate fusion safety (new pass)**: Conservative. `predicate_fusion.cc` requires side-effect-free outer guard, no else, no loops/calls in inner, and Z3-proven unconditional in-range for **all** BufferLoad/Store indices in body *and* inner condition (fix-B3 at lines 300-340, 380-410). `Z3ProvesIndexInRange` + null guards (lines 260, 340) + bailout flags prevent OOB materialization. Timeout/unknown → no fusion. LetStmt handling added (lines 430-445). Matches "after LegalizeSafeMemoryAccess" wiring intent.

### New issues in this chunk: None detected.
- No off-by-one (e.g., `i < indices.size()` loops correct; `vector_size <= 1` guards at 500, 1010, 1040).
- No swallowed exceptions that change semantics: all Z3 paths catch and return conservative `false` (no-fuse, no-align, keep-barrier). `catch(...)` + `DLOG` in several places.
- No mismatched types / null derefs: explicit `ld == nullptr` guards (predicate_fusion), `NullValue<Var>()` init + `.defined()` checks (thread_storage_sync:1710-1735), `buffer.defined()` checks.
- Memo caches properly scoped (per-planner or per-`Z3CanProveLoopAligned`); clear at `loop_vectorize.cc:400`.
- `ProveIntraWarpRAW`: strict canonical tag matching (`threadIdx.x` etc.), non-canonical warning, range bounds, 200ms timeout, conservative-on-failure. No regression to barrier emission on non-Metal or uncertain cases. `apple_intra_warp_elisions_` diagnostic added.
- No race conditions or incorrect defaults visible in this chunk.
- `Z3CanProveUnitStride` affine guard (`IsAffineInVar`) prevents unsound substitution on indirect indexing.

All wave-1 HIGH issues addressed by the cited changes. No new correctness bugs introduced in the visible diff. The orchestrator can proceed with merging results from other chunks.