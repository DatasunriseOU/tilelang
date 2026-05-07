---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..z3-final
base_ref: 1d3b44bad357079be305d33ab48bb3006ddbf56d
head_ref: a8ec234281722049a4762a71e4476fb711345d0b
timestamp: 2026-05-07T02:48:12.734146+00:00
files: ['3rdparty/tvm', 'CMakeLists.txt', 'conftest.py', 'src/op/builtin.cc', 'src/op/builtin.h', 'src/op/copy.cc', 'src/op/reduce.cc', 'src/target/codegen_cuda.cc', 'src/target/codegen_cuda.h', 'src/target/codegen_cutedsl.cc', 'src/target/codegen_cutedsl.h', 'src/target/codegen_metal.cc', 'src/target/codegen_py.cc', 'src/target/codegen_py.h', 'src/target/rt_mod_cuda.cc', 'src/target/rt_mod_cutedsl.cc', 'src/transform/auto_double_buffer.cc', 'src/transform/drop_provable_bound_checks.cc', 'src/transform/loop_vectorize.cc', 'src/transform/predicate_fusion.cc', 'src/transform/thread_storage_sync.cc', 'src/transform/vendored/z3_constraint_scope.h', 'src/transform/vendored/z3_prover.cc', 'src/transform/vendored/z3_prover.h', 'testing/python/analysis/test_int24_overflow_proof.py', 'testing/python/language/test_fp8_dot4_packed_legality.py', 'testing/python/transform/test_auto_double_buffer.py', 'testing/python/transform/test_drop_bound_checks.py', 'testing/python/transform/test_intra_warp_2d_launch.py', 'testing/python/transform/test_loop_vectorize_z3_contiguity.py', 'testing/python/transform/test_predicate_fusion.py', 'testing/python/transform/test_simd_reduction_lift.py', 'testing/python/transform/test_simd_reduction_rewrite.py', 'testing/python/transform/test_simdgroup_matrix_detection.py', 'testing/python/transform/test_simdgroup_matrix_rewrite.py', 'testing/python/transform/test_tma_legality.py', 'testing/python/transform/test_vectorize_alignment.py', 'testing/python/transform/test_z3_bv_mode.py', 'tilelang/analysis/int24_overflow_proof.py', 'tilelang/contrib/nvcc.py', 'tilelang/engine/phase.py', 'tilelang/language/fp8_op.py', 'tilelang/layout/fragment.py', 'tilelang/transform/__init__.py', 'tilelang/transform/metal_fragment_to_simdgroup.py', 'tilelang/transform/metal_simd_lift.py', 'tilelang/transform/pass_config.py']
---
**No new performance regressions or hot-path concerns introduced in this visible chunk (chunk 2 of 6).**

Wave-1 HIGH issues (5 of 6 previously flagged) appear resolved based on the changes shown:

- **Memoization fixes (B6/B8/idea712)**: The old FNV-xor hash mixer (collision pathology when equal values cancel under XOR) is replaced by `std::tuple` keys with a custom `TupleHashMix` (boost::hash_combine-style rotate + `+=` to avoid cancellation) for both `indices_can_vectorize_memo_` (`VectorizeMemoKeyHash`) and the alignment memo (`AlignmentMemoKeyHash`). See `loop_vectorize.cc:32-88` (new namespace helpers), `:436` (tuple construction with `StructuralHash`), `:449` (memo lookup/emplace), and similar in the new `Z3CanProveLoopAligned` probe at `loop_vectorize.cc:~1050-1070`. Cache lifetime is now explicitly scoped with `clear()` at `loop_vectorize.cc:355` (conservative invalidation after top-level `Plan`). This eliminates the prior aliasing/false-negative risk without introducing quadratic behavior or heavy per-lookup overhead.<grok:render card_id="738910" card_type="citation_card" type="render_inline_citation"><argument name="citation_id">5</argument></grok:render>

- **RAII for Z3 scopes (fix-B2)**: New `z3_constraint_scope.h` (full file, lines 1-171) provides `ConstraintScope` (move-only RAII wrapper around `EnterConstraint` recover lambda) and `BVBoundsForDtype` (dtype-aware signed/unsigned ranges, fixing prior unsound flat `[0, 2^31)`). Used at `loop_vectorize.cc:602` (alignment free-var bounds) and referenced in `predicate_fusion.cc`. Prevents scope leaks on early return/exception.

- **Predicate fusion safety (fix-B3)**: New `predicate_fusion.cc` (full file) adds `Z3ProvesConditionLoadsWellDefined` and reuses `Z3ProvesIndexInRange` for inner-condition loads. Conservative (bail on timeout/unknown/malformed). No hot-path impact (disabled by default via `kPredicateFusion`).

- **Thread sync Metal elision (idea #11)**: `thread_storage_sync.cc:1556` (planner ctor), `~1625-1735` (`ProveIntraWarpRAW` with tag-based axis lookup, strict canonical `thread_tag` matching, range bounds via `EnterConstraint`, 200ms timeout), and `~2251` (elision check + counter). Conservative-by-default (keep barrier on any uncertainty). Bounded by per-conflict Z3 queries only on Metal shared RAW/WAR.

- **Other guards**: `ICHECK` in unseen chunk (z3_prover.cc:489); empty-range memo bind (z3_prover.cc:436-449); tests and phase.py sentinel gating (not in this chunk).

**New issues in visible code: none qualifying as regressions/hot-path concerns.**

- Memo lookups use `std::unordered_map` + custom hash/tuple (good asymptotic; tuple comparison is linear in arity but arity is tiny: 4 or 3+vector). `StructuralHash` on exprs is already present in original code; no new allocation in inner loops visible. Cache clears are infrequent (per `Plan`).

- Z3 calls remain timeout-bounded (50ms or 200ms) and conservative (false on exception/unknown/timeout). No blocking in async paths; no N+1 visible; no large synchronous payloads. Alignment/stride proofs are best-effort and off-by-default or Metal-only.

- New `PostOrderVisit` walks and collectors (`BufferLoadCollector` etc.) are linear in body size; no quadratic nesting shown.

- `TupleHashMix` is a fast inline rotate/add (no regression vs prior FNV).

- CMake test default flip and phase.py sentinel are non-hot-path.

All visible Z3 usage follows the hardened pattern (RAII scopes, dtype bounds, affine guards via `IsAffineInVar` at `loop_vectorize.cc:~1200`, degenerate-case short-circuits). No memory-growth or redundant I/O patterns introduced in this chunk. Prior wave-1 concerns do not reappear.

**No new performance regressions or hot-path concerns found in this chunk.**

This chunk (chunk 4 of 6) consists entirely of **new test files** under `testing/python/transform/`. It adds regression and correctness tests for several Z3-related features (intra-warp barrier elision, loop vectorize contiguity, predicate fusion, simd reduction lift/rewrite, simdgroup matrix detection/rewrite, and TMA legality).

### Key observations (restricted to visible code):
- No changes to production C++/Python implementation paths in this chunk.
- No modifications to `src/transform/vendored/z3_prover.cc`, `z3_prover.h`, `loop_vectorize.cc`, `predicate_fusion.cc`, `thread_storage_sync.cc`, `metal_fragment_to_simdgroup.py`, `metal_simd_lift.py`, `copy.cc`, or any hot-path lowering/codegen.
- All added code is under `testing/python/transform/` and exercises:
  - New test helpers (`_run_thread_sync_metal`, `_count_storage_sync`, `_stringified_ir`, `_run_pass`, etc.).
  - Conservative-by-default Z3 fallback paths with timeout guards (e.g., 200ms in intra-warp, small budgets elsewhere).
  - Structural assertions and "no crash" contracts on UNKNOWN/timeout/exception cases.
- Tests are gated (`@tilelang.testing.requires_metal`, `@_z3_skip`, `PassContext` configs default-OFF where appropriate).
- No O(n²) loops, no redundant I/O, no N+1 patterns, no allocations in tight loops, no blocking calls in async contexts, no large synchronous payloads visible.
- CMake change and phase.py double-clear sentinel are **not visible** in this chunk (per instructions: restrict to visible code).
- The ICHECK guard and empty-range memo binding in `z3_prover.cc` are **not visible** here.

### Prior issues (Wave-1b HIGH findings)
The prompt states Wave-1b found 6 HIGH issues (5 fixed). This chunk contains **only tests** that appear designed to cover exactly those categories (unusual thread tags, degenerate 3D launches, Z3 timeout fallback, indirect indexing, nested predicates with buffer loads, repeated-pass solver scope leaks, signed int32 bounds, etc.). No implementation code is present, so no new regressions can be introduced here. The tests themselves look clean and focused on pinning conservative behavior.

**Verdict for this chunk:** Clean. No findings.

(The orchestrator should merge with results from other chunks for the full picture, especially any remaining `z3_prover.cc:436-449`, `:489`, phase.py, and CMake changes.)