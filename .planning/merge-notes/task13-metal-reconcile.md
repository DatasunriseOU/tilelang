# Task 13 — Metal reconcile (lightweight)

Date: 2026-05-28
Branch: `merge/upstream-codegen-reorg`
HEAD: `70206712`

This pass is **analysis-only**. Per plan, we do NOT rewire Metal ops onto the
new `src/backend/common/op/` shared headers. Our Metal backend carries
simdgroup / fp8 / sync-proof code that is intentionally ahead of upstream;
aggressive delegation is unsafe and out of scope.

## 1. Diff inventory: `src/backend/metal/op/*.cc` vs `src/backend/common/op/*.h`

| Metal file (lines)                       | Common analog (lines)                   | Responsibility                                              | Verdict                                                                 |
|------------------------------------------|------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------------|
| `metal/op/fill.cc` (168)                 | `common/op/fill.h` (83)                  | Lower `tl.fill`. Metal version handles `IsSIMDGroupBuffer` explicitly with constant-extent ICHECK before falling through to SIMT path. | **Metal-specific (simdgroup) → keep standalone.** The simdgroup branch has no upstream analog. |
| `metal/op/finalize_reducer.cc` (126)     | `common/op/finalize_reducer.h` (119)     | Lower `tl.finalize_reducer`. Metal layer encodes Metal-specific workspace strides (`same_simdgroup_metal_fast_path_safe`) and ScalarWorkspaceStride / BatchWorkspaceStride helpers. | **Metal-specific (reduction plan) → keep standalone.** Upstream template `FinalizeReducerLowerer<Impl>` doesn't expose the fast-path-safe stride decision; thin specialization would still need a Metal-only `Impl`. Safe-future-delegation candidate only after the upstream template grows a stride hook. |
| `metal/op/reduce.cc` (114)               | `common/op/reduce.h` (596)               | Lower `tl.reduce` AllReduce. Metal uses Metal-only simdgroup fast-path scalar/batch AllReduce string generation; common header is general-purpose AllReduce machinery. | **Metal-specific (simdgroup fast path) → keep standalone.** Common header is much larger and covers the general path; Metal's narrow specialization is intentional. |
| `metal/op/transpose.cc` (74)             | `common/op/transpose.h` (62)             | Lower `tl.transpose`. Bodies are structurally identical (SIMT loop, fuse, vectorize fallback for CPU/local; else infer + partition + vectorize + unroll). | **Thin specialization → safe-future-delegation candidate.** Out of scope this pass; revisit when we next refactor Metal lowering. |
| `metal/op/gemm.cc` (153)                 | (no upstream common analog)              | Metal SIMD-group warp partition + Metal `metal.simdgroup` intrinsic selection. | **No upstream analog.** Metal-only by design. |
| `metal/op/copy.cc` (227)                 | (no upstream common analog)              | Metal `CheckSIMDGroupCopy` + simdgroup load/store path. | **No upstream analog.** Metal-only by design. |

Metal-specific marker counts (simdgroup / fp8 / threadgroup_barrier / fp16x):

```
src/backend/metal/op/finalize_reducer.cc: 4
src/backend/metal/op/reduce.cc:           3
src/backend/metal/op/transpose.cc:        0
src/backend/metal/op/fill.cc:            10
src/backend/metal/op/copy.cc:            11
```

Only `transpose.cc` has zero Metal-specific markers — it is the single clear
candidate for future delegation to `backend/common/op/transpose.h`. Even there
the body is small enough that the win is marginal and we leave it untouched
this pass to avoid touching Metal during the merge gate.

## 2. Open PR #2252 — Metal M5 Cooperative Tensor `T.gemm`

`gh pr view 2252 --repo tile-ai/tilelang --json title,state,files`:

- State: **OPEN**
- Title: `[Metal] M5 Cooperative Tensor T.gemm`
- File overlap with our tree:
  - `src/backend/metal/op/copy.cc`, `fill.cc`, `gemm.cc`, `utils.h` — files we **already have**, but #2252 adds the M5 `mpp::tensor_ops::matmul2d` code path on top.
  - `src/op/builtin.{cc,h}`, `src/target/codegen_metal.{cc,h}`, `src/transform/lower_thread_allreduce.cc`, `src/transform/storage_rewrite.cc`, `src/transform/plan_update_buffer_allocation_location.cc`
  - `tilelang/intrinsics/metal_macro_generator.py`, `tilelang/tileop/gemm/gemm_metal.py`, `tilelang/transform/metal/metal_fragment_to_simdgroup.py`, `tilelang/utils/language.py`, `tilelang/engine/{callback,lower}.py`, `tilelang/cuda/intrinsics/layout/mma_layout.py`, `tilelang/language/builtin.py`, `tilelang/backend/metal/gemm.py`
  - `docs/deeplearning_operators/metal_gemm.md`
  - `testing/python/metal/test_metal_gemm_v2{,_linux}.py`
  - `3rdparty/tvm` (bump — would conflict with our pin `66438efa7`; do not import).

Cooperative-tensor presence in our tree:

```
git grep -l "cooperative_tensor\|matmul2d\|MPSIntrinEmitter\|mpp::tensor_ops" -- src tilelang
=> no matches in src/, no matches in tilelang/.
```

Two filenames in our tree contain the string `cooperative_tensor` /
`matmul2d` / `tensor_ops`: none. The two unrelated hits earlier were
historical `kUseCooperativeLaunch` (#1572) and the
`test_tilelang_language_cooperative.py` register fix (#1593), both
pre-merge and unrelated to MPS M5.

### Verdict: **future port required** (not absorbed; not partial).

PR #2252 is open upstream, not yet merged, and would require:

- The `mpp::tensor_ops::matmul2d` / cooperative-tensor code in
  `codegen_metal.cc` (+430 lines).
- A coordinated TVM bump (the PR touches `3rdparty/tvm`), which our pin
  `66438efa7` cannot accept without re-rebasing our TVM fork.
- Updates to `gemm_metal.py`, `metal_macro_generator.py`, and the M5
  fragment→simdgroup pass.

Recommendation: track #2252 separately and import after it merges
upstream, in a dedicated Metal-M5 follow-up branch where we also rebase
the TVM bump on top of our pin. **Not part of this codegen-reorg merge.**

## 3. Trivial alignments applied

None. Every Metal op file either carries simdgroup / fp8 specialization
that is intentionally ahead of upstream, or (in the case of
`transpose.cc`) is small enough that delegating now risks include-path
churn for a marginal LOC saving. Status quo retained.
