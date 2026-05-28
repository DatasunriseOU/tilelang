# PR #2252 Port Plan — Metal M5 Cooperative Tensor T.gemm

Date: 2026-05-28
Branch: `merge/upstream-codegen-reorg`
HEAD baseline: `a06a8ad2`
TVM pin: `66438efa7e046dcee1e7f8697816b9cb99ce1668` (held)
Hardware: **Apple M4 Max** (not M5). Metal toolchain `metalfe-32023.883`. macOS 26.5.

## Hardware/feature gate

The PR adds Metal 4 cooperative tensor matmul (`mpp::tensor_ops::matmul2d`),
which is a **Metal 4 / M5-only** feature. Our test machine is M4 Max, so the
new path **cannot be exercised at runtime** here. We still port the codegen
and Python dispatch so that an M5 user gets the new path; on M4 we keep the
existing simdgroup path as the default and gate the new path off.

## Per-file map: ALREADY-OURS / RECONCILE / NEW

| Upstream file | Our state | Verdict | Action |
|---|---|---|---|
| `3rdparty/tvm` (bump) | Pinned to `66438efa7…` | RECONCILE | **Do not bump.** The bump only enables Metal 4 shader compile in the vendored TVM metal runtime. We instead route the optional MSL4 compile through a Python callback (`register_default_metal_compile_callback`) that shells to `xcrun metal -std=metal4.0`. Our existing default compile remains unchanged. |
| `docs/deeplearning_operators/metal_gemm.md` | absent | NEW | Add verbatim from PR. |
| `src/backend/metal/op/copy.cc` | ours-ahead (simdgroup copy + dst_strides handling) | RECONCILE | Add `CheckCooperativeTensorCopy` + `LowerCooperativeTensorCopy`. Keep our existing simdgroup logic. |
| `src/backend/metal/op/fill.cc` | ours-ahead | RECONCILE | Add the `IsCooperativeTensorBuffer(op.dst)` early branch in `Fill::Lower`. |
| `src/backend/metal/op/gemm.cc` | ours-ahead (full simdgroup partition impl) | RECONCILE | Add `kMetalCooperativeTensor` const + `CanUseCooperativeTensor` + extend `ComputeMetalWarpPartition` signature (we currently call it `ComputeSIMDGroupWarpPartition` — add gemm_inst overload). |
| `src/backend/metal/op/utils.h` | absent (helpers live in `src/op/utils.h`) | RECONCILE | Add `IsCooperativeTensorBuffer` next to `IsSIMDGroupBuffer` in `src/op/utils.h` (NOT create a new utils.h in metal/op/). |
| `src/op/builtin.cc` | ours has 700+ lines with our own builtins; lacks `tma_load_gather4`/`tma_store_scatter4` (PR adds coop_tensor builtins next to those) | RECONCILE | Append the 4 new `cooperative_tensor_*` TIR_DEFINE_TL_BUILTIN entries at the end. |
| `src/op/builtin.h` | similar — ours lacks gather4/scatter4 | RECONCILE | Add the 4 new `TVM_DLL const Op&` declarations. |
| `src/target/codegen_metal.cc` | ours-ahead (fp8, atomic, fallback helpers; 2157 lines) | RECONCILE | Add: `<MetalPerformancePrimitives/...>` include is conditionally emitted only when a `metal.cooperative_tensor` alloc is seen (avoid forcing MSL4 on M4 by default); the AllocBuffer branch for `metal.cooperative_tensor`; `EnsureCooperativeTensorBuffer` helper; `GetAddrSpaceOf` helper; the four `tl::cooperative_tensor_*` CallNode emitters. Skip upstream additions that conflict with our shape (the `__base_row/__base_col` lane decode emitted in `AddFunction`, the `__gridDim` parameter, and the per-kernel `#pragma clang loop unroll(disable)` heuristic) UNLESS they are needed by the coop tensor emit. The lane vars ARE needed by the coop-tensor load/store emitters, so emit them lazily (only when first coop_tensor call is seen). |
| `src/target/codegen_metal.h` | ours-ahead | RECONCILE | Add `GetAddrSpaceOf`/`EnsureCooperativeTensorBuffer` declarations and `cooperative_tensor_dtype_`/`ct_c_inlined_`/`emitted_frag_lane_vars_` fields. |
| `src/transform/lower_thread_allreduce.cc` | unchanged from upstream | RECONCILE | Add the 1-line `metal.cooperative_tensor` skip in `AllocateCollector::VisitStmt_` + the early-return in the pass. |
| `src/transform/plan_update_buffer_allocation_location.cc` | unchanged | RECONCILE | Add `CooperativeTensorScopeDetector` + early-return. |
| `src/transform/storage_rewrite.cc` | unchanged | RECONCILE | Add `MetalCooperativeTensorScopeDetector` + early-return. |
| `tilelang/backend/metal/gemm.py` | ours has `register_gemm_impl("metal.simdgroup", GEMM_INST_METAL, _match_metal, GemmMetal)` | RECONCILE | Register two impls — `GemmMetalSimdGroup` (rename of current `GemmMetal`) and `GemmMetal` (new coop tensor). |
| `tilelang/cuda/intrinsics/layout/mma_layout.py` | ours-ahead | RECONCILE | Append `metal_ct_store_32x16_to_16x32_layout` + `metal_ct_store_index_map`. |
| `tilelang/engine/callback.py` | ours-ahead | RECONCILE | Append `_compile_metal4` + `register_default_metal_compile_callback`. DO NOT auto-register on import. |
| `tilelang/engine/lower.py` | ours-ahead | RECONCILE | We do NOT auto-call `register_default_metal_compile_callback` on metal lowering, because doing so would force `-std=metal4.0` on M4 hardware where MSL4 is not available. Instead, expose it; user must call explicitly. (Optionally autodetect M5 via `sysctl`, but skip for v1.) |
| `tilelang/intrinsics/metal_macro_generator.py` | ours-ahead | RECONCILE | Add `OPERAND_*` constants, `use_cooperative_tensor`/`a_stride_override`/`b_stride_override`/`inner_k_steps` kwargs, the dual emit-path in `ldmatrix_a`/`ldmatrix_b`/`mma`/`simdgroup_copy`, and `make_cooperative_tensor_store_layout`. |
| `tilelang/language/builtin.py` | ours-ahead | RECONCILE | Append 4 cooperative_tensor_* python builtins. |
| `tilelang/tileop/gemm/gemm_metal.py` | ours has `GemmMetal` as the simdgroup impl | RECONCILE | Rename current → `GemmMetalSimdGroup`. Add new `GemmMetal` (cooperative tensor). Keep our defensive shape checks. |
| `tilelang/transform/metal/__init__.py` | ours has only `MarkHostMetalContext` (no `metal_fragment_to_simdgroup` here — ours lives in `tilelang/transform/`) | RECONCILE | Leave the upstream symlink-style re-export ALONE — we already import `metal_fragment_to_simdgroup` from `tilelang/transform/`. Skip the `MetalFragmentToCT` alias since our tree structure differs. |
| `tilelang/transform/metal/metal_fragment_to_simdgroup.py` | ours lives at `tilelang/transform/metal_fragment_to_simdgroup.py` (different path), 595 lines (ours-ahead) | RECONCILE | The upstream rewrite re-orgs this file. Our copy is much larger and Metal-fragment specific. Skip the upstream changes; the alias `MetalFragmentToCooperativeTensor = MetalFragmentToSimdgroup` is a no-op rename. We will not introduce the alias since downstream Python wiring in this PR already references the simdgroup pass name. |
| `tilelang/utils/language.py` | ours-ahead | RECONCILE | Append `is_metal_cooperative_tensor`. |
| `testing/python/metal/test_metal_gemm_v2.py` | ours-existing | RECONCILE | Add `test_gemm_v2_cooperative_tensor_non_square`, guarded by an M5-feature skip so it doesn't crash on M4. |
| `testing/python/metal/test_metal_gemm_v2_linux.py` | ours-existing | RECONCILE | Add `matmul_gemm_v2_shared_c`, `assert_metal_gemm_v2_cooperative_tensor_codegen`, and `test_metal_gemm_v2_cooperative_tensor_codegen` — this is codegen-only so it does NOT require M5 hardware. |

## Out of scope for this port

- **TVM submodule bump** (`3rdparty/tvm`). The upstream bump enables Metal 4
  shader compilation inside TVM's vendored metal runtime. We carry our own
  TVM fork pinned at `66438efa7e...` (apache+tirx-based) and we explicitly do
  NOT want to push the M5/MSL4-coupled bump there from this agent. If a user
  wants the MSL4 compile path, they call
  `register_default_metal_compile_callback()` which shells out to
  `xcrun metal -std=metal4.0` — independent of the TVM-level runtime. No
  patch needed for TVM (`/tmp/pr2252_tvm.patch` will not be created).

- **The upstream `__base_row/__base_col` fragment-lane decoding emitted
  unconditionally in `AddFunction`.** This is required only when coop-tensor
  load/store calls are emitted. We emit it lazily, gated on
  `emitted_frag_lane_vars_`, so M4 kernels that never use coop_tensor stay
  byte-identical to the pre-PR output.

- **The upstream `__gridDim [[threadgroups_per_grid]]` kernel parameter
  addition + `threadblock_swizzle_pattern` AttrStmt emitter.** We don't use
  this swizzle pattern today, and adding the kernel parameter would change
  every emitted kernel signature. Defer.

- **The `For` loop `#pragma clang loop unroll(disable)` heuristic** for
  loops with `extent > 4`. This silently changes our entire simdgroup
  codegen path; defer.

- **`MetalFragmentToCT` alias** in `tilelang/transform/metal/__init__.py` —
  our tree's pass is at a different path with a different shape; skip.

- **`BuildTileLangMetalWithoutCompile`** — we already use
  `MetalModuleCreateWithFallback` which handles the no-compiler case; the
  `tilelang/engine/lower.py` `device_codegen_without_compile` change is not
  needed.

## Risk register

- **R1**: Including `<MetalPerformancePrimitives/...>` unconditionally would
  force `-std=metal4.0` on every Metal kernel. Mitigation: emit the include
  ONLY when an `metal.cooperative_tensor` alloc is seen.
- **R2**: Adding lane-decode variables (`__lane`, `__qid`, `__base_row`,
  `__base_col`) at function entry changes every kernel's emitted source.
  Mitigation: gate on `emitted_frag_lane_vars_` and emit lazily before the
  first cooperative_tensor call.
- **R3**: New TIR pass skip checks (`metal.cooperative_tensor` scope) are
  pure additions — no risk to non-Metal targets and no risk to Metal
  kernels that don't allocate this scope.
- **R4**: Renaming `GemmMetal` → `GemmMetalSimdGroup` in
  `tilelang/tileop/gemm/gemm_metal.py` breaks any direct imports. Mitigation:
  keep `GemmMetal = GemmMetalSimdGroup` alias and add a new
  `GemmMetalCooperativeTensor` class (rather than reusing the `GemmMetal`
  name for the coop path as upstream does — that's a confusing rename).
  Actually, upstream uses `GemmMetal` for the NEW class. To minimize churn we
  follow upstream: `GemmMetalSimdGroup` = existing; `GemmMetal` = new.
  Then `from tilelang.tileop.gemm.gemm_metal import GemmMetal` callers now
  pick up the new class. Audit imports first.

## Imports audit

`git grep -n "from tilelang.tileop.gemm.gemm_metal import\|tileop.gemm.gemm_metal"` shows only `tilelang/backend/metal/gemm.py` imports `GemmMetal`. Safe to rename.

## Execution order

1. Add helpers in `src/op/utils.h` (IsCooperativeTensorBuffer).
2. Add 4 builtins in `src/op/builtin.{cc,h}`.
3. Add 3 TIR pass skips (lower_thread_allreduce, plan_update_..., storage_rewrite).
4. Extend backend ops: `metal/op/copy.cc`, `metal/op/fill.cc`, `metal/op/gemm.cc`.
5. Extend codegen: `codegen_metal.{cc,h}` (gated coop tensor path).
6. Python: `tilelang/utils/language.py`, `tilelang/cuda/intrinsics/layout/mma_layout.py`, `tilelang/language/builtin.py`, `tilelang/intrinsics/metal_macro_generator.py`, `tilelang/tileop/gemm/gemm_metal.py`, `tilelang/backend/metal/gemm.py`, `tilelang/engine/callback.py`.
7. Tests: append cooperative-tensor codegen-only test to v2_linux; append M5-gated runtime test to v2.
8. Docs: add `metal_gemm.md`.
9. Build + run regression.
