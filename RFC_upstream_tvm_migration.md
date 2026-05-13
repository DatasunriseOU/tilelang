# RFC: migrate tilelang from TileLang/tvm to apache/tvm + apache/tvm-ffi

## TL;DR

We have a downstream TileLang branch running on current apache/tvm plus
apache/tvm-ffi, while still carrying the TileLang fixes we needed for CUDA,
HIP, Apple Metal, and MLX interop.

This RFC is a request to the TileLang maintainers:

1. Please confirm whether the current TileLang/tvm submodule fork is an
   intentional long-term fork, or just a historical pin.
2. If the fork is not intentionally load-bearing, we would like to upstream
   our migration work and help TileLang move to apache/tvm + apache/tvm-ffi.
3. Independently of that larger decision, we can split out the useful fixes
   we found during the migration into small PRs, so the team can take value
   without accepting the whole stack at once.

Working proof-of-concept repos:

- tilelang fork: https://github.com/DatasunriseOU/tilelang
- tvm fork: https://github.com/DatasunriseOU/tvm
- tvm-ffi fork: https://github.com/DatasunriseOU/tvm-ffi

The short version: the migration is real and buildable in our branch, but the
first upstream PR should be staged carefully. The current TileLang/tvm history
is more nuanced than "a frozen fork": main is frozen, but the TileLang
submodule actually points into tilelang_main, which carries local TVM
changes. This RFC reflects that chronology explicitly.

---

## Verified chronology

I re-checked the repository state on 2026-05-13 after fetching
tile-ai/tilelang, TileLang/tvm, apache/tvm, DatasunriseOU/tvm,
apache/tvm-ffi, and DatasunriseOU/tvm-ffi.

### tile-ai/tilelang -> TileLang/tvm

Upstream tile-ai/tilelang@main still declares:

text
3rdparty/tvm -> https://github.com/TileLang/tvm


The submodule commit pinned by tile-ai/tilelang@main is:

text
0e15b274bce8b46f971abf5ac390e844aa6acee5
2026-04-16 Fix duplicate __weakref__ declaration in derived_object wrappers (#33)


That commit is not on the frozen TileLang/tvm main line. It is on
TileLang/tvm's default branch, tilelang_main.

### TileLang/tvm branch split

TileLang/tvm has two relevant lines:

- TileLang/tvm main
  - commit: 3e8fe753800d55e843c441f302a6a90b26dd22b9
  - date: 2025-08-23
  - subject: [ROCm] Minor fixes for latest refactor (#18225)
  - relation to apache/tvm main: 648 apache commits ahead,
    0 TileLang commits ahead.
  - interpretation: this branch is effectively a frozen apache/tvm pin.

- TileLang/tvm tilelang_main
  - current remote HEAD: 02037fee0c80967dbd78a181c1aac92587fa8f68
  - date: 2026-04-24
  - subject: Enhance Float4E2M1FNUnpacked support across the codebase
  - merge-base with apache/tvm main: 85a877085714b4d10d65e2c267dab3937915e8a1
    from 2025-12-10.
  - relation to apache/tvm main: about 420 apache commits behind and
    178 local TileLang commits ahead.
  - the submodule pin used by tile-ai/tilelang@main is 0e15b274, three
    commits behind current tilelang_main.

So the migration question is not "can we drop a fork with zero local commits?"
The real question is:

> Can the useful tilelang_main deltas be either upstreamed to apache/tvm,
> moved into tilelang, or made unnecessary by current apache/tvm, so TileLang
> no longer needs a permanent TVM fork?

That is the framing this RFC should use.

### DatasunriseOU migration branch

Our current downstream state:

- DatasunriseOU/tilelang
  - current branch includes recent TileLang upstream changes through the
    May 2026 backend split work, including PRs in the #2131 to #2191
    range.
  - it is a migration branch, not a clean PR branch yet.

- DatasunriseOU/tvm
  - current submodule commit: c747da46f3a2b1cebad944a97240002333717926
  - date: 2026-05-13
  - subject: Update tvm-ffi for TileLang copy split
  - relation to fetched apache/tvm main: 17 commits behind,
    48 commits ahead.

- DatasunriseOU/tvm-ffi
  - current nested submodule commit: 9f3f3075036f6ad896a371278e72c95bf6b232e3
  - date: 2026-05-13
  - subject: Fix reflection compatibility for TileLang tirx
  - relation to fetched apache/tvm-ffi main: 7 commits behind,
    11 commits ahead.

These ahead/behind counts will move with apache main. The important point is
that our migration has a small, inspectable delta over current apache/tvm and
apache/tvm-ffi, rather than a permanent divergent compiler fork.

---

## What we changed downstream

### 1. TVM base swap

We changed TileLang's TVM dependency from TileLang/tvm to apache/tvm, with a
nested apache/tvm-ffi checkout:

- .gitmodules: 3rdparty/tvm -> https://github.com/DatasunriseOU/tvm
  for the proof-of-concept branch. The upstreamable version would point to
  apache/tvm or to a small TileLang-owned pin repo.
- 3rdparty/tvm/3rdparty/tvm-ffi -> https://github.com/DatasunriseOU/tvm-ffi
  for the proof-of-concept branch. The upstreamable version would point to
  apache/tvm-ffi.
- TileLang source was adapted to the apache/tvm tirx namespace split.
- TileLang source was adapted to the new tvm-ffi ABI and runtime object APIs.
- Compatibility fixes were added for apache/tvm changes accumulated since the
  TileLang fork's December 2025 merge base.

The current migration branch also contains recent TileLang upstream refactors,
including the backend split work:

- #2138 split tl.copy lowering by backend.
- #2153 split GEMM implementations by backend.
- #2156, #2163, #2164, #2165 moved backend-specific lowering out of
  shared paths.
- #2191 and nearby fixes were cherry-picked into the downstream branch.

This matters because our Metal backend now fits the new backend layout instead
of fighting it.

### 2. Z3 prover rescue, but not as a TVM patch

The Z3 chronology in TileLang/tvm is important:

- e633295de9 on 2025-11-28 introduced Z3 integration with TVM.
- 0297c0b016 on 2025-12-08 made Z3 optional.
- e9f9392c07 on 2025-12-12 moved toward static Z3 linking.
- 75142425bc on 2025-12-15 maintained the z3_prover_off.cc stub path.
- 6dc8b76f10 and 79ed747db6 on 2025-12-19/20 hardened Z3 context
  lifetime and thread safety.
- d9d3e9dff8 on 2025-12-22 removed the 3rdparty/z3 submodule only. It did
  not remove the src/target/z3/z3_prover_on.cc and
  src/target/z3/z3_prover_off.cc sources.
- ce96c6085e, 65ae814bab, b82c74d2e2, and 8d494cacae continued
  improving timeout behavior, bitwise support, BV/int conversion, and value
  enumeration after the submodule removal.
- Current tilelang_main still has the TVM-core style Z3 integration:
  Analyzer owns a z3_prover, Analyzer::Bind forwards to it, and
  USE_Z3 defaults OFF in TVM's CMake.

Our downstream migration keeps the idea, but moves ownership out of TVM:

- The prover lives in tilelang under src/transform/vendored/.
- apache/tvm is not patched to add Analyzer::z3_prover.
- Existing TileLang call sites keep using tvm::arith::Z3Prover(analyzer);
  in our branch this is a free-function shim returning a per-Analyzer cached
  tilelang::tlz3::Z3Prover.
- Constraint helpers live beside it:
  - src/transform/vendored/z3_prover.cc
  - src/transform/vendored/z3_prover.h
  - src/transform/vendored/z3_constraint_scope.h
  - src/transform/vendored/z3_proof_hooks.h
  - src/transform/vendored/z3_prover_stub.h

Current implementation size:

text
1407 src/transform/vendored/z3_prover.cc
 237 src/transform/vendored/z3_prover.h
 175 src/transform/vendored/z3_constraint_scope.h
 125 src/transform/vendored/z3_proof_hooks.h
  12 src/transform/vendored/z3_prover_stub.h
1956 total


The prover is used by:

- src/backend/cuda/op/copy_analysis.cc
  - TMA legality proof, including 16-byte alignment checks.
- src/transform/drop_provable_bound_checks.cc
  - drop buffer-bound checks only when Z3 proves the guard.
- src/transform/predicate_fusion.cc
  - fuse predicates only when loads remain well-defined.
- src/transform/thread_storage_sync.cc
  - barrier minimization and proof hooks.
- src/transform/loop_vectorize.cc
  - contiguity/vectorization proofs.
- src/transform/auto_double_buffer.cc
  - currently a safe-stub pass: it detects candidates and logs proof status
    but does not rewrite IR yet.
- src/transform/verify_parallel_loop.cc
  - model/debug output for verification failures.

There is also a granular runtime kill switch:

- TILELANG_DISABLE_Z3=1 disables all Z3 usage.
- TILELANG_DISABLE_Z3_TMA_LEGALITY=1, ..._PREDICATE_FUSION=1,
  ..._DROP_BOUND_CHECKS=1, etc. disable individual pass integrations.

Important upstream-readiness note:

The current downstream branch keeps USE_Z3 ON in top-level TileLang CMake so
we can exercise the rescued prover constantly. That is good for our branch but
not the right default for upstream. A clean upstream PR should either:

- flip TileLang's top-level USE_Z3 default to OFF and make the OFF path a
  real no-Z3 compile path, or
- make PR 2 explicitly experimental and CI-gated until the no-Z3 path is
  restored.

My recommendation: upstream PR 2 should default OFF, compile without Z3, link
without Z3, and use conservative fallbacks unless -DUSE_Z3=ON is requested.
That matches the original TileLang/tvm design and avoids adding a mandatory
solver dependency to normal TileLang builds.

### 3. Metal backend

The downstream branch adds Apple Metal as a first-class backend. It is
additive and follows the new src/backend/<target>/ layout.

Core C++ paths:

- src/target/codegen_metal.cc
- src/target/codegen_metal.h
- src/target/intrin_rule_metal.cc
- src/transform/metal_scalar_intrinsics.cc
- src/backend/metal/CMakeLists.txt
- src/backend/metal/op/copy.cc
- src/backend/metal/op/gemm.cc
- src/backend/metal/op/reduce.cc
- src/backend/metal/op/finalize_reducer.cc

Python/runtime paths:

- tilelang/backend/metal/__init__.py
- tilelang/backend/metal/gemm.py
- tilelang/jit/adapter/torch/metal.py
- tilelang/analysis/metal_graph_sync.py
- tilelang/analysis/metal_sync_proof.py
- tilelang/transform/metal_scalar_intrinsics.py
- tilelang/transform/metal_fragment_to_simdgroup.py
- tilelang/transform/metal_simd_lift.py
- tilelang/transform/metal_merge_round.py
- tilelang/transform/metal/mark_host_metal_context.py
- tilelang/tileop/metal_simdgroup.py
- tilelang/tileop/metal_gdn.py
- tilelang/tileop/metal_quant.py
- tilelang/tileop/gemm/gemm_metal.py

Tests and coverage notes:

- testing/python/metal/test_metal_codegen.py
- testing/python/metal/test_metal_codegen_linux.py
- testing/python/metal/test_metal_gemm_v2.py
- testing/python/metal/test_metal_gemm_v2_linux.py
- testing/python/metal/test_metal_reduce.py
- testing/python/metal/test_metal_reduce_perf_smoke.py
- testing/python/metal/test_metal_simdgroup_store.py
- testing/python/metal/test_metal_local_var.py
- testing/python/metal/test_fp8_scaled_matmul_metal.py
- testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py
- testing/python/metal/test_metal_internal_scaffolding.py
- testing/python/metal/metal_internal_runtime_coverage.md
- testing/python/analysis/test_metal_graph_sync.py
- testing/python/analysis/test_metal_sync_proof.py
- testing/python/transform/test_metal_pass_pipeline.py
- testing/python/transform/test_metal_merge_round_barrier.py
- testing/python/transform/test_tilelang_transform_metal_scalar_intrinsics.py
- testing/python/transform/test_loop_vectorize_z3_contiguity.py

Benchmarks and research notes:

- benchmark/matmul_metal/benchmark_matmul_metal.py
- docs/research/apple_silicon_fp8_hardware.md
- docs/research/fp8_codegen_patterns.md
- docs/research/fp8_simdgroup_layout.md
- docs/research/numerical_parity_metal.md
- docs/research/nvfp4_mlx_metal.md

This backend does not require CUDA/HIP behavior changes. The PR should still be
reviewed carefully because it touches shared lowering and JIT adapter paths.

### 4. MLX interop

The branch adds an optional bridge for running TileLang Metal kernels from
Apple MLX tensors with DLPack/stream interop:

- src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_ext.cpp
- tilelang/contrib/mlx_interop.py
- tilelang/contrib/mlx_tvm_ffi.py
- tilelang/jit/adapter/tvm_ffi.py
- testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py

This lets a TileLang Metal kernel operate directly on mlx.array tensors
without a host round-trip.

Recent local hardening in this area includes:

- runtime fallback across MLX command-buffer API variants
  (metal::current_command_buffer(Stream) vs older
  Device::{end_encoding,get_command_buffer} symbols).
- native Metal bfloat emission at the ABI boundary instead of private
  helper types for owner-output buffers.
- graph-safe Metal sync metadata and device-event wiring.

This should probably be a separate PR from the core Metal backend. The Metal
backend is broadly useful; MLX interop is valuable but optional and depends on
Apple's MLX runtime details.

### 5. Smaller fixes that can be standalone PRs

These are good candidates to upstream independently, even if the full TVM
migration takes longer:

- b0ba04fb: carver dtype checks use bit-widths instead of string equality.
  This is more robust for bf16/fp8 variants.
- 5cb13313: wrap autotune cache misses in a lock.
- 4f18ef67: clearer PyTorch AOT Autograd version compatibility check.
- 031e9a25: unify TCGEN5MMA 2CTA annotation logic.
- subset of 386a4552: atomic float16/bfloat16 support and loop-vectorization
  ramp configuration improvements.
- subset of 29e277a0: restrict TMA lowering to Hopper / SM100 targets.
- Metal-specific fixes that may have shared value:
  - local-var conditional-store codegen fix.
  - bfloat16 Metal ABI/store fixes.
  - stream/command-buffer fallback fixes around tvm-ffi interop.

---

## Apache upstream candidates

While preparing this migration, we found changes in DatasunriseOU/tvm and
DatasunriseOU/tvm-ffi that look independent of TileLang. We can submit them
directly to apache as small PRs. Every accepted apache PR reduces the eventual
TileLang migration delta.

Likely apache/tvm candidates:

- arith::Analyzer Bind / EnterConstraint hooks for external sub-analyzers.
  This is the clean version of the extension point that the old TVM-core Z3
  integration relied on.
- iter_affine_map divisibility proof zero-check fix.
- TVM_LIBRARY_PATH support for runtime library discovery.
- fixes against the tvm/tirx/* namespace migration in:
  - stmt_functor
  - script_complete
  - split_host_device
  - storage_rewrite
- Metal runtime improvements:
  - TVM_METAL_STORAGE_MODE opt-in for Shared/Managed buffers.
  - borrowed command-buffer support for external frameworks.
  - bf16 runtime header restoration.
  - preserve pragma_unroll_factor through Metal codegen.
  - add tirx.metal.simd_sum and related visitor support.
- small string / any-union codegen fix.

Likely apache/tvm-ffi candidates:

- Python 3.13 core-loading fix.
- DLPack error-path test coverage.
- dev-tree/editable-import fix.
- reflection overload compatibility needed by current TileLang/tirx code.

---

## Proposed PR sequence

### PR 0: maintainer decision and fork inventory

Before opening a huge diff, we should agree on the interpretation of
TileLang/tvm:

- Is tilelang_main intended to remain a permanent fork?
- Which local tilelang_main TVM commits are still required by TileLang?
- Which commits are already obsolete because apache/tvm now has equivalent
  functionality?
- Which commits should move into TileLang proper instead of TVM core?
- Which commits should be upstreamed to apache/tvm?

This can be done as a short issue/discussion with a table of local TVM commits.

### PR 1: apache/tvm + tvm-ffi base swap

Scope:

- switch the TVM submodule away from TileLang/tvm.
- add or expose the nested tvm-ffi submodule expected by apache/tvm.
- apply the mechanical tir -> tirx and tvm-ffi ABI adaptation in TileLang.
- keep CUDA/HIP behavior intact.
- exclude Metal, MLX, and Z3 rescue unless needed for build compatibility.

Goal:

- TileLang builds against apache/tvm and apache/tvm-ffi with the smallest
  possible behavioral change.

This is the heavy lift. Our migration branch is the prior art, but the PR
should be cleaned into a reviewable base-swap diff.

### PR 2: optional vendored Z3 prover

Scope:

- move the Z3 prover into TileLang, not TVM core.
- keep apache/tvm unpatched.
- provide tvm::arith::Z3Prover(analyzer) as a compatibility shim for
  TileLang call sites.
- include per-pass runtime gates.
- restore a true no-Z3 compile path.
- default upstream builds to USE_Z3=OFF.

This PR must be skippable. If maintainers do not want Z3 in TileLang, PR 1 and
the remaining migration work should still stand.

### PR 3: Metal backend

Scope:

- Metal codegen, intrinsics, backend ops, Python backend registration,
  transforms, tests, benchmarks, and docs.
- no MLX interop in the first Metal PR unless maintainers prefer bundling it.

Goal:

- make Apple Metal a first-class TileLang target without regressing CUDA/HIP.

### PR 4: MLX interop

Scope:

- optional contrib bridge for MLX arrays, DLPack, and Metal stream /
  command-buffer interop.
- tests under testing/python/metal/.

Goal:

- enable zero-copy MLX execution while keeping MLX optional.

### PR 5+: small independent fixes

Submit the smaller fixes as separate PRs:

- carver bit-width dtype checks.
- autotune cache locking.
- PyTorch AOT compatibility guard.
- TCGEN5MMA 2CTA annotation unification.
- fp16/bf16 atomic support.
- loop vectorization ramp config.
- Hopper/SM100 TMA restriction.
- isolated Metal codegen correctness fixes.

These do not need to wait for the full migration if maintainers want them
sooner.

---

## Why this helps TileLang

- TileLang can stop carrying a long-lived TVM fork if the remaining local
  deltas are upstreamed, moved into TileLang, or made obsolete.
- Tracking apache/tvm reduces drift around tvm-ffi, tirx, TVMScript, runtime,
  and frontend changes.
- The migration converts a hidden fork-maintenance cost into explicit,
  reviewable PRs.
- Metal and MLX support become additive backend features rather than local
  downstream patches.
- The old Z3 idea survives without requiring a TVM-core patch or mandatory
  solver dependency.

---

## Open questions for maintainers

1. Is TileLang/tvm tilelang_main intentionally a long-term fork, or is it a
   staging branch that can be unwound?
2. Which tilelang_main TVM commits are still required for current TileLang?
3. Would you prefer 3rdparty/tvm to point directly at apache/tvm, or to a
   TileLang-owned pin repo that tracks apache/tvm plus short-lived hotfixes?
4. Should the migration target apache/tvm tags, or periodically track main?
5. What CI matrix would you require for the base swap? We can contribute Linux,
   macOS, CUDA, and Metal configurations from our side.
6. Would you accept a vendored Z3 prover if it defaults OFF and has a true
   no-Z3 build path?
7. Would you accept Apple Metal as a first-class backend?
8. Should MLX interop ship separately from the Metal backend?

---

## References

- apache/tvm tirx namespace introduction:
  https://github.com/apache/tvm/pull/18913
- apache/tvm TVMScript dialect refactor:
  https://github.com/apache/tvm/pull/19479
- apache/tvm cooperative_tensor + metal storage scope:
  https://github.com/apache/tvm/pull/19423
- apache/tvm-ffi:
  https://github.com/apache/tvm-ffi
- TileLang/tvm Z3 history:
  - e633295de9 introduced Z3 integration on 2025-11-28.
  - 0297c0b016 made Z3 optional on 2025-12-08.
  - d9d3e9dff8 removed only the 3rdparty/z3 submodule on 2025-12-22.
  - 8d494cacae added CountSatisfyingValues on 2026-02-04.
  - current tilelang_main still contains src/target/z3/z3_prover_on.cc
    and src/target/z3/z3_prover_off.cc.
