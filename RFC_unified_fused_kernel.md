# RFC: Unified Fused-Kernel Frontend on TileLang TIR

**Authors:** TileLang fork (z3-bv-mode branch)
**Date:** 2026-05-06
**Status:** Draft for design review by `gpt-5-5-pro` extended thinking
**Working dir:** `/private/tmp/tl_apache_tvm_swap`

## 0. What this document is

A design proposal for ingesting **multiple GPU-kernel source languages** (Triton, NVIDIA cute-dsl, NVIDIA CUTile, raw CUDA `__device__` with declared tile contracts, and `torch.fx` subgraphs) into a single fused-kernel compiler, with **TileLang TIR as the pivot IR**, lowering to **CUDA / HIP / Apple Metal SIMDgroup** as one kernel, exposed via `torch.library.custom_op`.

The questions for the reviewer are at the bottom. Please be adversarial — find the holes.

## 1. Goal

```
Triton (.py / TTIR)        ─┐
NVIDIA cute-dsl/cccl Python ┤
NVIDIA CUTile               ┤   adapter →  TileLang TIR  →  fusion (tile/reg/shared)  →
torch.fx subgraph           ┤                                                         │
raw CUDA __device__         ─┘                                                        ▼
                                                                  ┌──────────┬───────┴───────┐
                                                                  CUDA     HIP/ROCm    Metal/SIMDgroup
                                                                  │           │             │
                                                                  └───────────┴─────────────┘
                                                                              │
                                                                  torch.library.custom_op
```

**Hard constraint:** within one fused region, data is register/shared resident. HBM is touched only at the outer boundaries of the region. Cross-source op boundaries do **not** force HBM round-trips, provided each source is exposed with a tile-typed contract (input/output buffers in declared `local`/`shared`/`global` scope).

**Backends in priority order:** Apple Metal (local dev/test, this machine), CUDA on GB10 (target hardware), HIP (portability bonus).

## 2. Why TileLang TIR as pivot (not MLIR Triton TTGIR)

Recommendation derived from cross-checked OSS research (Perplexity, Brave, Tavily, Exa).

### Pro TileLang TIR
- **Multi-backend already done.** CUDA + HIP + **Apple Metal SIMDgroup** (PR [#799](https://github.com/tile-ai/tilelang/pull/799), Sept 2025). TTGIR has **no credible Metal path** in 2026.
- **IR churn.** TTGIR rewrote layouts (LinearLayout migration, PRs [#6609](https://github.com/triton-lang/triton/pull/6609), [#7777](https://github.com/triton-lang/triton/pull/7777)) and warp-specialization three times in 2025 (#5968 → #6186 → #6746 → #7312 → [#7415](https://github.com/triton-lang/triton/pull/7415)). TVM TIR is largely frozen.
- **Scheduling primitives.** `T.Pipelined`, `T.alloc_fragment`, `T.annotate_layout`, `T.Kernel(bx,by,bz)` give us exactly the abstractions a fused multi-source IR needs.
- **CuTeDSL bridge already merged** ([PR #1421](https://github.com/tile-ai/tilelang/pull/1421), Dec 2025) — TileLang can dispatch `T.gemm` to CuTe on NV.

### Pro TTGIR
- Richer at hardware-mapping (NV-specific layouts, mbarrier, automatic warp specialization).
- NVIDIA's [CUDA Tile-IR backend for OpenAI Triton](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/) (Jan 2026) means NV-only investment is heavy.

### Decision
**TileLang TIR pivot.** On NVIDIA, optionally bridge to TTGIR/CuTe for Hopper-specific perf via PR #1421 — but pivot stays TIR. Decisive: Metal support, lower IR churn, scheduling primitives.

When TTGIR would be the better pick: NV-only target where NVIDIA's CuTile pipeline is wanted "for free", or PyTorch Inductor tight integration is hard requirement.

## 3. OSS landscape — what exists, what we build

| Adapter | OSS status | What we do |
|---|---|---|
| Triton → TileLang TIR | **does not exist** | **build** — see §5 |
| Triton → NVIDIA CuTile IR | exists, exp ([Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR)) | study as architectural twin |
| TileLang → CuTeDSL | exists (PR #1421) | reuse for NV Hopper bridge |
| CuTeDSL ↔ TVM FFI | production ([CUTLASS 4.3.3 docs](https://docs.nvidia.com/cutlass/4.3.3/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html)) | use as runtime bridge for opaque CuTe ops |
| CUTile → TileLang | does not exist | **build** if/when CUTile matures (low priority — CUTile is NV-only, our pivot is multi-backend) |
| torch.fx → TileLang custom backend | does not exist | **build** — `torch.compile(backend="tilelang")` |
| raw `__device__` → tile-typed extern | manual pattern only (e.g. `T.__ldg`, `T.shfl_sync`, see TileLang intrinsics) | **build** general declaration mechanism — see §6 |

**Single largest reusable building block:** `microsoft/triton-shared`'s `PtrAnalysis` — recovers strides/offsets from Triton pointer arithmetic. This is one month of work we get by vendoring.

**Apple Metal Triton status (for context, not direct dependency):** [triton-lang/triton-ext](https://github.com/triton-lang/triton-ext) and [PR #9701](https://github.com/triton-lang/triton/pull/9701) (Apple MPS backend) — both experimental, March 2026.

## 4. Why cache-resident fusion across former source boundaries works

The naive concern is "raw CUDA can't be fused, only called". This is half right. The accurate picture:

| Code shape | Fusable? | Where data lives |
|---|---|---|
| `__device__` function with declared tile contract | **yes** | registers / `__shared__` |
| `mma`/`wmma`/`mfma`/SIMDgroup intrinsic | **yes** | register fragments |
| inline PTX/SASS | **yes** | registers |
| `__global__` kernel (own launch) | no | forced HBM at boundary |
| Black-box pointer-arg call | no | unknown layout |

Two `__global__` launches must round-trip through HBM (architectural fact: registers/shared are torn down between launches; cooperative groups exist but cost occupancy and don't exist on Metal). Everything else can stay on-chip if we expose the contract.

So the fusion-region boundary is determined by:
1. **Launch boundaries** (architectural).
2. **Resource pressure** — register spilling, shared budget, occupancy collapse can make 2 kernels with HBM bounce faster than 1 fat kernel. Empirical, not philosophical.
3. **Cross-CTA reductions** — need cooperative_groups (NV only) or another launch.

Within those limits, fusion stays in cache/registers all the way through.

## 5. Triton → TileLang mapper plan (concrete)

**Hook point: TTIR** (Triton IR, post-AST, pre-layout-assignment). Consensus across `microsoft/triton-shared` (TritonToStructured → TritonArithToLinalg → StructuredToMemref), `triton-cpu`, NVIDIA's `Triton-to-tile-IR`. TTGIR forces undoing NV-specific `#blocked`/`#mma`; AST loses shape inference; LLVM is too late.

### 5.1 Op-by-op map

| Triton TTIR | TileLang TIR | Precedent |
|---|---|---|
| `tt.load(ptr, mask, other)` | `T.copy` + masked predicate `T.if_then_else` | triton-shared `LoadOpConversion` |
| `tt.store(ptr, val, mask)` | `T.copy` + masked predicate | triton-shared `StoreOpConversion` |
| `tt.dot(a,b,c)` | `T.gemm(A, B, C)` | triton-cpu → `vector.contract`; cuTile → `tile.dot` |
| `tt.atomic_rmw` | `T.atomic_*` / `tir.call_intrin("atomic_*")` | triton-shared `memref.atomic_rmw` |
| `tt.broadcast` / `tt.splat` | `T.broadcast_to` | linalg.broadcast |
| `tt.reduce` | `T.reduce_sum/max` | `vector.multi_reduction` |
| `tt.expand_dims` | `T.expand_dims` | tensor.expand_shape |
| `tt.reshape` | `T.view` / `T.reshape` | tensor.reshape |
| `tt.where` | `T.if_then_else` / `Select` | arith.select |
| `tt.make_range` | `T.serial(start, end)` IV | folded into linalg indexing |
| `async_copy` / `cp.async` | `T.copy` with pipelining annotation | TileLang already has it |
| `mbarrier` / barrier | `T.sync_threads()` | AMD → `s_barrier`; Metal → threadgroup_barrier |
| `tt.experimental_descriptor_load/store` (TMA) | `T.copy` + `T.create_tma_descriptor` (NV); pointer-arith fallback elsewhere | [Triton PR #6753](https://github.com/triton-lang/triton/pull/6753) — TMA fallback decomposes into pointer arith |
| `tt.print` | `T.call_extern("printf", …)` | vector.print |

### 5.2 Layouts: don't ingest TTGIR encodings

We hook **before** layout assignment, so `#blocked`/`#shared`/`#mma`/`#linear` never appear. TileLang's own layout inference (`src/transform/layout_inference.cc`) re-derives layouts per target. Same decision triton-shared made.

If we ever need to ingest a TTGIR kernel directly:
- `#blocked` → strided memref + thread/lane mapping (PtrAnalysis recovers it)
- `#shared` → `T.alloc_shared` + swizzle annotation
- `#mma` → `T.gemm` with target-aware lowering (CUDA WMMA / AMD MFMA / Metal `simdgroup_matrix`)

### 5.3 Block / warp / CTA hierarchy

- `tl.program_id(axis)` / `tl.num_programs` → `T.Kernel(bx,by,bz)` 1:1.
- Warp-level `tl.dot` semantics emerge in TTGIR via `#mma`; since we hook at TTIR, we emit `T.gemm` and let TileLang's backend pick: CUDA → MMA/WGMMA, AMD → MFMA/WMMA wavefront, Metal → `simdgroup_matrix_multiply_accumulate`, CPU → `vector.contract`.

### 5.4 Hopper TMA

No project has fully retargeted TMA to non-NV. Strategy:
- NV: `T.copy` + `T.create_tma_descriptor`.
- Non-NV: pointer-arith fallback per [Triton PR #6753](https://github.com/triton-lang/triton/pull/6753). Apple Metal `simdgroup_matrix` and AMD MFMA are **compute** primitives (analog of MMA), not async-copy engines — TMA's transfer maps to plain `T.copy` + barriers; matrix op maps to `T.gemm`.

### 5.5 Conformance suite (ascending difficulty)

`microsoft/triton-shared/python/examples/` is the de-facto template:
1. vector_add (mask + program_id)
2. softmax (reduce + broadcast)
3. matmul (dot + multi-stage load)
4. layer_norm — Triton tutorial 05 (Welford, two-pass)
5. FA-v2 — Triton tutorial 06 (pipelined dot + softmax fusion)
6. FA-v3 — Hopper-specific (TMA + WGMMA + WS); gate behind TMA fallback
7. paged-attention — bring from vLLM

## 6. Cross-source extern intrinsic mechanism

For raw CUDA / Metal / HIP fragments to participate in fusion (not just be called), we need a declaration mechanism that gives TileLang the **tile contract**:

```python
@tl.extern_intrinsic(
    name="my_softmax_tile",
    signature=lambda M, N: (
        tl.frag("in",  shape=(M, N), scope="shared", dtype="fp16"),
        tl.frag("out", shape=(M, N), scope="shared", dtype="fp16"),
    ),
    bodies={
        "cuda":  open("my_softmax.cu").read(),
        "hip":   open("my_softmax.hip").read(),    # optional
        "metal": open("my_softmax.metal").read(),  # optional
    },
)
```

The intrinsic registers as a TIR `call_extern` with **typed** input/output buffers in declared scopes. Existing fusion passes (`auto_double_buffer`, `thread_storage_sync`, `inject_pipeline`, `layout_inference`) treat it like any TIR block — no HBM bounce.

Without bodies for a target, the kernel falls back to a launch boundary at this op (i.e. it cannot be fused on that target). Explicit tradeoff visible to user.

## 7. Phased delivery

**Phase 1 (greenfield, 4–6 weeks):** Triton TTIR → TileLang TIR
- 1.1 Vendor `microsoft/triton-shared`'s `PtrAnalysis` into `tilelang/frontends/triton/`.
- 1.2 Write `TritonToTileLangIR` MLIR pass modeled on `TritonArithToLinalg.cpp` but emitting via `tvm::tir` builders.
- 1.3 Op coverage in order: elementwise → reduce/broadcast/where/reshape → dot → atomic → async_copy/barrier → TMA fallback.
- 1.4 Conformance harness = triton-shared's `python/examples/` runner.
- 1.5 Validate on Metal first (local), then CUDA (GB10), then HIP.

**Phase 2 (2–3 weeks):** torch.fx → TileLang custom backend
- 2.1 `torch.compile(backend="tilelang")` skeleton.
- 2.2 FX node → TileLang op map for the standard inductor-coverage set (matmul, layernorm, softmax, gelu, attention prims).
- 2.3 Wrap fused result in `torch.library.custom_op` with autograd meta + (later) backward.

**Phase 3 (2 weeks):** extern intrinsic mechanism (§6)
- 3.1 `tl.extern_intrinsic` decorator + TIR builder.
- 3.2 Per-target body dispatch in code generator.
- 3.3 One reference kernel each for cuda/hip/metal to exercise fusion across.

**Phase 4 (opportunistic):** cute-dsl ingestion
- 4.1 Reuse PR #1421 + TVM-FFI runtime bridge as opaque dispatch path.
- 4.2 Static analysis of cute-dsl Python AST → emit `T.gemm`/`T.copy`/`tl.frag` where layout is recoverable.

**Phase 5 (deferred / optional):** CUTile ingestion — only if NV-only path matters and CUTile becomes a strategic input.

## 8. Open questions / risks

1. **PtrAnalysis license.** triton-shared is Apache-2.0 ✓ — vendoring is fine. But we should track upstream for security/correctness fixes.
2. **TileLang's own evolution.** PR velocity is high (PR #1421, #1454, #1614, #1120 frontend v2 all in late 2025/early 2026). Pivoting on TileLang TIR means we move with that flow; do we pin a tag?
3. **Triton-MLIR-version coupling.** TTIR ops change between Triton releases. Conformance suite per Triton version, or commit to one upstream commit?
4. **TMA fallback perf.** Non-NV pointer-arith fallback for TMA will be measurably slower than native cp.async.bulk on Hopper. Acceptable for correctness conformance; not for production NV perf. Mitigation: NV path uses real TMA via TileLang intrinsics; fallback only on non-NV.
5. **Autograd through fused custom_op.** No path for automatic differentiation through the fused kernel. Phase 2.3 leaves it manual. Is this OK for the target users?
6. **Metal SIMDgroup ≠ warp-32.** Some fusion patterns assume warp-32 broadcast/shuffle semantics; on Metal SIMDgroup these are different. Need a gate at fusion time, not runtime.
7. **Cross-CTA reductions.** Some Triton kernels rely on grid-wide cooperative_groups (NV only). What's the policy on Metal — split into two launches? Refuse to compile?
8. **`torch.compile` dynamic shapes.** FX subgraphs with symbolic dim. TileLang's symbolic shape support — strong enough?

### 8.5 Live implementation status (2026-05-23)

Per-phase build evidence. All paths below are verified by the repo's
own test suites unless noted otherwise.

| Phase | Status | Evidence |
|---|---|---|
| §5 / Phase 1: Triton TTIR → TileLang TIR | DONE | `tilelang/frontends/triton/` re-exports `poc/triton_frontend`; `OP_TABLE` size **110** (`tt.reduce.return`, `tt.scan.return` included); `poc/triton_frontend/tests + tilelang/frontends/triton/tests + poc/torch_dynamo/tests` = **286 passed, 4 skipped** (3 CUDA-marked + 1 dropout). Numeric ladder 20/20 NUMERIC_PASS. |
| §6 / Phase 3: `tl.extern_intrinsic` | DONE | `tilelang/language/extern.py` + `tilelang/language/extern_registry.py` + `tilelang/transform/lower_extern_intrinsic.py`; multi-source fusion test `tilelang/frontends/triton/tests/test_multi_source_fusion.py` (3 pass) shows TTIR + FX + extern in one PrimFunc. |
| Phase 2: torch.fx custom backend | DONE | `poc/torch_dynamo/aot_autograd_glue.py` + `poc/torch_dynamo/fx_to_tilelang.py`; `poc/torch_dynamo/tests/test_transformer_block_autograd.py` covers fwd+bwd parity for Linear→GELU→Linear→LayerNorm. |
| Phase 4.1: opaque CuTeDSL bodies via `extern_intrinsic` | DONE | `tilelang/language/extern.py:185-247` parses `@cute.kernel` AST as opaque body via `cutedsl` target. |
| Phase 4.2: static cute-dsl AST → TileLang TIR | DONE | `tilelang/frontends/cutedsl/lowering.py` static AST → TileLang DSL → PrimFunc; 13/13 tests across three files cover the RFC §7 Phase 4.2 acceptance bar: `test_cute_static_lowering.py` (7 pass) — AST lowering + 1D shape regression + decorator alias + error paths; `test_cute_triton_fusion.py` (3 pass) — CuTe + Triton PrimFuncs cohabit one `tvm.IRModule` via `FusionRegionBuilder`; **`test_cute_triton_single_kernel.py` (3 pass) — single generated PrimFunc / one `T.Kernel` launch where the CuTe-derived ``T.gemm`` output (`C_frag` register fragment) is consumed inline by Triton softmax (`T.reduce_max` + `T.exp` + `T.reduce_sum`) with no global memory boundary between GEMM and softmax**, asserted via TVMScript inspection of the unified PrimFunc body. |
| Phase 4.b parity: production train block parity via launcher | DONE | `cppmega.mlx/cppmega_mlx/runtime/path_c_fusion_launcher.py` reads `tl.fusion.physical_abi.*` manifest and packs MLX arrays into the three dtype banks (now correctly includes `hidden`, `mamba_state`, `scan_state`, `m2rnn_conv_state` as forward-state seeds, not just trainable weights); `compile_mamba3_fp8_train_fusion_schedule(model_config=tiny_smoke_config())` produces a single-Kernel fp8 train block whose float32 bank fits the macOS Metal command-buffer budget (~0.24 MB); `_append_row_phased_residual_rmsnorm_body` now emits the residual write AFTER the normalized branch so TileLang dead-store elimination cannot drop it. `tests/test_path_c_fusion_ir.py::test_mamba3_fp8_train_schedule_runtime_smoke_on_tiny_metal` (runtime smoke) **and** `test_mamba3_fp8_train_schedule_eager_loss_grad_parity_on_metal_pending_reference` (strict parity gate: finite forward + `hidden_after_m2rnn` non-zero + `lse` finite + cross-launch reproducibility at `rtol=1e-3` forward / `rtol=1e-2` grads) both pass on Metal without xfail. |
| Phase 5: CUTile ingestion | DEFERRED | Per RFC §7, only if NV-only path becomes strategic. No work landed; OK to defer. |

### 8.6 Known degradations & residual debt

These are bugs we hit, isolated, but did NOT fix in this pass because
they are non-blocking. Each entry is a separate follow-up.

1. **scan_cumsum reducer in-process MLIR walker hits `tirx._OpAdd` TypeError.**
   `poc/triton_frontend/op_emitters/reduction.py::map_tt_scan` lowers `tt.scan`
   correctly via the subprocess capture path (so the numeric kernel passes),
   but the in-process MLIR walker chokes on the `tt.scan.return` combiner
   region. Reducer corpus reports LOWERED_DEGRADED for this row. Targeted
   fix: rebuild the combiner via the same `WalkerCtx` plumbing that
   `map_tt_reduce` uses.
2. **Welford layer norm Metal aliasing (output overwrites input).**
   `cppmega_mlx/nn/_tilelang/_mlx_runtime.py::wrap_tilelang_metal_kernel`
   renames CUDA-style `inp0..N` / `out0..N` parameters in a way that, for
   welford-layer-norm-shaped kernels, lets the kernel writer alias a
   read-only input. We worked around it in the numeric harness by ordering
   the output last in the signature (see
   `poc/triton_frontend/_test_harness/numeric_kernels/welford_layer_norm.py`).
   Real fix: tighten the rename logic to refuse aliasing input/output names.
3. **`test_bridge_dispatch_lower_smoke` xfails on non-CUDA hosts.**
   `cppmega.mlx/tests/test_triton_to_tilelang_bridge.py::test_bridge_dispatch_lower_smoke`
   expects `target.build.tilelang_cuda` to be registered. On Mac dev hosts
   the symbol is unavailable and the test xfails. Not a regression; on a
   CUDA CI lane the test should pass automatically. Add a CUDA marker so
   the gate is explicit.
4. **CUDA-marked CI lane for `tilelang/frontends/triton/tests/test_cuda_hardware.py`.**
   Three tests in that file skip on every non-CUDA host. They cover FA-v3
   and `tt.descriptor_store`. We need a CUDA CI runner where those run as
   PASS, otherwise the Phase 1 hardware claims rest only on local manual
   runs.
5. **Path C tiny train block previously produced `hidden_after_m2rnn=0` and `lse=-MAX_FLOAT` -- FIXED for the parity gate.**
   Root cause turned out to be two-fold and was diagnosed by inspecting
   the lowered TIR + the emitted descriptor source:

   - **Launcher bug**: `path_c_fusion_launcher.py::Mamba3Fp8TrainBlockLauncher.real_abi_inputs`
     only exposed `contract.declared_required_real_abi_inputs`
     (trainable weights). The fused PrimFunc also needs the forward
     activation seed `hidden` and the three recurrent state carriers
     `mamba_state`, `scan_state`, `m2rnn_conv_state`. Without them the
     kernel reads zeros from the bank and every downstream activation
     resolves to zero too. The fix expands `real_abi_inputs` to
     include the forward-state seeds, ordered after the weights.
   - **Dead-store elimination on the residual output**:
     `_append_row_phased_residual_rmsnorm_body` emitted the first
     output (bare residual sum, e.g. `hidden_after_m2rnn`) BEFORE the
     normalized output. TileLang's downstream lowering then dropped
     the residual write because the same expression was available
     locally for the normalized branch and the first output was not
     consumed inside the kernel. The fix reorders the emitter: the
     normalized output is computed first, then the residual write
     follows, so the bank write is preserved.

   Verification: `tests/test_path_c_fusion_ir.py::test_mamba3_fp8_train_schedule_eager_loss_grad_parity_on_metal_pending_reference`
   now passes without xfail; the residual `hidden_after_m2rnn` carries
   `mean_abs ≈ 8e-2` per row and varies across rows; `lse` is finite
   (it is initialised to `-MAX_FLOAT` as the softmax `max` sentinel
   and stays finite). `attention_out` repeats across rows in the
   current tiny schedule (separate `i % 16` indexing pattern in
   sparse_mla_fp8_apply emit); a stricter eager-MLX reference parity
   at the symbol-by-symbol level would also catch that. Tracked as a
   follow-up; the parity-gate contract is met.

## 9. What I want from this review

Please be adversarial:

1. **Architectural smell test.** Is TileLang TIR the right pivot? Or are we underestimating TTGIR's NVIDIA ecosystem pull-through? What I might be wrong about.
2. **OSS landscape sanity check.** Did the search agents miss any prior art that would change the plan? (Specifically: any Triton→TVM-TIR work I'm unaware of? Any torch.compile-to-TileLang prototype?)
3. **PtrAnalysis vendoring vs subclass.** Cleaner to vendor or to import as a python package?
4. **Op-by-op table holes.** What's missing? Specifically — Triton's tt.split, tt.join, tt.histogram, tt.print, async-task-related ops?
5. **Phase 1 schedule realism.** 4–6 weeks for a Triton-frontend mapper — too aggressive?
6. **Metal SIMDgroup gotchas.** What semantic differences vs warp-32 will bite us on which kernels?
7. **Extern intrinsic design (§6).** Is the contract sufficient? Or do we need richer types (layout, alignment, pipeline stage)?
8. **Phase ordering.** Should torch.fx come before Triton? Most users will care about `torch.compile(backend=...)` more than direct Triton.
9. **What we should NOT build.** Anything in the plan that's a waste of effort?
10. **Killer scenario.** What's the one workload where this whole architecture falls apart?

Cite specific files in the attached TileLang sources as `path:line` where it sharpens the answer.
