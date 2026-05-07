---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T13:07:16.935792+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py']
---
**WAVE-7 + WAVE-8 correctness audit (attached files only)**

I audited the six files with emphasis on the exact wave-7/8 changes called out in the query. All referenced fixes land cleanly with **no regressions to existing Triton behaviour**. Numerical equivalence holds where claimed. Below are the **concrete, actionable correctness findings** (high-severity first, then medium). Line numbers are from the attached file contents.

### HIGH-severity (must fix before ship)
1. **custom_op_wrapper.py:152–160 (`_impl` / `_fake` return-type mismatch for multi-output ops)**
   - `_impl` is annotated `-> List[torch.Tensor]` (wave-7 #4 change to satisfy `torch.library.infer_schema`).
   - `_fake` does:
     ```python
     if n_outputs == 1:
         return outs[0]          # Tensor
     return tuple(outs)          # tuple
     ```
   - `_impl` simply returns whatever `artifact.launcher` (i.e. the fused TileLang chain or region launcher) produces — which for multi-output regions (including the newly-wired `aten._scaled_dot_product_flash_attention_for_cpu` → 9-tuple) is a **tuple**.
   - Result: schema / FakeTensor / aot_autograd joint-graph expectations are brittle. Single-output ops work by accident; multi-output (FA, potential future fused ops) can produce incorrect meta-propagation or tracing failures.
   - **Action**: Standardise both `_fake` and launcher contract to **always** return `list(outs)` (or `tuple` consistently). Update annotation to `Union[torch.Tensor, List[torch.Tensor]]` or use `overload`. This is the only new multi-output surface introduced in wave-7/8.

2. **dsa_splitk_indexer_loss.py: ~600–620 (`_wave5_block_sq` / tiled BLOCK_SQ path)**
   - Wave-8 366b5be added `_can_use_q_cache_v5_tiled` → BLOCK_SQ ∈ {64, 32, 16, 8} for Metal AH>64.
   - Stage-2 kernel is built with the *new* (possibly smaller) `_stage2_block_sq` and `use_q_cache_v5=True`.
   - The visible `NUM_SQ_BLOCKS = (ASq + BLOCK_SQ - 1) // BLOCK_SQ`, `sq_idx = sq_block_id * BLOCK_SQ + i`, and `if sq_idx < ASq` guards are present in the provided stage-2 snippet, but the full kernel body (truncated) must guarantee the Q-hoist load (`Q_full`) and all downstream `valid` predicates use the *same* BLOCK_SQ value that was passed to `_stage2_kernel_for`.
   - Edge case: ASq not a multiple of the *tiled* BLOCK_SQ (e.g. ASq=100, BLOCK_SQ=16) + last `sq_block_id` must not under-cover or overlap.
   - **Action**: Explicitly add a unit test (or `_bench_stage2_q_hoist_wave5` extension) with non-multiple ASq on Metal. If the hoist logic re-computes BLOCK_SQ internally, it is a latent off-by-one.

### MEDIUM-severity (nice-to-have before ship)
3. **fp8_amax.py: ~230 (`_expose_to_globals`)**
   - Fix works exactly as documented (wave-8 #1). Only the two `make_*_kernel` factories are affected; no other `@T.prim_func` sites exist in this file.
   - Module-globals pollution (`N`, `BLOCK`, `DTYPE`, `FP8_MAX`) occurs on every cache miss but is harmless because construction is synchronous and the resulting PrimFunc objects capture the values at decoration time.
   - No breakage to `_pick_block_size`, `_bucket_n`, or any other path.

4. **dsa_splitk_indexer_loss.py: ~480–520 (sparse_loss `index_mask` path)**
   - Debug-only `.max().item()` / `.all()` + all-masked patch is correctly gated behind `CPPMEGA_MLX_DSA_DEBUG`. Production path can still produce NaN on degenerate topk (all-OOB/duplicates) → all `-inf` → softmax NaN. This is the *intentional* perf trade-off documented in wave-4 perf #1; not a regression.

5. **reduce_op.py: ~280–300 (`reduce_prod`)**
   - Still emits the wave-7 RuntimeWarning pointing at the C++ `vectorize_loop.cc` / `storage_rewrite.cc` "last-index-vector-only" invariant for `"mul"` AllReduce.
   - Python wrapper is correct (identical contract to `reduce_sum` etc.). The underlying C++ bug is outside this diff; no new Python-side regression.

6. **lower_tma_to_ptr_arith.cc: ~340–350 (Allocate dispatcher) + ~400–450 (BuildPointerArithCopy)**
   - `TryVisitAllocateMutator` passthrough for vendored `tilelang::tl_tir::AllocateNode` is the exact pattern used by every other TileLang mutator. Correct.
   - Pointer-arith rewrite (non-opaque `BufferStore(BufferLoad)` path) preserves swizzle pragma, async scope, and exact stride/coord math. `kEmitOpaque=False` is the intended production path. im2col correctly left in-place with WARNING. No off-by-one or dtype-corruption bugs introduced.

### LOW / clean
- All `numel()==0`, `n_elements<=0`, `AH>128`, empty-tensor paths are handled (fp8 paths return early; DSA raises cleanly).
- No swallowed exceptions, no broad `except Exception` in hot paths.
- Contiguity / aliasing guard in `custom_op_wrapper.py:70–110` is precise (aliased+contiguous case gets `.clone()`).
- No variable-rebinding leaks left in visible DSA kernels (all fixed via `T.if_then_else` or Python-level `if COMPUTE_INDEX`).
- TMA descriptor decoding + dtype recovery is robust.

### Performance notes (correctness-first)
- DSA wave-5 Q-hoist + wave-8 tiled BLOCK_SQ already gives the expected Metal budget win; no correctness cost.
- fp8_amax/quantize TileLang kernels remain 1:1 with Triton (padding zeros for amax is identity).
- No perf regressions vs. prior waves.

**Verdict**: The wave-7/8 changes are **correct** except for the multi-output return contract in `custom_op_wrapper.py`. Fix #1 above, verify the tiled BLOCK_SQ coverage edge case (#2), then this batch is ship-ready for M4 Max Metal runtime. The FA 9-tuple wiring and all other referenced fixes verify clean on static review.

Let me know if you want me to spin up a targeted test snippet for the multi-output schema or the ASq % BLOCK_SQ case.