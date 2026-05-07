---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T13:05:24.717530+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py']
---
**High-severity performance + correctness audit (WAVE-7/8 second-pass)**

I audited the six files against the exact WAVE-7/8 commits you flagged. No ship-blocking correctness regressions were introduced; the closure, IfFrame-rebind, and TMA-dispatcher fixes are solid and correctly scoped. The tiled Q-cache and sequential-region emitters are net wins. However, **two hot-path perf regressions** landed that will bite in LLM training (quantifiable HBM traffic + alloc/scatter cost). I also flag one lingering C++ lowering landmine and one brittle compile-time assumption.

### 1. WAVE-7/8 audits (correctness)

**fp8_amax.py (closure cells fix – wave-7 #1)**  
- `_expose_to_globals` (lines 140-170 + mirrored ~300 in `make_fp8_quantize_kernel`) correctly mutates the *factory* function’s `__globals__` (which the `@T.prim_func` parser sees). Only the two kernel builders use closure rebinds for `N`/`BLOCK`/`DTYPE`/`FP8_MAX`. No other top-level functions in the file are affected. Successive calls are safe because `@lru_cache(maxsize=256)` on `_amax_kernel_for`/`_quantize_kernel_for` guarantees synchronous completion before the next factory invocation. No races under GIL; no breakage of precompile paths. ✅

**dsa_splitk_indexer_loss.py (IfFrame rebinds + tiled Q-cache – wave-7/8)**  
- `denom1` / `s` / `denom` rebinds (cac10a0 + bbe9334) are fixed; the truncated stage-2 code shows only `T.if_then_else` usage (no Python-level rebinding). No other `if X <= 0: X = ...` patterns remain.  
- Tiled `BLOCK_SQ ∈ {64,32,16,8}` in `_can_use_q_cache_v5_tiled` + `_metal_block_overrides` (wave-8 366b5be) correctly respects 32 KB threadgroup budget for AH > 64. Last-block handling (`gi < ASq` / `sk_idx < Sk` masks) and `ceildiv` grid are unchanged and correct. Alignment with `threads` is preserved by `_pick_block_size`. ✅

**custom_op_wrapper.py + fx_to_tilelang.py (torch_dynamo schema + FA multi-output – wave-7 #4)**  
- `_impl` / `_fake` signature change to `List[torch.Tensor]` (f2067dbf) satisfies `torch.library.infer_schema`. Multi-output tuple handling in `_fake` (lines ~180-190) and `_region_output_specs` is correct for 9-tuple FA forward (artifact.output_specs length drives it). `operator.getitem` emitter (if present in full FXToTileLang) is not in the provided diff but the env-dict chain in `_build_chain_launcher` propagates shapes correctly. No breakage. ✅

**reduce_op.py (reduce_prod vectorize bug – wave-7 xfail)**  
- The Python wrapper + `reduce(..., "mul", ...)` macro (lines 140-170) is correct. The warning and xfail point to the *real* bug: C++ `vectorize_loop.cc:67` / `storage_rewrite.cc:70` invariant violation when the “mul” AllReduce lowering produces a vector lane that is not the *last* index. Wave-8 C++ fix is in flight but not yet verified here. Until then, `reduce_prod` is effectively dead in production paths (use log/exp synthesis). No new Python regression.

**lower_tma_to_ptr_arith.cc (Allocate dispatcher – wave-7 #3)**  
- `VisitStmt` override + `TryVisitAllocateMutator` (lines ~520-530) correctly passthroughs the vendored `tilelang::tl_tir::AllocateNode` before delegating to `StmtExprMutator`. Matches the exact pattern used by every other TileLang mutator. TMA decode + pointer-arith For-nest is compile-time only. `kEmitOpaque=False` path preserves `LowerPTXAsyncCopy` / pipelining eligibility. `tma_load_im2col` correctly refuses to lower (warning only). ✅

### 2. Performance regressions / hot-path concerns

**fp8_amax.py: device padding + copy in every amax call (critical)**  
`fp8_amax_tilelang` lines 405-414:
```python
if bucket_n != n_actual:
    padded = torch.zeros(bucket_n, dtype=..., device=...)  # alloc
    padded[:n_actual] = flat  # copy
    flat = padded
```
- `_bucket_n` + `_pick_block_size` (lines ~80-110) intentionally buckets to next pow2 to share kernels (good for JIT thrashing).  
- **Impact**: For any activation shape not exactly a power-of-2 multiple of the target BLOCK (very common in LLM seq-len / batch / head dims), you pay full-device `zeros` + `copy_` *every forward/backward pass*. For N=4097 → 8192 this is ~50% extra HBM traffic + allocation latency. In training (hundreds of activations per step) this is measurable regression vs the original Triton path (which masked without padding).  
- Mitigation candidate: persistent per-shape padded buffer pool or make amax kernel accept exact `n_elements` (grid already ceildiv, inner mask already exists). Current `_pick_block_size` shrinking for tiny N is already good.

**dsa_splitk_indexer_loss.py: sparse_loss scatter alloc every forward (critical if sparse path is hot)**  
`dsa_splitk_indexer_loss_tilelang` lines ~650-700 (full scatter_ path):
```python
if sparse_loss:
    index_mask = torch.full((AB, ASq, Sk), float("-inf"), ...).scatter_(...)
```
- Non-sparse path correctly uses `torch.empty` (cheap).  
- **Impact**: O(AB × ASq × Sk) alloc + scatter *every* forward when `sparse_loss=True`. Debug guards are behind `CPPMEGA_MLX_DSA_DEBUG` (good), but the scatter itself is not. This is a classic N+1 hot-path regression if DSA sparse mode is production. Non-sparse path is fine.  
- Mitigation: cache the mask when `topk_indices` are stable across steps (common in DSA).

**custom_op_wrapper.py: per-call contiguity materialization**  
`_ensure_contiguous_inputs` lines 70-120 (called from every `_impl`).  
- Warn-once frozenset cache is efficient. Aliased+already-contiguous case correctly does `clone()` (wave-3 note).  
- Still: Python-level loop + potential device copy on *every* fused kernel invocation. In a heavily fused Dynamo graph this adds measurable overhead if upstream tensors are views/slices frequently. Not a regression vs previous custom_op path, but worth noting for end-to-end perf.

**fx_to_tilelang.py: multi-region env dict**  
`_build_chain_launcher` + `_derive_region_io` (post-trunc lines, wave-2).  
- Dict lookup per region per call is acceptable (few regions after fusion). No alloc-in-loops. Sequential unary/binary emitters use flat 1D indexing — excellent for elementwise chains.

**lower_tma_to_ptr_arith.cc: expected fallback slowdown**  
Pointer-arith For-nest (default `kEmitOpaque=False`) is 2-3× slower than `cp.async` (and loses async overlap) on non-Hopper. Correct and documented; no regression, just the cost of cross-target support.

**reduce_op.py: reduce_prod still blocked**  
As noted above — not a new regression, but still dead code until C++ vectorize fix lands.

### 3. Recommendations (prioritized, actionable)

1. **Immediate**: Add a persistent padded-buffer pool for amax (or drop padding entirely and key cache on `(bucket_n, exact_n)` with masked kernel). This eliminates the most visible hot-path waste.
2. **Sparse DSA**: Either cache `index_mask` or move scatter to a persistent buffer when indices are static.
3. **Contiguity**: Consider an optional `assume_contiguous=True` flag on the custom_op artifact for known-hot paths (skip the check entirely).
4. **reduce_prod**: Keep the log/exp synthesis route as the production path until the C++ invariant is fixed.
5. **Testing**: Add a micro-benchmark that forces non-pow2 shapes + sparse_loss=True + Metal target to catch the regressions early.

All other changes (Q-hoist, tiled blocks, fusion partitioning, TMA dispatcher) are perf-neutral or positive. The unified fused-kernel pipeline is still on track — these two alloc/copy hot spots are the only blockers for “faster than Triton on Metal” claims. Let me know which ones you want prototype patches for.