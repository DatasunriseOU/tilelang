---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T09:09:30.690771+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/Volumes/external/sources/cppmega/tests/test_dsa_splitk_tilelang.py']
---
**WAVE-3 FINAL REGRESSION REVIEW — CLEAN ON PRIOR FIXES, 3 REMAINING HIGH-SEVERITY HOT-PATH ISSUES**

Wave-1 + wave-2 + wave-3 changes (Q_full hoist in stage 1, partial Q hoist + MD_pre budget gate in stage 2, `_metal_block_overrides` AH-aware down-sizing, causal `_active_sk_tiles` trim, `USE_MD_PRE` register-spill guard, sparse NaN-patch guard, topk validation, `empty()` non-sparse path, `lru_cache` kernel factory, contiguous guards) have **not** introduced regressions or half-fixes.

- All prior perf fixes are live and functional (stage-1 Q hoist is outside the `sk_tile` loop; stage-2 partial hoist eliminates the old inner `d_tile` HBM Q loads; Metal always stays under 32 KB threadgroup/register budget; CUDA large shapes fall back gracefully without spill crashes).
- Numerical parity tests (dense + two sparse regimes + full-TOPK-degenerate + hand-crafted mask sign test + validation) all pass with the tightened tolerances.
- No new O(n²) loops, no allocations inside `T.Pipelined`/`T.serial` bodies, no missing caches, no Python-side blocking payloads, no memory growth patterns beyond the (unavoidable) per-call buffers.

**Remaining HIGH-severity performance issues (hot-path, training-throughput impact)**

1. **dsa_splitk_indexer_loss.py:1160–1175 (sparse_loss=True path in `dsa_splitk_indexer_loss_tilelang`)**  
   `topk_idx64.max().item()` + `min().item()` + `bool(_row_has_valid.all())` force full GPU→CPU synchronizations + extra reduction kernels **on every sparse forward pass**.  
   This is a latency regression vs. the original Triton wrapper (which did the scatter but no explicit bounds/NaN guards). For production sparse_loss workloads (TOPK << Sk) this adds measurable step-time overhead.

2. **dsa_splitk_indexer_loss.py:~700–720 (inside `make_dsa_splitk_stage2_kernel`, the `for sk_tile in T.Pipelined(...)` → `for h in T.serial(AH)` → `for i, dd in T.Parallel(BLOCK_SQ, AD): Q_full[i, dd] = Q[...]`)**
   Q tile is reloaded from HBM **SK_TILES × AH times per sq_block** (identical to stage 1’s successful hoist at ~lines 380–400).  
   On Metal (BLOCK_SK=32, Sk=4096) this is ~128× redundant Q bandwidth; on CUDA even higher. Directly hurts roofline / arithmetic intensity. The partial-hoist comment acknowledges the TODO, but the current state is still a major hot-path regression relative to an optimal single-load-per-h design.

3. **dsa_splitk_indexer_loss.py:~430–450 (stage-1 kernel, `idx_scores_f`, `m1_i`, `d1_i` etc. fragments)**  
   Index-softmax fragments + online-softmax accumulators are allocated **for every head** (via the uniform `T.alloc_fragment` before the `sk_tile` loop), but the entire index path (`if h == 0`) and final `M1`/`D1` write only execute for head 0.  
   Wastes register pressure on the 127/128 non-head-0 blocks (noticeable on Metal’s tight SIMDgroup budget and on CUDA when AH=128).

Everything else (kernel caching, block-constant selection, Metal vs. CUDA paths, test coverage, contiguous handling, non-sparse `empty()` optimization) is solid. The three items above are the only remaining HIGH-severity hot-path concerns in the current code. Once triaged, this package is ready to ship green.