---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T09:07:28.790938+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega/cppmega/megatron/fp8_activations.py', '/Volumes/external/sources/cppmega/tests/test_fp8_amax_tilelang.py']
---
**WAVE-3 FINAL REGRESSION REVIEW: CLEAN WITH ONE REMAINING HIGH-SEVERITY PERF ISSUE**

Wave-1 (kernel correctness, Metal/CUDA parity, partial-block/mask handling, nonfinite guard, zero-handling) + wave-2 (caches, bucketing invariants, dispatch ordering, TE preference, contiguous safety) fixes have **not** introduced regressions.  
- Dispatch in `fp8_activations.py:340-347` (TE → TileLang → Triton → unfused) is correct and stable.  
- `tilelang_supports` / `fp8_pack_tilelang` path (including MPS) works as intended.  
- All tests in `test_fp8_amax_tilelang.py` (parity, padding correctness, nonfinite, partial blocks, signed-zero) still pass with no new failures.  
- No new O(n²), redundant I/O, N+1, blocking calls, or memory growth in hot paths.  
- Kernel style matches `fp8_vecmat_path_c.py` (direct fragment load, no shared double-pass, `T.reduce_max` + `T.atomic_max`).  
- No allocation thrashing from repeated JITs (lru_cache 256 + `_bucket_n` + `_pick_block_size` still effective).

**Only remaining HIGH-severity performance issue (hot-path regression vs. original Triton):**

**fp8_amax_tilelang (fp8_amax.py:373-382)**  
```python
block, _threads = _pick_block_size(target, n_actual)
bucket_n = _bucket_n(n_actual, block)
if bucket_n != n_actual:
    padded = torch.zeros(bucket_n, dtype=flat.dtype, device=flat.device)  # <--- alloc
    padded[:n_actual] = flat  # <--- copy
    flat = padded
```
- Triggered on **every** FP8 pack for any activation shape that is not an exact power-of-2 bucket (extremely common: 4097, 5000, 8193, most layer activations, variable seq lengths, etc.).
- Introduces GPU allocation + memcpy on the *critical path* of gradient checkpointing (multiple tensors per microbatch per layer).
- Regression vs. original Triton `_amax_kernel` (exact `n_elements`, no padding, just mask in `T.Parallel`).
- Quantize path (`fp8_quantize_tilelang`) already uses exact `n_elements` and has no such cost — inconsistency in the fused pack flow.
- Impact: extra allocator pressure + temporary memory spike + measurable slowdown in training loops (especially with many checkpointed blocks or smaller tensors). The `_bucket_n` comment correctly identifies JIT thrashing as the original motivation, but the per-call cost now dominates.

This is the **only** blocker to marking the package green for ship. Everything else (caches, TE fallback, Metal atomic CAS, deferred fused-amax+quantize, precompile helper) is solid and production-ready.

**Recommendation (one-line fix):**  
Drop `_bucket_n` / padding entirely for amax (make `_amax_kernel_for` key on exact `n_elements` exactly like `_quantize_kernel_for`). Shapes repeat in training; lru_cache(256) is more than enough, and you eliminate the hot-path alloc/copy. Update the two tests that explicitly exercise bucketing and the `_bucket_n` unit test accordingly.

Once that single change lands, the TileLang Path C FP8 amax/quantize implementation is clean, performant, and ready to ship. No other high-severity issues.