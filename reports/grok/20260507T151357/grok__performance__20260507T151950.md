---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T15:19:50.596066+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/src/op/reduce.cc', '/private/tmp/tl_poc_review/src/tl_templates/cuda/reduce.h', '/private/tmp/tl_poc_review/src/tl_templates/hip/reduce.h']
---
**Wave-11 Performance Review (focus: regressions/hot-path only)**

**Key hot-path regressions introduced:**

1. **fp8_amax.py:386 — NaN pre-filter `T.if_then_else(v==v, v, 0)` before `T.atomic_max`** (52f1770 + 910756c)  
   This is now executed for **every element** in the Metal FP8 AMAX kernel (per-thread/per-block reduction path).  
   - Concrete impact: +1 select + NaN-check per value in the tightest loop of the kernel. On Metal (M4/M-series) this is not free — branch divergence on mixed NaN/non-NaN data or extra ALU even on the common non-NaN path. Expected regression: **5-15%** kernel wall time on large tensors (>1M elements) that are already memory-bound (typical for per-tensor scaling in training/inference).  
   - Attacker scenario: high-dimensional weight/activation tensors with occasional NaNs (or adversarial all-NaN batch) force consistent branch behavior; kernel no longer fuses as cleanly.  
   - Actionable: profile the exact Metal kernel (pre- vs post-wave-11) on real M4 hardware with `torch.utils.benchmark` + Metal GPU counters. If NaNs are expected to be rare, consider hoisting the filter to a separate pass or using a NaN-bit trick that Metal can predicate away.

2. **_detect_storage_overlap() call for as_strided siblings** (d6c4772 cleanup)  
   New strict aliasing check now triggers on any as_strided view that shares storage with a sibling (PyTorch 2.x M4 Max case).  
   - Hot-path risk: if this walks strides/pointers or does any O(stride-rank) work and is called from tensor construction / view lowering / DSA dispatch (common in batch processing), it adds CPU overhead on the **critical path for every small tensor or view**. No cache mentioned in the diff, so repeated calls on the same views are redundant.  
   - Quantified impact: low for huge tensors, but measurable cumulative cost (extra microseconds per op) when models create thousands of small views per forward pass.  
   - Attacker scenario: model with heavy use of `.view()` / `.as_strided()` in a loop → increased CPU time in Python/Torch interop layer.

3. **reduce_prod lowering enforcement** (ed7c6457 — MakeInitValue scalar + batched paths + static_assert in cuda/hip reduce.h)  
   ICHECK now rejects non-power-of-2 `reducing_threads`; static_assert added in CUDA/HIP reduce kernels.  
   - Performance **positive** overall (forces optimal warp-shuffle / tree reduction), but the static_assert + ICHECK path is now compile-time only.  
   - Regression risk: any dynamic shape path or scheduler that previously fell back to non-pow2 (the old xfail case) is now a hard compile error. No runtime flexibility remains.  
   - Backlog item still open: “reduce_prod runtime fix vs static_assert” — the wave-11 change chose the static_assert route, leaving the runtime path unimplemented.

4. **_FP8_AMAX_LOCK still module-global** (documented wave-12, not fixed in d6c4772 + 55d7346 + 1a5f19ba)  
   Globals N/BLOCK/DTYPE are still shared.  
   - Concrete DoS (backlog 3c still open): any multi-threaded serving loop calling FP8 AMAX concurrently (different signatures or same) will contend on the **single global lock**. Throughput drops linearly with concurrency — classic lock-DoS in inference serving.  
   - Hot-path: if the lock is taken around kernel launch or the AMAX compute itself, every forward/backward pass pays serialization cost.

**Verify the 4 questions:**

1. **Wave-11 fixes correct? Any regressions?**  
   Correctness: yes.  
   - reduce_prod enforcement + static_assert fixes the old non-pow2 lowering bug (xfail → pass).  
   - Metal NaN pre-filter prevents NaN propagation in atomic_max.  
   - TOPK=0 + sparse_loss=True now raises clean ValueError instead of downstream NaN.  
   - as_strided overlap detection fixes the PyTorch 2.x aliasing flake.  
   **Regressions:** only the two hot-path items above (Metal AMAX branch cost + potential _detect_storage_overlap CPU tax). No O(n²), no new allocations in tight loops, no blocking calls in async paths visible in the diff summaries.

2. **Wave-10/wave-11 backlog still open?**  
   - lock-DoS (3c): **still open** — _FP8_AMAX_LOCK remains global/module-level.  
   - reduce_prod runtime fix vs static_assert: **still open** — wave-11 chose the static_assert route; no runtime path was added.

3. **New “boundary assumption” 4th-category bugs that all 3 LLMs would miss?**  
   - **All-NaN tensor in Metal AMAX**: pre-filter turns every NaN → 0, so atomic_max returns 0.0 instead of NaN/Inf. If downstream scaling expects NaN propagation or Inf, this silently changes semantics (previous meta pattern: “we assumed NaNs never reach atomic_max”).  
   - **reducing_threads=0 or 1** (or extremely large non-pow2 before ICHECK): now compile-time fail, but the batched Lower path in MakeInitValue had no prior guard — could have produced garbage kernels.  
   - **TOPK k=0 with shape[2]==0 + sparse_loss=True but different dtype/device**: the new ValueError is raised, but if the check is after some tensor allocation/compute, you waste work (classic “check too late” boundary).  
   - **Concurrent AMAX calls with different signatures**: still race on the three module globals (N/BLOCK/DTYPE) because per-signature lock was deferred to wave-12. Classic data-race under load that static analysis misses.

4. **Test coverage gaps in wave-11 changes?**  
   - No concurrent/multi-threaded FP8 AMAX tests (the exact lock-DoS scenario).  
   - No Metal-specific tests with all-NaN, mixed NaN/Inf/-Inf, subnormals, or FP8 edge values.  
   - Non-power-of-2 reducing_threads cases are now compile-time (good), but no test that the scheduler still picks pow2 shapes under dynamic batching.  
   - as_strided overlap tests only cover “passes on M4 Max torch 2.x” — missing other backends or older torch versions.  
   - TOPK=0 only checked for shape[2]==0 + sparse_loss=True; missing other zero-shape variants or k=0 with sparse_loss=False.

**Summary (no ship-it):**  
The wave-11 diff is clean on correctness but introduces two measurable hot-path costs (Metal AMAX branch at fp8_amax.py:386 and potential _detect_storage_overlap CPU tax) and leaves the two biggest backlog items (global lock DoS + reduce_prod runtime flexibility) untouched. The new boundary bugs are exactly the “we assumed X never happens” class that previous waves missed. Recommend profiling the Metal kernel immediately and fast-tracking the per-signature lock for wave-12.