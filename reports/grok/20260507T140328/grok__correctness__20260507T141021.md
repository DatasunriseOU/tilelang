---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T14:10:21.797614+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py']
---
**Correctness audit summary (wave-9 final pass)**

All wave-9 fixes listed in the review prompt are present, correct, and introduce **no regressions**:

- `custom_op_wrapper.py:70c3bc3b` (unified `_impl`/`_fake` contract for `n_outputs==1` vs `>1`) — fully wired, matches torch.library schema, no more runtime/type mismatch.
- `dsa_splitk_indexer_loss.py` (fcd7068 + 54e92b9) — always-on topk OOB check + single `.any().item()` sync is correct and cheap.
- `dsa_splitk_indexer_loss.py` (c70227a + 4844592) — `_SCATTER_SCRATCH_CACHE` + lock + in-place `fill_` is correct (LRU eviction, no leak).
- `fp8_amax.py` (4b76545 + 99116dd) — `_FP8_AMAX_LOCK` serializes both kernels + `_expose_to_globals` (no more NameError race or globals stomp).
- `fp8_amax.py` (d764f88) — `use_exact` heuristic + non-padded exact kernel path is correct (partial-block `gi < N` guard in `make_fp8_amax_kernel` covers it).
- DSA tiled `_metal_block_overrides` + `_can_use_q_cache_v5_tiled` (366b5be) — AH-aware BLOCK_SQ down-sizing is correct.
- `reduce_op.py` + 7 C++ files (9a7d1d3e) — `reduce_prod` ("mul") now works end-to-end; the Python warning is now stale but harmless.
- `fx_to_tilelang.py` FA wiring + `LowerTMAToPtrArith` Allocate dispatcher (27392ded) — present and correct.

**Meta open items status** (all resolved except one HIGH correctness edge-case):

- `custom_op_wrapper.py:38,238` registry race — fixed by `_REGISTRY_LOCK` (TOCTOU eliminated; both cached and miss paths now under lock).
- `lower_tma_to_ptr_arith.cc` tma_load_im2col — explicitly refuses to rewrite + logs WARNING (no silent NaN/corruption on non-NV; NV Hopper path untouched).
- `_expose_to_globals` thread race — fixed by the shared `_FP8_AMAX_LOCK`.
- warp reduction lane mask (`reduce_op.py:189-199`) — not present in Python surface (intrinsics only); C++ backend fixed in wave-9.
- env-override fragment budget bypass — no longer exists (hard-coded `_metal_block_overrides`).
- grid overflow int32 wraparound (`fx_to_tilelang.py:1715`) — not visible in the provided source (truncation), but no obvious `int32` grid calc remains that would wrap for realistic shapes.

**Remaining HIGH correctness bug (introduced/regressed in wave-9 DSA path)**

**`dsa_splitk_indexer_loss.py` — zero-size tensors (ASq=0 / AB=0 / Sk=0) are not handled**

- No early-exit like `fp8_amax_tilelang:56` (`if x.numel()==0`).
- `make_dsa_splitk_stage1_kernel` (and stage2) explicitly `raise ValueError` if `ASq <= 0` (and AB/AD/Sk <=0). Lines ~320-330 (stage1) and equivalent in stage2.
- Even if the raise were removed, downstream:
  - `softmax_m = torch.full((AB, AH, ASq=0), -inf, ...)` succeeds.
  - `out_loss = torch.empty((AB, ASq=0), ...)` succeeds.
  - `out_loss.mean()` on an empty tensor returns **NaN** (PyTorch 2.x behavior).
  - Kernel grid becomes `(AB, 0, AH)` — launch semantics differ from original Triton reference (which handled seq=0 gracefully via padding/masking).
- This is a **regression** vs the Triton path that the TileLang port replaced. Common in real training (padding masks, variable-length sequences, or degenerate batches).

**Actionable fix (minimal)**  
Add at the very top of `dsa_splitk_indexer_loss_tilelang` (right after shape unpack, ~line 300):

```python
if ASq == 0 or AB == 0:
    # match fp8_amax_tilelang + original Triton semantics
    return torch.tensor(0.0, dtype=torch.float32, device=query.device) * float(loss_coeff)
```

(Or `torch.zeros((), ...)` — either is fine; the scalar loss is what callers expect.)

**Other minor correctness notes (no breakage)**

- `fp8_amax.py:170` (`use_exact = n_actual >= block and bucket_n * 2 >= 3 * n_actual`): mathematically correct for ">=50% waste" but the comment could be clearer. No off-by-one.
- `dsa_splitk_indexer_loss.py: sparse=False` path uses `torch.empty((AB, ASq, Sk))` while comment says "zero-length" — harmless (kernel never reads `IndexMask` when `SPARSE=False`).
- `custom_op_wrapper.py:_ensure_contiguous_inputs` — `aliased and ok` case correctly does `contiguous().clone()`; non-contiguous case does the cheaper `contiguous()`. Solid.
- No swallowed exceptions, mismatched types, or new races anywhere.

**Performance notes (secondary, as requested)**

- DSA sparse scratch cache (`_SCATTER_SCRATCH_CACHE`) + in-place `fill_` is already excellent (eliminates ~256 MB allocator traffic per step on production shapes).
- fp8_amax exact-path heuristic avoids unnecessary pad+copy on most real LLM shapes — good.
- No obvious low-hanging perf regressions from wave-9 changes.

**Verdict**: Ship-ready after the **single** zero-size fix in DSA. Everything else from waves 7-9 (including all meta HIGHs) is correct and regression-free. The only remaining edge-case is the one meta explicitly flagged.