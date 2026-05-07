---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T01:41:22.625241+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/Volumes/external/sources/cppmega/cppmega/megatron/dsa_splitk_indexer_loss.py', '/Volumes/external/sources/cppmega/tests/test_dsa_splitk_tilelang.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/sparse_mla.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py']
---
**Code review: `dsa_splitk_indexer_loss.py` (TileLang DSA split-K port) + wrapper / test / sparse_mla reference**

I reviewed the full diff against the original Triton kernels (`attachments/dsa_splitk_indexer_loss.py:lines 55-193` for stage1/2), the `sparse_mla_path_c.py` style guide, and TileLang primitives (`tilelang/language/__init__.py` for `T.Pipelined`, `T.gemm`, `T.alloc_fragment/shared`, `T.reduce_max/sum`, `T.Kernel`). Focus is **performance regressions/hot-path issues** (bandwidth, memory footprint, redundant work, allocations) on both CUDA and especially Metal (32 KB threadgroup limit). Numerical parity tests pass on small shapes, but several production-scale bugs/regressions are present.

### 1. Critical correctness bug: OOB indexing on `IndexMask` / `IndexScores` in boundary tiles (affects sparse_loss=True)
**File**: `dsa_splitk_indexer_loss.py` (TileLang port)

- Stage 1 mask application (`make_dsa_splitk_stage1_kernel:lines 280-300`):
  ```python
  for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
      sq_idx = sq_block_id * BLOCK_SQ + i
      sk_idx = sk_tile * BLOCK_SK + j
      valid = (sq_idx < ASq) and (sk_idx < Sk) and (sq_idx >= sk_idx)
      s = scores_f[i, j] * SCALE
      if SPARSE:                                      # ← unconditional
          s = s + IndexMask[b, sq_idx, sk_idx]        # ← OOB when sk_idx >= Sk or sq_idx >= ASq
      if valid:
          scores_f[i, j] = s
      else:
          scores_f[i, j] = -inf
  ```
  Same pattern in the `h == 0` index_scores path (`lines 310-330`) **and** in stage 2 (`make_dsa_splitk_stage2_kernel:lines 520-530` for attention mask, `lines 620-630` for softmax_idx).

- Original Triton guards every load: `tl.load(..., mask=(sq_valid[:,None] & sk_valid[None,:]), other=-inf)` (`dsa_splitk_indexer_loss.py:lines ~130, ~260`).
- TileLang direct indexing on last `sk_tile` / last `sq_block_id` reads beyond tensor bounds → garbage, NaNs, or crash.
- **Impact**: Silent correctness failure on any shape where `ASq % BLOCK_SQ != 0` or `Sk % BLOCK_SK != 0` (common). Tests use tiny shapes (`ASq=64/128`, `Sk=128/256`) where it luckily fits, hiding the bug.

**Fix (actionable)**: Wrap the sparse add with `if (sq_idx < ASq and sk_idx < Sk): s += IndexMask[...] else: pass` (or compute masked value). Matches sparse_mla_path_c.py post-processing pattern for bounds checks.

### 2. Major Metal memory regression: fragment buffers exceed 32 KB threadgroup limit
**File**: `dsa_splitk_indexer_loss.py`

- `_metal_block_overrides` only halves shared `Q_s`/`K_s` (`lines 90-110`, docstring `lines 70-100` claims "16 KB").
- Forgets `T.alloc_fragment` buffers (all fp32):
  - Stage 1: `scores_f` + `idx_scores_f` = 2 × (64×64×4) = **32 KB** + shared **~16 KB** = **~48 KB**.
  - Stage 2: `h_scores` + `softmax_attn` + `softmax_idx` + `kl_term` = **64 KB** + shared = **~80 KB**.
- Metal threadgroup budget = 32 KB. Result: register spilling → global mem fallback, massive slowdown, or compile/runtime failure on M-series.
- CUDA (96 KB+ shared) survives but still high reg pressure with `BLOCK_SQ/BLOCK_SK=128`.

**Impact**: The exact target this PoC was written for (Metal path) is broken or 2-5× slower than expected. Sparse-MLA Path C (`sparse_mla_path_c.py:lines ~800-900` lowering + `_postprocess_lowered_msl`) aggressively reuses buffers and removes checks to stay under limit.

**Fix**: 
- Tune Metal overrides further (`BLOCK_SQ=32`, `BLOCK_SK=32`, `BLOCK_D=16` → fragments ~4 KB each).
- Reuse fragments across phases (e.g. reuse `scores_f` for `kl_term`).
- Or port the MSL post-processing transforms from `sparse_mla_path_c.py` (`_remove_redundant_*_checks`, `_canonicalize_fwd_hot_loops` etc.) once `lower_tilelang_to_msl_inline` is exposed for this kernel.

### 3. Performance regression: no causal SK trimming → ~1.5-2× extra GEMM/reduce work
**Files**: `dsa_splitk_indexer_loss.py:lines 260 (stage1), 410 (stage2)` + original Triton `dsa_splitk_indexer_loss.py:lines ~120, ~250`

- `for sk_tile in T.Pipelined(SK_TILES, ...)` always full `Sk` (docstring acknowledges "pessimises CUDA perf marginally" `lines 140-150`).
- Triton: `causal_sk = tl.minimum(tl.min(sq) + 1, Sk)` then `tl.range(0, causal_sk, BLOCK_SK)`.
- For `ASq ≈ Sk`, TileLang processes full rectangle while Triton does upper triangle → ~2× GEMMs + reduces + exp/log in both stages.

**Impact**: Significant on production seq lengths (4k-32k). Early `sq_block_id` waste is extreme (first block does 32× more work with `BLOCK_SQ=128`). Metal suffers doubly because of smaller blocks + memory pressure.

**Fix**: Compute `causal_sk_tiles = ceil_div(min(sq_block_max + 1, Sk), BLOCK_SK)` per sq_block (use `T.reduce_min` on `sq` or simple arithmetic) and limit the `Pipelined` range. TileLang supports runtime bounds in `T.serial`/`T.Pipelined` (see `tilelang/language/__init__.py`).

### 4. Hot-path memory allocation / zeroing in non-sparse case (common path)
**File**: `dsa_splitk_indexer_loss_tilelang:lines 780-795`

```python
if sparse_loss:
    index_mask = full(-inf).scatter_(...)
else:
    index_mask = torch.zeros((AB, ASq, Sk), ...)  # ← full size!!
```

- Triton uses `empty((0,))` + stride hacks when `SPARSE_LOSS=False`.
- `SPARSE` is constexpr (dead-codes the `if SPARSE` branches), so the tensor is never read when `False`.
- Still allocates **AB×ASq×Sk×4 bytes** (e.g. 0.5 GB+ for typical DSA shapes) + `contiguous()` + zeroing every call.

**Impact**: Unnecessary peak memory + alloc/zero time in every forward pass. Bad for training throughput.

**Fix**: Use `torch.empty((AB, ASq, Sk), ...)` (uninitialized) when `!sparse_loss`. Or make PrimFunc signature conditional on `SPARSE` (harder but cleaner, like sparse_mla).

### 5. Other hot-path / bandwidth concerns (smaller but cumulative)
- **Redundant Q loads**: Q tile reloaded `SK_TILES` times per `sq_block/h` (stage1 `lines 290-300`, stage2 per-head `lines 450-470`). Same as original Triton, but still wasteful (Q independent of sk). Hoistable to before `sk_tile` loop if shared memory budget allows after the fragment fix above.
- **Stage 2 m_h/d_h reloads**: Reloaded `SK_TILES × AH` times (`lines 430-440`). Hoist per-head outside `sk_tile` (they are constant).
- **Globals() hack**: `make_dsa_splitk_*_kernel:lines ~210-230` (and stage2) does `g = globals(); g.update(...)`. Fragile, unnecessary (params are already in scope), and risks stale values across lru_cache calls. Sparse-MLA style avoids this.
- **Wrapper contiguous/to calls** (`lines 610-620`): Defensive but can copy large tensors every call. Add `if not x.is_contiguous():` guards like other cppmega paths.
- **No obvious O(n²) Python loops or blocking calls** in kernels—good. lru_cache on kernels is solid.

### Summary & priority order (performance-first)
1. **Fix OOB indexing** (correctness blocker).
2. **Fix Metal fragment memory overrun** (makes Metal path unusable today).
3. **Add causal SK trimming** (biggest compute win).
4. **Eliminate full `index_mask` alloc in non-sparse case**.
5. Hoist Q / m_h/d_h loads + smaller Metal blocks.

The port is structurally excellent (single-source, `T.gemm` + `T.Pipelined` + online-softmax exactly as spec'd, matches sparse_mla style, nice lru_cache wrapper). Once the above are fixed it will be production-ready and faster than Triton on Metal (no materialised scores tensor). The redundant-Q and full-SK issues were inherited from the Triton reference, but the fragment size and OOB are new regressions introduced by the TileLang translation.

Let me know if you want patched code snippets for the OOB + memory fixes.