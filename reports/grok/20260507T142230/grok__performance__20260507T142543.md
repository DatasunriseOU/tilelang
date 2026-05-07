---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T14:25:43.626245+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py', '/private/tmp/tl_poc_review/src/tl_templates/cuda/reduce.h']
---
**WAVE-10 FINAL TRIPLE-LLM REVIEW (Grok + Harper + Benjamin consensus)**

All three wave-10 commits are **correct with no new regressions**:

1. **DSA zero-size guard** (`dsa_splitk_indexer_loss.py:580-590`): `if ASq == 0 or AB == 0 or ... return torch.zeros(())`. Prevents `ceildiv(0, BLOCK)` → div-by-zero and kernel launch crashes. Matches the documented scalar-return contract. Clean.

2. **TMA i64** (`lower_tma_to_ptr_arith.cc:320-340` + comment at ~330): `kIdx = DataType::Int(64)` is already wired into every offset/stride/accumulator in `BuildPointerArithCopy` (global_byte_offset, global_elem_offset, smem_elem_offset). Meta's CRITICAL int32-overflow claim was a false positive (wave-3 340439d6 already fixed it). The added comment closes the audit trail.

3. **reduce_prod MulOp contract** (`reduce_op.py:220-260` docstring + `reduce.h:80-100` MulOp comment + wave-10 cf25c0e1): Correctly documents the "inactive warp lanes MUST be 1.0" requirement for the XOR-butterfly in `AllReduce<MulOp>` / `warp_reduce`. The `xfail` test in the tree is the right signal. **However** — this is *documentation-only*. Real lowering-time identity-pad (`MakeInitValue` in `src/op/reduce.cc`) is wave-11. Today this is a **silent correctness regression** (wrong products, not necessarily NaN) on any `dim` length that is not a multiple of warp width (32/64). Not a "NaN attack" vector per se, but real production risk for non-power-of-2 reductions.

**Hot-path performance regressions / concerns introduced or left open by this diff** (prioritized by impact):

- **`dsa_splitk_indexer_loss.py:550-560` (OOB check — HIGH impact regression)**  
  ```python
  if topk_idx64.numel() > 0:
      _oob = ((topk_idx64 < 0) | (topk_idx64 >= Sk)).any()
      if bool(_oob.item()):
  ```
  Always-on (no longer debug-gated) GPU reduction + `.item()` sync **every forward pass** when `sparse_loss=True`. For production DSA shapes (AB=8, ASq=2048, TOPK=8-16) this is 65k–260k elements → ~5-20 µs of GPU→CPU sync + small kernel launch per DSA loss call. In a typical LLM training step (DSA loss in 10-20 layers) this compounds to measurable overhead (1-5% forward time). Wave-9 #4 made it "always-on for security"; the foot-gun is closed but at real hot-path cost. (Previous debug path was the right trade-off.)

- **`dsa_splitk_indexer_loss.py:100-130` (`_get_scatter_scratch`)**  
  Lock + dict lookup + `buf.fill_(float("-inf"))` + `scatter_` **every sparse forward**. The LRU reuse is a huge win vs the old `torch.full` (wave-9 #5), but a full 256 MB HBM write (fill_) + scatter_ per step is still bandwidth-bound. For high-throughput training this is the dominant cost in the sparse path.

- **`dsa_splitk_indexer_loss.py:~720` ( `_dsa_debug_enabled()` + wave-4 debug path)**  
  `os.environ.get(...)` *every forward pass* (string compare). Trivial but unnecessary hot-path work; should be cached at module import time.

- **`fp8_amax.py:480-510` (use_exact heuristic in `fp8_amax_tilelang`)**  
  The heuristic itself is good (avoids 50%+ padding waste), but the else branch still does `torch.zeros(bucket_n) + copy_` on the device for any shape that doesn't hit the exact-path predicate. Combined with `flat = x.reshape(-1).contiguous()` this creates extra HBM traffic + potential allocator pressure on variable-shape paths. The `lru_cache(maxsize=256)` on `_amax_kernel_for(n_actual)` also means JIT thrashing is possible in variable-seq training even with the bucket fallback.

- **`custom_op_wrapper.py:80-130` (`_ensure_contiguous_inputs`)**  
  Runs on **every** custom_op invocation (i.e. every fused region). The Python loop + `is_contiguous()` / `_base` checks + possible `.contiguous().clone()` is a measurable tax when FX graphs are full of views/slices (common in LLM forward). The warn-once cache helps, but this is still a regression vs the pre-fused path that assumed caller contiguity.

- **reduce_prod path (`reduce_op.py:300+` + `reduce.h:80-100`)**  
  As noted above — MulOp butterfly can pull garbage/0 from inactive lanes today. The docstring + xfail test is the right signal, but without the wave-11 lowering fix this is a correctness time-bomb that can force fallback recompute (perf regression).

**Other minor notes (not regressions but worth closing):**

- `fp8_amax.py:450` and `dsa_splitk_indexer_loss.py:700+`: `_pick_block_size` / `_bucket_n` are called multiple times per hot path (once for heuristic, once inside the cached kernel builder). Negligible but easy to hoist.
- `lower_tma_to_ptr_arith.cc` pointer-arith fallback is correct and necessary, but confirm downstream vectorization (`LegalizeVectorizedLoop` / Metal SIMDgroup / HIP wave vectorizer) actually recovers good bandwidth on Ampere/Metal/HIP. The `kEmitOpaque=False` `BufferLoad/Store` path was chosen to preserve pipelining — good call.
- No O(n²), no alloc-in-tight-loops, no large synchronous payloads, no memory growth leaks visible.

**Actionable perf recommendations (priority order):**

1. Make the OOB check in `dsa_splitk_indexer_loss.py:550-560` debug-only again (or fuse the bounds check into the TileLang kernel itself — TileLang can do it without host sync).
2. Cache `_dsa_debug_enabled()` at module level.
3. Expose a `precompile_dsa_kernels` helper analogous to `precompile_amax_kernel` and call it at model init with the known training shapes.
4. In FX-to-TileLang path, guarantee contiguous inputs *before* hitting the custom_op wrapper (avoids the `_ensure_contiguous_inputs` tax).
5. Wave-11: land the `MulOp` identity initialization in `src/op/reduce.cc:MakeInitValue` (remove the xfail).

The wave-10 changes are solid overall (zero-size guard + TMA i64 + documentation are clean wins), but the **always-on OOB sync + scatter fill_** in the DSA sparse path and the **contiguity enforcement** in the custom_op wrapper are the two biggest hot-path regressions introduced. They are fixable without re-opening security holes.

Ready for wave-11. No ship-it yet.