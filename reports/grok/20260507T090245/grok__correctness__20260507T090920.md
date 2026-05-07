---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T09:09:20.286049+00:00
files: ['/private/tmp/tl_poc_review/poc/triton_frontend/__init__.py', '/private/tmp/tl_poc_review/poc/triton_frontend/op_mapping.py', '/private/tmp/tl_poc_review/poc/triton_frontend/pipeline.py', '/private/tmp/tl_poc_review/poc/triton_frontend/layout.py', '/private/tmp/tl_poc_review/poc/triton_frontend/ptr_analysis.py', '/private/tmp/tl_poc_review/poc/triton_frontend/conformance/__init__.py', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py', '/private/tmp/tl_poc_review/testing/python/triton_frontend/test_conformance_kernels.py']
---
**WAVE-3 FINAL REGRESSION REVIEW: Status = NOT SHIP-READY**

Wave-1 (full 16-op table, PtrAnalysis integration, text/MLIR walkers, sanitizer) + Wave-2 (tile paths via `_ptrstate_*` helpers, `_emit_*_copy`, masked pre-clear, tuple binding, conformance kernels) have been cleanly integrated in the scalar/MVP paths. No regressions to existing behaviour were introduced:

- `_walk_text_ttir` / `_walk_mlir_module` still cover exactly the ops in `OP_TABLE` (text path still raises on unknown; MLIR path still silently skips non-table ops).
- Scalar `tt.load`/`tt.store` (non-tile) still use correct `tir.if_then_else` + `BufferLoad`/`BufferStore` (unchanged).
- `from_ttir` pre-pass + fallback still works when shim is absent.
- Pipeline Tier-1 subset, `map_tt_reduce` (including `reduce_prod` / log-exp fallback), `map_tt_print` sanitizer, and all conformance kernels execute without new crashes or behavioural changes.

**Remaining HIGH-severity correctness bugs (tile + PtrAnalysis paths introduced/exposed in Wave-2):**

1. **Masked tile load is semantically broken** (`attachments/op_mapping.py` – `_emit_load_copy` function, the entire body after the `if other_ssa is not None and mask_ssa is not None:` block)  
   - Pre-fill with `other_expr` (correct intent) is immediately overwritten by unconditional `T.copy(src_slice, frag)`.  
   - Masked-out lanes therefore receive global memory data instead of `other`.  
   - The subsequent `ctx.value_map[result] = (frag, mask_expr)` stash (and `return frag`) is never consumed by any downstream emitter (`map_tt_dot`, `map_tt_reduce`, `map_tt_where`, `map_tt_broadcast`, etc. all do plain `ctx.get(...)` with no tuple unpack).  
   - Scalar fallback path (non-`_ptrstate_is_tile`) is still correct, but the Wave-2 tile path (the whole point of PtrAnalysis) is now incorrect for any masked `tt.load`.  
   - Exact trigger: any Triton kernel using `tt.load(..., mask=..., other=...)` on a multi-element tile.

2. **PtrState lookup is dead code / always falls back to scalar** (`attachments/__init__.py: ~290` in the `if shim_available()` pre-pass block + `attachments/op_mapping.py:map_tt_load` at the `resolved = ctx.get(ptr_ssa) if ptr_ssa in ctx.value_map else None` line, and identically in `map_tt_store` / `_emit_store_copy`)  
   - Pre-pass does `ctx.value_map[state.source] = { "_ptrstate": state, ... }` where `state.source` is a **string** (printed SSA form).  
   - Emitters do `ctx.get(ptr_ssa)` where `ptr_ssa` is the live MLIR `Value` object (or dict in tests).  
   - String key never matches object key → `_ptrstate_is_tile` path is unreachable, `_emit_load_copy` / `_emit_store_copy` never used for real TTIR.  
   - This also affects store region slicing (see #3).

3. **Tile store region slicing is missing / incorrect** (`attachments/op_mapping.py:_emit_store_copy`)  
   - Always does `T.copy(val_expr, dst_buf)` on the *full* buffer (shape from `_ptrstate_sizes_int`).  
   - No `T.region` construction (compare to the partial attempt in `_emit_load_copy`).  
   - Stores therefore write to the wrong global offset/range when the `tt.store` pointer is a non-zero-offset tile.  
   - Combined with #2, the entire tile store path is non-functional.

4. **Minor but HIGH-impact follow-on** (`attachments/op_mapping.py:_emit_load_copy`, the `T.region` call)  
   ```python
   src_slice = T.region(src_buf, "r", *[r[0] for r in region])  # only starts!
   ```
   Even if the lookup were fixed, this would construct an incorrect `BufferRegion` (passes only start indices, not `(start, end)` pairs). Off-by-one / wrong-slice bug.

These four issues are the *only* remaining high-severity correctness problems. Everything else (error handling, None guards, default values, edge cases in scalar path, `make_range`, `program_id`, async/mbarrier/TMA fallbacks, printf sanitizer, pipeline ordering, layout placeholder stance) is solid and matches the RFC exactly. No swallowed exceptions, no new race conditions, no broken null handling, no regressions to MVP/elementwise kernels.

**Recommendation**  
Fix the three `op_mapping.py` functions (`_emit_load_copy`, `_emit_store_copy`, and the `resolved` lookup in `map_tt_load`/`map_tt_store`) so that:
- `value_map` population uses a key that actually matches the MLIR `ptr_ssa` (either change pre-pass to key by object or make emitters do `str(ptr_ssa)` lookup).
- Masked tile loads either (a) pass the mask to `T.copy` (if the API supports it) or (b) apply a post-copy masked select using the stashed `(frag, mask)` tuple, and make *every* consumer that does `ctx.get` unpack the tuple.
- Stores use the same region slicing logic as loads.

Once these are addressed the package is green and ready to ship. No other blockers.