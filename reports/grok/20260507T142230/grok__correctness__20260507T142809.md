---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T14:28:09.776078+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py', '/private/tmp/tl_poc_review/src/tl_templates/cuda/reduce.h']
---
**Correctness review (WAVE-10 final triple-LLM sign-off)**

I audited the six attached files for bugs introduced or exposed by the Wave-10 diff (DSA zero-size guard + TMA i64 clarification + reduce_prod contract documentation). Focus was strictly on the requested categories: off-by-one, error handling, swallowed exceptions, type mismatches, races, defaults, None handling, edge cases, and regressions vs. pre-Wave-9/10 behavior.

### 1. Wave-10 fixes verified correct, no regressions

- **DSA zero-size guard** (`dsa_splitk_indexer_loss.py: ~lines 480-490`, right after shape unpack, before any `_stage1_kernel_for` / `_get_scatter_scratch` / kernel launch):  
  `if ASq == 0 or AB == 0 or AH == 0 or AD == 0 or Sk == 0: return torch.zeros((), ...)`  
  Correct. Prevents the exact div-by-zero that was the consensus HIGH from grok+meta. Early return happens *before* any `T.ceildiv`, `_block_constants_for_target`, or buffer allocation. Non-zero shapes unchanged. Matches the documented 0-d scalar contract. No regression.

- **TMA i64 offsets** (`lower_tma_to_ptr_arith.cc: ~lines 280-300` in `BuildPointerArithCopy`, plus the big comment block referencing wave-3 commit 340439d6):  
  `const DataType kIdx = DataType::Int(64);` used for *every* coord/stride/offset/accumulator. Meta’s “CRITICAL int32-overflow OOB” was already fixed pre-Wave-10 (false positive). The new comment correctly documents this. No regression; non-NV targets remain safe.

- **reduce_prod contract** (`reduce_op.py: reduce_prod docstring` + `reduce.h: MulOp comment block`):  
  Documentation-only change (Wave-10 #3). Correctly states the XOR-butterfly + full-warp-mask requirement and the identity-pad-to-1.0 rule. The xfail test for non-warp-multiple `N` is also present. No behavioral regression.

All three Wave-10 items land cleanly.

### 2. reduce_prod contract is *not* sufficient today (still a silent correctness foot-gun)

The documentation is accurate but the high-level API does **not** enforce the contract, and the C++ lowering does **not** yet do the identity-pad (deferred to wave-11 `src/op/reduce.cc:MakeInitValue`).

- `reduce_op.py: reduce_prod` (and the `reduce` macro that it calls) does `copy(buffer, red_frag_in)` with no tail padding.
- `reduce.h: AllReduce<MulOp>` / `warp_reduce` (lines ~180-220) still pulls uninitialized register values via `shfl_xor_sync(0xffffffff, ...)` for inactive lanes when `dim % warp != 0`.
- Result: silent wrong product (or NaN if uninit lane happens to be NaN) on any reduce dimension not divisible by 32 (CUDA) / 32-64 (HIP). This was already present pre-Wave-10 but the new documentation makes the gap explicit. The “silent NaN attack” vector is **not** closed yet.

### 3. New correctness bug introduced in this diff (missed by previous waves)

**Data race on shared scatter scratch buffer** (`dsa_splitk_indexer_loss.py`)

- `_get_scatter_scratch` (the Wave-9 #5 perf change):  
  ```python
  with _SCATTER_SCRATCH_LOCK:
      ... lookup / full / fill_(-inf) / evict ...
      return buf   # lock released here
  ```
- Caller (`dsa_splitk_indexer_loss_tilelang`, sparse_loss branch, right after the OOB check):  
  `index_mask = _get_scatter_scratch(...).scatter_(-1, topk_idx64, 0.0)`

The `scatter_` (and any subsequent kernel use of `index_mask`) happens **outside** the lock. Any two concurrent calls to `dsa_splitk_indexer_loss_tilelang(..., sparse_loss=True)` for the same shape can have one thread’s `fill_(-inf)` + `scatter_` overlap with another thread’s `fill_` or `scatter_`. This corrupts the `-inf` mask → wrong softmax statistics → wrong KL loss (or NaNs).

This is a clear regression from the pre-reuse `torch.full(..., -inf)` path (which was per-call and therefore thread-safe). The `_SCATTER_SCRATCH_LOCK` only protects the *cache metadata*, not the mutable tensor it returns. Even single-threaded async/stream usage is now racy.

### 4. Other correctness issues (none critical, all pre-existing or minor)

- `dsa_splitk_indexer_loss.py: _dsa_debug_enabled` path still does `.all().item()` (GPU↔CPU sync) inside the hot forward path when the env var is on. Not a regression, but the comment claims “zero per-step overhead in production” — the debug path is still expensive.
- `fp8_amax.py: fp8_amax_tilelang` exact-path heuristic (`use_exact = n_actual >= block and bucket_n * 2 >= 3 * n_actual`) is correct and avoids unnecessary padding, but the comment “Wave-9 #3” still describes the old pad+bucket behavior. Minor documentation staleness, no functional bug.
- `custom_op_wrapper.py: _ensure_contiguous_inputs` correctly handles aliased views (`t._base is not None`) but the warning cache (`_CONTIGUITY_WARN_SEEN`) uses a global `frozenset` + lock only on update. The read path (`key not in new_seen`) is lock-free; theoretically racy under extreme contention, but harmless in practice (duplicate warning at worst).
- No swallowed exceptions, no None/None-handling regressions, no off-by-one in the new zero-size guard or TMA pointer math, no type mismatches in the new Metal block overrides or batch-reduce paths.

### Performance notes (brief, since correctness was the focus)

- The DSA scratch reuse is a big win when shapes are stable (intended), but the race makes it unsafe. A cheap fix that preserves most of the perf is to do `buf = _get_scatter_scratch(...).clone()` then `buf.scatter_(...)` (still far cheaper than `torch.full` every time).
- `reduce_prod` MulOp path is now documented but still pays the full AllReduce cost on inactive lanes; wave-11 init-to-1.0 will also improve perf (fewer garbage shfls).

**Summary**: Wave-10 DSA + TMA fixes are solid. The only new correctness bug introduced by the diff is the DSA scratch race (dsa_splitk_indexer_loss.py). The reduce_prod contract documentation is helpful but does not yet close the silent-wrong-results vector. Everything else is clean.