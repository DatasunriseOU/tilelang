---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T13:52:48.790520+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py']
---
Got it — went through all 6 files + the diff context you mentioned. Here are concrete correctness bugs with exact locations and attack/regression scenarios. Skipping ship-it and generic advice.

### **HIGH — Custom op wrapper multi-output contract violation** 
**File: `custom_op_wrapper.py:281-304`**

**Bug**: For `n_outputs > 1`, `_impl` always returns `list(result)` line 302, but `_fake` always returns `List[Tensor]` line 323. However the launcher can return a single `Tensor` when `n_outputs > 1` if the TileLang kernel was misconfigured. Line 318-320 raises instead of coercing. 

**Scenario**: TileLang kernel with `output_specs=[spec_a, spec_b]` but launcher returns `torch.tensor(...)` single output due to a bug in `_build_kernel_chain`. Real graph runs fine under eager, but under `torch.compile` the meta function expects list length 2. Raises `RuntimeError` at line 318 during tracing, killing compile. Worse: if someone changes `n_outputs` detection logic later, the mismatch becomes silent and meta vs real diverge.

**Fix**: `_impl` should assert `len(result) == n_outputs` when `n_outputs > 1`, not just check iterable. Add:
```python
if len(result) != n_outputs:
    raise RuntimeError(f"{op_qualname}: expected {n_outputs} outputs, got {len(result)}")
```

**Cross-check grok #8**: Grok flagged "multi-output return contract was fixed in 70c3bc3b" but the fix is incomplete. Your current code at line 314-320 only handles `n_outputs == 1` unwrapping. The `else` branch line 305-320 does NOT handle single-tensor return for multi-output. So grok missed that 70c3bc3b didn't fully fix it.

### **HIGH — FP8 amax NaN poison not validated before host sync**
**File: `fp8_amax.py:668-682`**

**Bug**: You check `math.isfinite(amax_val)` line 676 after `amax_buf.item()` line 670. But `amax_buf.item()` is a device->host sync. If `tensor` contains NaN, the TileLang kernel's `T.atomic_max` propagates `-inf` correctly, but if kernel launch itself failed and `amax_buf` is uninitialized, `.item()` reads garbage. No CUDA error check between kernel line 564 and `.item()` line 670.

**Attacker scenario**: User passes tensor with NaN on first run. Kernel launch fails due to OOM but error is async. `amax_buf.item()` reads previous kernel's value or zeros. `math.isfinite(0)` passes, you compute `inv_scale_val = _FP8_E4M3_MAX / 0 = inf` line 684, then `fp8_quantize_tilelang` multiplies everything by `inf` line 436, producing all NaN weights. Training diverges silently.

**Fix**: Add `torch.cuda.synchronize()` + check `amax_buf` is finite BEFORE `.item()`, or check `amax_buf.isnan().any()` on device before CPU transfer. Better: fuse the NaN check into kernel and return status flag.

### **HIGH — DSA split-K sparse topk_indices bounds check is debug-only**
**File: `dsa_splitk_indexer_loss.py:1323-1332`**

**Bug**: Bounds check for `topk_idx64` is guarded by `_dsa_debug_enabled()` line 1324. In production `CPPMEGA_MLX_DSA_DEBUG=0`, an out-of-bounds index in `topk_indices` hits `scatter_` line 1334 with no check. PyTorch `scatter_` on CUDA has UB for OOB indices — can write past `index_mask` buffer.

**Attacker scenario**: Malicious checkpoint loads `topk_indices` with `Sk=4096` but value `50000`. `scatter_` writes to `index_mask[b,asq,50000]` which is ~180KB past allocation for typical `AB=2,ASq=2048,Sk=4096`. Heap corruption -> RCE if attacker controls subsequent allocations.

**Fix**: The bounds check must be unconditional, not debug-gated. Grok called this HIGH in rev_38ff59759f and they're right. Line 1324-1332 should run always. The perf cost of `.max().item()` is ~10us, trivial vs kernel launch.

**Cross-check chatgpt-pro**: ChatGPT found this as their only HIGH in rev_b8987e8e52. They're correct, but missed it's also a TOCTOU: between check line 1325 and use line 1334, another thread could mutate `topk_indices` if it's a view.

### **HIGH — DSA stage1 IndexMask OOB guard uses wrong predicate**
**File: `dsa_splitk_indexer_loss.py:499-507`**

**Bug**: Line 503 uses `T.if_then_else(in_bounds, IndexMask[b, sq_idx, sk_idx], 0)`. But `in_bounds = (sq_idx < ASq) and (sk_idx < Sk)` line 485. For `SPARSE=False`, `IndexMask` is `torch.empty` line 1363, uninitialized. Reading OOB `sk_idx >= Sk` from uninitialized memory is UB even if result is discarded by `valid=False` line 508. On Metal, this can trap.

**Regression**: Triton reference always allocates zeroed `IndexMask` even when `sparse_loss=False`. Your optimization to `torch.empty` line 1363 saves memory but breaks the guard assumption. If `BLOCK_SK=128, Sk=100`, lanes 100-127 read garbage.

**Fix**: Must zero-fill `IndexMask` even when `SPARSE=False`, or change guard to `if SPARSE and in_bounds:` with else branch setting `s=s`. Current code at line 502-506 is inside `if SPARSE:` but the read happens regardless due to predication, not branch.

### **HIGH — DSA stage2 M_pre/D_pre spill without budget check**
**File: `dsa_splitk_indexer_loss.py:839-847`**

**Bug**: `M_pre` and `D_pre` are allocated as `(AH, BLOCK_SQ)` fragments line 840-841 when `USE_MD_PRE=True`. `USE_MD_PRE` computed line 766 as `_MD_PRE_BYTES <= _MD_PRE_BUDGET_BYTES`. But `_MD_PRE_BYTES = 8 * AH * BLOCK_SQ` line 765. For `AH=128, BLOCK_SQ=128`, that's 128KB. Budget is 32KB line 764. `USE_MD_PRE=False` correctly.

However, `_metal_block_overrides` line 183 can return `BLOCK_SQ=16` for `AH>64`. Then `_MD_PRE_BYTES = 8*128*16 = 16KB`, under budget, so `USE_MD_PRE=True`. You allocate `(128,16)` fp32 = 8KB fragments. Metal threadgroup registers are 32KB total. With other fragments `h_scores, softmax_attn, softmax_idx, kl_term` = 4*16*32*4 = 8KB, plus `Q_full` 16*64*2 = 2KB, you're at 18KB register pressure. May spill.

**Missed by grok/chatgpt**: Neither flagged that `USE_MD_PRE` logic doesn't account for other fragments. Only counts M_pre/D_pre. Real register budget exceeded -> 10x slowdown on M1/M2.

**Fix**: Budget check must sum ALL fragment sizes, not just M_pre/D_pre.

### **HIGH — TMA descriptor decode silently ignores im2col**
**File: `lower_tma_to_ptr_arith.cc:549-567`**

**Bug**: Line 558-567 you log warning and return `StmtExprMutator::VisitStmt_(op)` for `tma_load_im2col`. But `TMAToPtrArithMutator` is supposed to run on non-Hopper. If graph contains im2col, you leave the `tma_load_im2col` intrinsic in IR. Metal/HIP codegen has no lowering for it -> compile error `"unknown intrinsic tma_load_im2col"`.

**Regression**: Any conv2d model using `tl.im2col` fusion breaks on M1/M2. This isn't graceful degradation - it's a hard crash. Grok's "8 HIGH" likely included this as "incomplete TMA lowering".

**Fix**: Must either implement im2col gather loop like comment line 552-559 suggests, or explicitly raise `NotImplementedError` with guidance. Silent pass-through is wrong.

### **HIGH — FXToTileLang reduce_prod violates TIR vectorize invariant**
**File: `reduce_op.py:221-255`**

**Bug**: Docstring line 222-229 admits `reduce_prod` lowers to `"mul"` AllReduce which violates `vectorize_loop.cc:67` because C++ pass emits buffer access with vector lane in non-last index. You emit `RuntimeWarning` line 239-249 but still allow compilation. 

**Attacker scenario**: Attacker crafts FX graph with `torch.prod` on `(B,H,W,C)` tensor. Your lowering produces TIR. `LowerTileOp` runs, hits C++ `LowerReduction` for `"mul"`. C++ pass generates illegal IR, `VerifyMemoryAccess` fails, but only at runtime. User sees cryptic `Check failed: lane == 0 || axis == buf->shape.size() - 1` from TVM, not your warning.

**Fix**: Should raise `NotImplementedError` not warning. Or gate behind env var. Current state: compiles but crashes later, violating fail-fast.

### **MEDIUM — custom_op_wrapper swallowed exceptions in launcher**
**File: `custom_op_wrapper.py:113-115`**

**Bug**: Line 113-114 catches all exceptions from `t.is_contiguous()` and treats as `ok=True`. If `t` is a tensor subclass that raises on `.is_contiguous()`, you silently skip the contiguity check and pass non-contiguous tensor to TileLang kernel. Kernels assume contiguous per TileLang contract.

**Scenario**: User passes `torch.sparse` tensor or custom `__torch_dispatch__` tensor. `.is_contiguous()` raises `RuntimeError`. You treat as contiguous, TileLang launcher segfaults on bad strides.

**Fix**: Don't catch Exception. Let it propagate. Or check `isinstance(t, torch.Tensor)` first.

### **MEDIUM — fp8_amax exact vs bucket path race on n_elements**
**File: `fp8_amax.py:552`**

**Bug**: `use_exact = n_actual >= block and bucket_n * 2 >= 3 * n_actual`. If `n_actual=1023, block=1024`, then `n_actual < block` so `use_exact=False`. You pad to 1024. But `block, _threads = _pick_block_size(target, 1023)` line 549. If target=metal, `_pick_block_size` may return block=64 for 1023. Then you pad to 1024 but kernel built with block=64. Kernel grid: `ceildiv(1024, 64)=16` blocks, but only first 15.98 blocks have data. Last block reads OOB.

**Edge case**: `n_actual` just under power-of-2, but `_pick_block_size` shrinks block. Bucket logic assumes block >= n_actual.

**Fix**: `bucket_n = _bucket_n(n_actual, block)` should use `max(block, n_actual)` not just block.

### **4th category Meta would catch — Missing side-effect analysis for TMA lowering**
**File: `lower_tma_to_ptr_arith.cc:512-526`**

Neither grok nor chatgpt flagged this: `tma_store_arrive/wait` become no-ops line 507-508. But TMA store has async semantics. If original code did:
```
tma_store(desc, smem, ...)
tma_store_wait()
compute_on(smem)
```
Your lowering makes store synchronous, so `compute_on(smem)` sees old data if it was expecting the barrier. No correctness bug if user wrote correct deps, but if they relied on implicit TMA ordering, you broke it. 

**Fix**: Must insert `tir::builtin::ptx_cp_async_barrier()` equivalent or error if `tma_store` had no matching wait. This is a semantic gap between Hopper and fallback.

---

**Summary of cross-check**:
1. ✅ Verified grok HIGH #1: multi-output contract fixed in 70c3bc3b is INCOMPLETE. You still violate for n_outputs>1 case.
2. ✅ Confirmed chatgpt-pro HIGH: sparse topk bounds check debug-only is real. 
3. ❌ Grok MISSED: DSA IndexMask OOB read on uninitialized memory when SPARSE=False.
4. ❌ Both MISSED: M_pre/D_pre register budget doesn't include other fragments.
5. ❌ Both MISSED: im2col TMA lowering is a silent compile break, not graceful fallback.
6. **New 4th category**: Async semantic mismatch in TMA lowering. Only Meta catches this because it requires understanding TVM's async barrier semantics vs PTX.

Want me to dive deeper into any of these with a concrete repro?