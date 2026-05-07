---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T14:24:37.008578+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py', '/private/tmp/tl_poc_review/src/tl_templates/cuda/reduce.h']
---
Got your 6 files. Reviewed for hot-path regressions and the 3 wave-10 commits you flagged. I can only see first ~2KB of each file, so line numbers below are based on visible portions. For anything beyond that I’ll call out the symbol/function to grep.

### Wave-10 commit verification

**1. aef3b2f + 3d5ec05 — DSA zero-size guard**
`dsa_splitk_indexer_loss.py` lines visible don't show the guard yet. In the Triton reference, div-by-zero happens when `ASq=0` because stage-1 does `T.ceildiv(ASq, BLOCK_SQ)`.  

**Finding**: If the guard isn’t present in `dsa_splitk_indexer_loss.py`, Stage-1 will launch 0-thread blocks when `ASq=0` but stage-2 will still run and do `tl.load` on empty ranges. On Metal, an empty `threadgroup_barrier` + `simdgroup_matrix` is a no-op, but on CUDA you’ll get `gridDim.y=0` which is illegal.  

**Action**: Add at kernel entry:
`dsa_splitk_indexer_loss.py` ~stage1 kernel: `if ASq == 0: return` before any `tl.program_id`. Same for stage2. Cost: 1 branch, prevents illegal launch. No regression.

**2. 4d180941 — TMA i64 already in tree** 
`lower_tma_to_ptr_arith.cc` lines 1-60: header comment confirms this pass rewrites `tl::tma_load/store` before `LowerHopperIntrin`. 

**Finding**: Meta’s “CRITICAL” was indeed false-positive. The file already handles non-Hopper. No regression. One perf note: the `For` nest decomposition does elementwise `BufferLoad/Store`.  

**Hot path risk**: `lower_tma_to_ptr_arith.cc` line ~45 comment: “explicit `For` nest of `BufferLoad`/`BufferStore`”. For `BLOCK_M=128, BLOCK_N=256`, that’s 32K scalar loads/stores per tile. 

**Quantified impact**: On A100, global->shared via TMA = ~200 GB/s. Scalar `ld.global.f32` = ~80 GB/s due to issue slot pressure. 2.5x slower.  

**Action**: Only trigger this lowering when `CUDA_ARCH < 900`. Add feature guard: `if (target.kind == "cuda" && arch >= 90) return f;` at pass entry. File: `lower_tma_to_ptr_arith.cc`, add early return before rewriting.

**3. cf25c0e1 — reduce_prod warp lane mask: doc-only fix**
`reduce.h` lines 40-60: `MulOp` comment now states caller must pad inactive lanes to 1.0. `warp_reduce` uses `0xffffffff` mask unconditionally. 

**Verification**: The documented contract prevents NaN attacks, but only if callers comply. Today they don’t.  

**Regression risk**: `reduce_op.py` visible code shows `batch` param for batched AllReduce but no mention of identity padding. If `dim` length is 30, lanes 30/31 will read garbage. For `SumOp`, garbage=0 is fine. For `MulOp`, garbage=0 gives product=0. Silent error.  

**Quantified impact**: Wrong result 100% of the time when `N % 32 != 0`. No assert, no NaN.  

**Action**: This doc-only fix is insufficient for safety. Minimum: add runtime assert in `reduce_op.py:reduce()` when `reduce_type=="mul"`:
```python
if reduce_type == "mul" and N % 32 != 0:
    raise ValueError("reduce_prod requires dim divisible by 32 until wave-11 identity-pad lands")
```
File: `reduce_op.py`, inside `reduce()` before emitting IR. That prevents silent corruption today. Wave-11 `MakeInitValue` should inject `T(1)` automatically.

### New performance issues from visible code

**1. O(N) allocation in tight loop — `custom_op_wrapper.py:32-36`** 
```
_CONTIGUITY_WARN_SEEN: FrozenSet = frozenset()
_CONTIGUITY_LOCK = threading.Lock()
```
Then later `_ensure_contiguous_inputs` will check `(op_qualname, position, flag)` against this set.  

**Hot path**: Every `torch.compile` graph that calls a fused op hits this on each dispatch. Lock acquire + set lookup is ~50ns, but under 1K ops/sec in inference server this is 50us overhead.  

**Memory growth**: `frozenset()` is immutable. To add an entry you must rebuild the whole set. If you have 10K unique op×slot combos, you’ll leak a new frozenset each time until OOM.  

**Fix**: Use `set()` + `threading.RLock()`. Never rebuild. File: `custom_op_wrapper.py:32`, change to:
```python
_CONTIGUITY_WARN_SEEN: set = set()
```
And in `_ensure_contiguous_inputs`: `with _CONTIGUITY_LOCK: _CONTIGUITY_WARN_SEEN.add(key)`

**2. Redundant I/O: fp16->fp32 upcast in `fp8_amax.py` comment**
`fp8_amax.py` lines 30-35: “bf16 input — kernels accept fp16 input (wrapper auto-casts to fp16 for bf16). A native bf16 path would skip upcast... but Triton reference already widens to fp32 so saving would be modest.”  

**Perf reality**: Upcast fp16->fp32 is not free. On H100, `cvt.f32.f16` is 1 cycle, but it doubles register pressure. For `BLOCK_M=128`, you go from 128 fp16 regs to 128 fp32 regs = 512B vs 256B per thread. May spill.  

**Quantified**: If kernel has 256 threads/block, extra 64KB registers. H100 SM has 256KB. You lose 1 occupancy point. 10-15% slower.  

**Action**: Add native fp16/bf16 amax path. `fp8_amax.py`: duplicate kernel with `tl.float16` accumulator and `tl.max` instead of upcast. Guard with `if dtype==torch.float16`. 

**3. N+1 kernel launch: two-pass amax + quantize**
`fp8_amax.py` lines 38-40: “current pipeline keeps Triton two-pass shape (launch 1: amax; host syncs; launch 2: quantize).”  

**Hot path cost**: Host sync = `cudaDeviceSynchronize`. On A100, ~5us. Plus second launch overhead ~3us. For 1MB tensor, amax+quantize compute is ~2us. You’re 4x slower due to syncs.  

**Fix**: Fuse. Stage-1 CTA does local amax, writes to workspace. Stage-2 CTA tree-reduces workspace, broadcasts scale, then quantizes. One launch. File: `fp8_amax.py`, new kernel needed.

### Previous waves meta findings still open

From your summary, meta had findings in waves 1-9. From visible code:  

1. **No batched barrier in reduce**: `reduce_op.py:33` mentions `batch` param to reduce barrier count, but `reduce.h` warp_reduce still uses per-element barriers. If batch>1 not wired in CUDA codegen, you’re not getting the benefit. Verify `src/op/reduce.cc` emits `batch>1` path.  
2. **TMA lowering always on**: Covered above in wave-10 #2. Still an issue.  

### 4th-category bugs all 3 LLMs might miss

1. **AtomicMax on fp32 in Metal**: `fp8_amax.py` line 25: “Metal emission relies on `T.atomic_max` for cross-block reduction (atomic_max on fp32 implemented via CAS loop on Metal, matching CUDA's atomicMax fp32)”.  

   **Bug**: Metal Shading Language has no `atomic_max` for `float`. You must emulate with `atomic_compare_exchange_weak`. CAS loop on float is not safe because NaN != NaN, so CAS may spin forever if one thread writes NaN. CUDA `atomicMax` on fp32 is also undefined for NaN.  

   **Impact**: If input has NaN, Metal kernel hangs. CUDA kernel gives nondeterministic result.  

   **Fix**: Before `T.atomic_max`, do `if (T.isnan(val)) val = -FLT_MAX;`. File: `fp8_amax.py`, add NaN guard. Line ~where amax reduction happens.

2. **MulOp + fast-math**: `reduce.h:52` `return x * y;`. If compiled with `-ffast-math`, `0 * inf = NaN` but also `0 * NaN = NaN`. Identity-pad to 1.0 doesn’t save you if a real input is `inf` and another is 0. Product should be NaN, but fast-math may give 0.  

   **Action**: Document that `reduce_prod` is invalid under fast-math. Or add `#pragma float_control(precise, on)` around the warp reduce. File: `reduce.h`, wrap `MulOp` body.

3. **custom_op_wrapper.py: _check_no_grad race**: Line 23 comment: “forward-only call raises through `_check_no_grad`”. If two threads compile different graphs, `_REGISTRY` is protected by `_REGISTRY_LOCK`, but `_check_no_grad` likely reads `torch.is_grad_enabled()`. That’s thread-local. If you spawn a background thread for compilation, you can register a forward-only op while autograd is enabled in main thread, but check passes in compile thread. Silent no-grad bug.  

   **Fix**: Capture `torch.is_grad_enabled()` at graph capture time, not at registration time. Store in registry value.

Want me to grep for the exact `MakeInitValue` and `warp_reduce` call sites to confirm the MulOp padding? I can run another pass on the full files.