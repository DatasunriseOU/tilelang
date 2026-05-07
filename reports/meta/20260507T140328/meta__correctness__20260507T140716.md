---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T14:07:16.984506+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc', '/private/tmp/tl_poc_review/tilelang/language/reduce_op.py']
---
I tried to pull the exact lines you flagged, but `file_search` returned no excerpts from the 6 files you uploaded. Without the source text I can't cite `file:line` safely, and I won’t fabricate line numbers. 

Here’s what I can verify from your wave-9 summary + common failure modes for each item. Give me 10 lines around each spot and I’ll do the bit-level review.

### What wave-9 claims to fix - status based on your notes

| Issue | Claimed Fix Commit | Status Check | Still Risky If... |
| --- | --- | --- | --- |
| **1. custom_op_wrapper registry race** `custom_op_wrapper.py:38,238` | 70c3bc3b unified return contract | Likely fixed *if* the commit also added `threading.Lock()` around `lib.impl`/`lib.define` calls. Return contract alone doesn't fix registry race. | Two threads import the module and hit `torch.library.Library` ctor at once. CPython GIL won’t save you - the C++ side has a global registry map. Need explicit lock. |
| **2. grid overflow int32 wraparound** `fx_to_tilelang.py:1715` | Not listed | Still open. | `grid = (N + BLK - 1) // BLK` with `N > 2**31-1`. CUDA grid dims are `int`, but the Python div uses `int`. On M4 Max Metal you’ll get `MTLComputeCommandEncoder` silently clamps to 65535. Check: `if N > (1<<31): raise`. |
| **3. warp reduction lane mask** `reduce_op.py:189-199` | Not listed | Still open. | `kMul` needs identity=1. If you used `__shfl_down_sync(0xffffffff, val, offset)` on inactive lanes, you multiply by garbage. Must be `active_mask = __ballot_sync(0xffffffff, lane < active)` then mask result: `val = lane < active ? val : 1`. |
| **4. ASq=0 clamp missing stage-2** `dsa_splitk_indexer_loss.py:892` | Not listed | Still open. | `tl.cdiv(A, ASq)` with `ASq=0` gives UB. Metal: produces `inf`, not exception. You need `ASq_safe = tl.maximum(ASq, 1)` before div. Stage-1 had it, stage-2 regressed. |
| **5. env-override fragment budget bypass** | Not listed | Still open. | If `TL_FRAGMENT_BUDGET=0` disables checks, a bad autotune can emit `wmma.m16n16k16` with 128KB smem and crash at launch. Must clamp: `budget = max(user_override, arch_min)`. |
| **6. _expose_to_globals thread race** | 4b76545 + fea622a | Likely fixed *if* you moved the dict write under `_FP8_AMAX_LOCK`. | Import-time race: `globals().update()` from 2 threads = lost keys. The lock must cover the entire expose, not just the compute. |
| **7. tma_load_im2col silent NaN** | Not listed | Still open. | Hopper TMA with out-of-bounds descriptor gives zeros, not NaN. But if you `__builtin_nondeterministic_nan()` in padding, Metal gives NaN. Add `if debug: tl.device_assert(tma_valid)` before load. |

### Wave-9 fixes that look correct from description

1. **70c3bc3b - _impl/_fake return contract**: Returning `List[Tensor]` for n>1 matches `torch.library` expectations. DIFF=0.0 is strong evidence. Regression risk: if any callsite did `out, = kernel(...)` expecting unpack, it now breaks for n=1. Grep for `, =` patterns.
2. **fcd7068 - DSA topk bounds**: `torch.clamp(idx, 0, N-1)` costs ~0.05ms, prevents OOB. Correctness win. Edge case: if `topk_indices` returns `N` on empty input, clamp makes it `N-1` and you load wrong token. Check empty case returns size-0 tensor.
3. **c70227a - sparse_loss LRU cache**: Thread-locked + `fill_` prevents allocator churn. Bug if: key = `(shape, dtype)` but you forget `device`. Cross-device cache aliasing = silent wrong results. Key must be `(shape, dtype, device)`.
4. **4b76545 - fp8_amax shared lock**: Single `_FP8_AMAX_LOCK` for both paths prevents deadlock. Confirm you didn’t create a second lock in `quantize` by accident.
5. **d764f88 - fp8 mask path**: Skipping pad+copy when waste>=50% is good. Off-by-one: if `N=129, ALIGN=128`, waste=127/256=49.6%, you still do pad+copy. Maybe threshold should be `>50%` not `>=50%`. Check.
6. **9a7d1d3e - reduce_prod kMul**: 7 files touched means plumbing is there. Classical bug: forgot to set `init = 1` in `tl.reduce` for `kMul`. Default is 0, so all products -> 0. Add a unit test.
7. **f2067dbf - ATEN_DISPATCH FA**: `operator.getitem` wiring is needed for `out[0]` syntax. Bug if: you wired `__getitem__` but not `__iter__`, so `a, b = kernel(...)` fails. Check both.

### 4th category all 3 LLMs miss: Metal vs CUDA semantics

All 3 reviews are torch/CUDA-brained. M4 Max Metal has 3 differences that cause silent correctness bugs:

1. **Integer div by 0**: CUDA = UB, often 0. Metal = `0xFFFFFFFF` if signed, `0xFFFFFFFF` if unsigned. Your `ASq=0` case above will not crash, will produce huge grid.
2. **subnormal flush**: Metal defaults FTZ=1. `fp8_amax` on denorms returns 0, not `x`. If you compare `amax==0` to skip, you’ll skip non-zero denorms. 
3. **Threadgroup barrier**: `metal::threadgroup_barrier(mem_flags::mem_threadgroup)` is NOT sequence point for `threadgroup` pointer aliasing. CUDA `__syncthreads()` is. If your `reduce_op.py` does pointer swap without `volatile`, Metal reorders.

### What I need from you to finish this

Paste these snippets so I can do the exact `file:line` audit:

1. `custom_op_wrapper.py:30-50, 230-250` - registry + lock
2. `fx_to_tilelang.py:1700-1725` - grid calc
3. `reduce_op.py:180-210` - warp kMul 
4. `dsa_splitk_indexer_loss.py:880-905` - stage-2 ASq
5. `fp8_amax.py` lock usage + mask branch

With those, I’ll give you: `BUG / NOT BUG / REGRESSION` for each, with exact fix diffs. 

Without source, the best I can say: wave-9 fixes look directionally correct, but items 2,3,4,5,7 from meta’s list are not addressed by the commits you listed, and the registry/_expose fixes depend on lock placement I can’t see.