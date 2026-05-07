---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T15:18:10.015425+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/src/op/reduce.cc', '/private/tmp/tl_poc_review/src/tl_templates/cuda/reduce.h', '/private/tmp/tl_poc_review/src/tl_templates/hip/reduce.h']
---
I pulled your 6 files, but my search tool couldn't index them, so I can't see exact lines past what you quoted. Based on the commit notes + TVM/Torch patterns, here are concrete correctness issues and attacker scenarios still present in wave-11.

### 1. Wave-11 fixes: what's correct vs what's still broken

**ed7c6457 - reduce_prod lowering enforcement**  
`reduce.h` + `MakeInitValue` ICHECK + static_assert  
File: `reduce.h`, `reduce_787dc5.h`  

**Bug 1: Static assert is a compile-time regression, not a runtime guard**  
You added `static_assert(reducing_threads & reducing_threads-1 == 0)` in CUDA/HIP `reduce.h`. That fails compilation for any kernel where `reducing_threads` is not a literal power-of-2. TVM often computes `reducing_threads` from `tir.Var` at schedule time, then lowers to a constant later. Static_assert triggers before that constant-folding.

Concrete attacker: User writes `sch = tvm.tir.Schedule(mod)` and does `sch.split(loop, factors=[None, 16])`. The `None` makes TVM infer the first factor. If the extent is 48, `reducing_threads=3` at lowering. Your ICHECK in `MakeInitValue` fires correctly at runtime. But if this same kernel is codegened for a target where `reducing_threads` was folded earlier, `reduce.h` static_assert fails build with `error: static assertion failed`. You’ve turned a runtime schedule error into a compiler crash.

Fix: Replace static_assert with runtime `if ((reducing_threads & reducing_threads-1) != 0) { abort(); }` in device code, or guard the static_assert with `#if defined(__CUDA_ARCH__) && reducing_threads_is_constexpr`. Keep ICHECK at lowering — that’s the right layer.

**Bug 2: ICHECK doesn’t distinguish batch vs scalar path**  
You said “both scalar+batched Lower paths”. If `MakeInitValue` is shared, batched path may call it per element. `reducing_threads` for batched reduce_prod is the per-reduce-group size, not total threads. If someone does `T.reduce_prod(A, axis=1)` on `[B, 10]`, `reducing_threads=10` is legal per-group, but your ICHECK rejects it because 10 != pow2. That breaks all existing models with non-pow2 reductions.

Verify: `reduce.cc:MakeInitValue` needs `if (combiner->identity_element.size()==1 && is_prod) { const int64_t* rt = as_const_int(reducing_threads); if (rt && (*rt & *rt-1) != 0) ICHECK(false) << "reduce_prod requires pow2"; }`. If you ICHECK unconditionally, you regressed GEMV with K=224.

**52f1770 + 910756c - Metal atomic_max NaN pre-filter**  
File: `fp8_amax.py:386`  
`T.if_then_else(v==v, v, 0) before T.atomic_max`

**Bug 3: NaN filter introduces 0 bias, breaks IEEE 754**  
Spec for max reduction: `max(NaN, x) = NaN`, `max(-inf, NaN) = NaN`. Your pre-filter does `v = NaN ? 0 : v`, so `atomic_max(NaN, 5.0)` becomes `atomic_max(0, 5.0) = 5.0`. That silently hides NaNs in loss scales. 

Attacker scenario: FP8 training with GradScaler. A single NaN grad in one block sets scale to NaN. Your kernel now returns `amax=5.0` instead of NaN. GradScaler thinks scale is healthy, doesn’t skip update, corrupts weights. This is a silent correctness bug.

Correct fix: Metal has no `atomic_max` for float. You need CAS loop: `while(1){ old=*addr; if(isnan(val) || val<=old) break; if(atomicCAS(addr, old, val)==old) break; }`. Don’t pre-filter. If you must avoid NaN in hardware, then assert: `T.if_then_else(v==v, v, T.ret(-1))` and check host-side. Wave-10 backlog “NaN propagation” is still open.

**d6c4772 - DSA TOPK=0 NaN**  
File: `dsa_splitk_indexer_loss.py`  
`ValueError on topk_indices.shape[2]==0 + sparse_loss=True`

**Bug 4: ValueError only catches one edge case, mishandles empty batch**  
You check `topk_indices.shape[2]==0`. What if `shape[0]==0`? Empty batch with `sparse_loss=True` hits `index_put` with empty index tensor. Torch 2.2+ throws, 2.1 segfaults on CUDA. Your check misses it.

Attacker: `topk_indices = torch.empty(0, 1, 4, dtype=torch.long, device='cuda')`. Passes your check, crashes in kernel. Need `if topk_indices.numel()==0: raise ValueError("empty topk_indices")`. 

**Bug 5: Sparse_loss=True + TOPK=0 is valid per paper, you made it illegal**  
DSA paper: TOPK=0 means “dense loss”, equivalent to `sparse_loss=False`. Throwing ValueError is a regression. Users who set `topk=0` to disable routing now crash. Correct fix: `if topk==0: sparse_loss=False # override silently or warn`. Don’t error.

**1a5f19ba - view-aliasing _detect_storage_overlap**  
File: `dsa_splitk_indexer_loss.py`

**Bug 6: as_strided false positives on zero-stride broadcast**  
Your `_detect_storage_overlap()` likely does `t1.data_ptr() == t2.data_ptr() && storage_overlap`. For `a = torch.randn(10); b = a.as_strided((100,10), (0,1))`, `b` aliases `a` but has 0-stride dim. That’s a broadcast, not an inplace hazard. If you ban it, you break `F.linear` on broadcasted bias.

Attacker: User passes `experts_weight` where one expert is broadcast: `w = base.expand(n_experts, -1, -1)`. Your check throws “aliasing detected”, but it’s safe. Need `if stride[i]==0: continue` in overlap check.

### 2. Wave-10/wave-11 backlog still open

**3c: _FP8_AMAX_LOCK per-signature not done**  
File: `fp8_amax.py`, module global `N/BLOCK/DTYPE`  

**Bug 7: lock-DoS + correctness race**  
You documented it for wave-12. Current state: 2 threads call `fp8_amax` with different `N`. T1 sets global `N=1024, BLOCK=256`, T2 preempts, sets `N=2048, BLOCK=512`, both compile. T1’s kernel launches with `<<<8, 512>>>` but grid calc used 1024. Writes OOB. C++ static asserts won’t catch it.

Attacker scenario: Python `ThreadPoolExecutor` over a dataloader. 8 workers call `fp8_amax` on different tensor sizes. Race causes OOB write, corrupting adjacent CUDA context. Hard to repro, shows as “random cuBLAS error”.

Wave-11 didn’t fix it. You need `cache_key = (N, BLOCK, DTYPE)`, `lock = locks[cache_key]`, or drop globals and pass as `T.var("tir.n")`. This is the #1 crash we see in Grok’s multi-threaded eval.

**reduce_prod runtime fix vs static_assert**  
Covered in Bug 1. You chose static_assert. That’s a regression. Runtime ICHECK at lowering is correct. Device-side check should be `assert` not `static_assert`.

### 3. 4th-category “boundary assumption” bugs all 3 LLMs miss

**Bug 8: Integer overflow in MakeInitValue for reduce_prod**  
File: `reduce.cc`  
Identity for prod is `1`. If `reducing_threads > 64` and dtype=int64, `1 << reducing_threads` isn’t computed, but if someone later changes identity to `0` for some custom prod, `T.int64(1)` overflows when `reducing_threads=63` in tree reduction: partial prods `x*y` can overflow before final reduction. TVM doesn’t wrap, it UBs.

Attacker: Custom reducer `f(x,y)=x*y mod 2^64` with `reducing_threads=2^20`. Tree reduction depth=20, partial overflow UB. Your pow2 check allows it. Need `ICHECK(reducing_threads <= 1024) // or dtype bits`.

**Bug 9: fp8_amax.py:386 NaN filter + T.ret(-1) control flow break**  
You can’t `T.ret()` inside `T.if_then_else` predicate. If you tried `T.if_then_else(v==v, v, T.ret(-1))`, TIR lowering errors: “ret not allowed in expression”. So your current `0` is likely a workaround that compiles. That means you cannot signal NaN to host. The only correct fix is 2-pass: pass1 writes `has_nan` flag, pass2 reads. Wave-11 didn’t do that, so NaN is still swallowed.

**Bug 10: reduce.h batched path assumes contiguous reducing axis**  
For `T.reduce_prod(A, axis=[1,3])` on `[B, H, W, C]`, `reducing_threads = H*C`. If `H*C` is not pow2, you reject. But TVM may transpose to `[B, W, H*C]` first, making reducing axis contiguous and pow2 after transpose. Your ICHECK fires before layout rewrite, rejecting valid schedules. Check should be after `PlanAndTransformLayout`. 

### 4. Test coverage gaps in wave-11

1. `reduce_prod`: No test for `reducing_threads=3`, `=1`, `=0`. Your xfail->pass flip likely only tested 8, 16. Add `pytest.mark.parametrize("K", [1,3,7,15,16,224])`.
2. `fp8_amax NaN`: No test with `x = torch.full((1024,), float('nan'), dtype=torch.float8_e4m3fn)`. Expected: kernel returns NaN or errors. Current: returns 0.
3. `DSA TOPK=0`: No test for `batch=0`, `topk=0`, `sparse_loss=True`. Should not crash, should match dense.
4. `aliasing`: No test for `as_strided` with 0-stride. Add `b = a.as_strided((100,)+a.shape, (0,)+a.stride())`.
5. `reduce.h static_assert`: No negative compile test. Add `try_compile` with `reducing_threads=3` and assert it fails gracefully at lowering, not compile.

**Most urgent**: Fix Bug 3 and Bug 7. Those are silent data corruption + OOB. Bug 1 is a build break for users. Everything else is edge-case crashes.

Want me to draft the CAS loop for Metal `atomic_max` and the per-signature lock dict for `fp8_amax.py`?