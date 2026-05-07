---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T16:27:24.248385+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/src/op/reduce.cc', '/private/tmp/tl_poc_review/src/tl_templates/cuda/reduce.h', '/private/tmp/tl_poc_review/src/tl_templates/hip/reduce.h']
---
I tried to locate the Wave-12 commits you mentioned across the 6 files you uploaded, but none of those commit IDs or symbol names appeared in `/mnt/data/fp8_amax.py`, `dsa_splitk_indexer_loss.py`, `custom_op_wrapper.py`, `reduce.cc`, `reduce.h`, or `reduce_4e0ab0.h`.

Without the actual diffs for `e946f98`, `dde5e28c`, `cc3194ae`, `284a907e`, `801c27d`, `e70277b`, I can't give file:line findings or verify regressions vs wave-11. The uploaded files don't contain `_q_cache_bytes`, `_can_use_q_cache_v5`, `CumSum1D/2D`, `_detect_storage_overlap`, or `_resolve_in_dtype`.

**To do the correctness audit you requested, I need one of these:**

1. **Paste the diff or relevant code blocks** for those 5 Wave-12 changes, or
2. **Upload the actual source files** that contain those functions: likely the TVM scheduler/cache code, `src/tir/transforms/`, `src/relay/op/tensor/`, and `python/tvm/` files that were patched.

### What I can review right now from your uploads

From the 6 files I do have:

| File | Quick notes |
| --- | --- |
| `fp8_amax.py` | Looks like FP8 absmax scaling helpers |
| `dsa_splitk_indexer_loss.py` | Split-K indexing + loss calc, probably for attention |
| `custom_op_wrapper.py` | Wrapper for custom ops |
| `reduce.cc/h`, `reduce_4e0ab0.h` | Reduction kernels, likely CUDA/HIP |

If you want me to audit *these* files for correctness bugs, I can start immediately. For off-by-one, None handling, type mismatches, race conditions, etc, I just need to know which function to prioritize.

### For the Wave-12 specific review

Drop the diffs or the actual patched files and I'll give you:

1. **Line-level correctness verdicts** for each of the 5 commits
2. **Attacker scenarios** where None/overflow/type assumptions break
3. **Regressions vs wave-11** - I’ll check if old edge cases now fail
4. **Test coverage gaps** - exact inputs missing for each change

Want to paste the diff for `e946f98` first and we’ll knock them out one by one?