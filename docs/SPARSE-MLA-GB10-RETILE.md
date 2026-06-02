# Sparse-MLA (DeepSeek-V3.2 / DSA) GB10 sm_121 re-tile

Arch-aware 99 KiB-fitting variant of `examples/deepseek_v32/sparse_mla_fwd.py`
and `sparse_mla_bwd.py`. The Hopper (sm_90) path is preserved byte-for-byte;
the GB10 (sm_120 / sm_121) path is selected via a new `gb10=` kwarg (or auto-
detected from the live compute-capability / `TILELANG_COMPUTE_CAP`).

## The constraint

GB10 / sm_121 caps **dynamic** shared memory at **99 KiB/block = 101376 B**
(`0x18c00`). Hopper sm_90 = 227 KiB, B200 sm_100 = 228 KiB. Confirmed live on
GB10: the ptxas reject prints `0x18c00 max` exactly (see Compile result below).

## What the re-tile changes (arch-gated, Hopper untouched)

Forward (`sparse_mla_fwd.py`), when `gb10` resolves True:
- `pass_configs += TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True` — overlays
  non-overlapping smem live ranges (Q dead after the QK GEMMs overlays the PV
  output region, etc.). This is the same load-bearing flag cppmega/Megatron's
  DSA ships on GB10.
- `pass_configs += TL_DISABLE_TMA_LOWER: True` — sm_12x has no Hopper TMA path.
- **Drop `O_shared`** (64 KiB at H_per_block=64, D=512): store `acc_o` straight
  to `Output` instead of the `acc_o -> O_shared -> Output` round-trip. Mirrors
  cppmega `tilelang_sparse_mla_fwd.py:287`.
- New `static_shape=(batch, seq_len, seq_len_kv)` build option to bake concrete
  dims into the prim_func (see the tirx regression note below).

Backward (`sparse_mla_bwd.py`), when `gb10` resolves True:
- Keeps `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True` (already present).
- `pass_configs += TL_DISABLE_TMA_LOWER: True`.
- **Halve `block_H` 64 -> 32** — cuts the three dominant 64 KiB Q/dO/dQ buffers
  in half (the highest-leverage bwd knob). An explicit `block_H_override`
  takes precedence. bwd is already fully static-shaped (B,S,S_kv are concrete
  ints), so it does not hit the tirx dynamic regression.

The original tile dims are otherwise unchanged: `block_I=64`, `dim=512`,
`tail_dim=64`, `threads=256`, `topk` a multiple of `block_I` — matching the
production tile that cppmega's DSA fits on GB10.

## Smem math (analytical, heads=128 -> H_per_block=64, D=512, D_tail=64, bf16=2B/fp32=4B)

Forward GB10 variant (O_shared dropped), single-stage buffer set:

| buffer        | shape    | bytes  |
|---------------|----------|--------|
| Q_shared      | [64,512] | 65536  |
| Q_tail_shared | [64,64]  |  8192  |
| KV_shared     | [64,512] | 65536  |
| K_tail_shared | [64,64]  |  8192  |
| S_shared      | [64,64]  |  8192  |
| Lse_shared    | [64]     |   256  |
| naive sum (stages=1)                | **155904 B = 152.2 KiB** |

The aggressive-merge pass overlays the non-simultaneously-live buffers. The
binding live set is the PV GEMM `S_shared @ KV_shared -> acc_o`:
`KV_shared(64K) + S_shared(8K) + K_tail(8K) + Q_tail(8K) + Lse(0.25K) ~= 88.2 KiB`,
which fits 99 KiB **at num_stages=1**. At **num_stages=2** the Pipelined loop
double-buffers KV_shared/K_tail/S_shared, pushing the lowered footprint to
**225 KiB**, which overflows (see below).

Backward GB10 variant: `block_H=32` drops the naive sum from 288 KiB
(block_H=64) to **180 KiB**; the merge pass overlays Q/dO/dQ (non-overlapping
lifetimes) plus the per-split-store `acc_dkv` reuse to land under 99 KiB. The
`acc_dkv*` accumulators must stay fp32 (they are the `atomic_addx4` targets).

## GB10 compile result (sm_121a, real ptxas)

Environment: gb10, `/home/dave/cppmega-venv/bin/python`, torch
2.13.0.dev+cu132, tilelang 0.1.9+cuda.git1d5473a5, branch
`merge/upstream-codegen-reorg`, `cap (12, 1)`.

- The re-tiled fwd **lowers all the way through to ptxas on `-arch=sm_121a`**
  (no TMA/warp-spec reject; the kernel body, GEMMs, online-softmax, direct
  acc_o store all codegen). This confirms the arch targeting is correct.
- At `block_I=64, num_stages=2, gb10=True` ptxas reports:
  ```
  ptxas error : Entry function 'main_kernel' uses too much shared data
                (0x38400 bytes, 0x18c00 max)
  ```
  = **used 0x38400 = 230400 B = 225.0 KiB** vs **max 0x18c00 = 101376 B =
  99 KiB**. So num_stages=2 still overflows: the aggressive-merge + drop-O
  reclaims O_shared but cannot defeat the 2x double-buffering of the staged
  KV/K_tail/S buffers. **num_stages=1 (or block_I=32) is required to fit.**

## REMAINING BLOCKER (honest): branch-wide tirx eager-builder regression

The configs that the smem math says fit (num_stages=1, and/or block_I=32) are
**blocked by a pre-existing regression on `merge/upstream-codegen-reorg`**, NOT
by this re-tile:

```
TypeError: '# from tvm.script import tirx as T  @T.prim_func def main(...) ...'
           is not a callable object
  at tilelang/language/eager/builder.py:1256 (inspect.signature(func))
  via  tilelang/jit/__init__.py:603 (prim_func(func, eager_jit=True))
```

The new `tilelang.jit` eager-builder calls `inspect.signature` /
`inspect.getsource` on the prim_func, and on this branch the tir->tirx
migration makes that receive the printed TVMScript instead of a function for
the sparse-MLA kernel. Evidence this is branch-wide and not the re-tile:
- A trivial tilelang kernel compiles fine on the same branch.
- The **Hopper path (`gb10=False`) of the SAME sparse-MLA fwd also fails** with
  the identical tirx TypeError — so the failure is independent of the GB10
  flags.
- The failure is intermittent vs the JIT cache: a clean first compile of
  `block_I=64,num_stages=2,gb10=True` reached ptxas (the 225 KiB number above);
  cached / repeated builds re-enter the buggy eager-builder path and raise the
  tirx TypeError before ptxas.

STATIC shapes (the new `static_shape=` option) were applied to dodge the
dynamic-symbolic part of the regression and DID get the kernel past TIR
lowering to ptxas once — but the eager-builder `inspect.signature` issue is the
deeper, dominant blocker and must be fixed in the branch's
`tilelang/language/eager/builder.py` / `tilelang/jit/__init__.py` before the
sub-99 KiB configs can be reliably built and a numeric parity check can run.

## Parity

NOT YET RUN. A re-tiled-fwd-vs-reference rel-err check requires a fitting
config (num_stages=1) to build, which is gated by the tirx regression above.
Once that branch bug is fixed, run `test_sparse_mla_fwd(..., num_stages=1,
gb10=True)` which calls `assert_tensors_similar(tl_out, ref_out, eps=1e-2)`.

## vs Megatron / cppmega tiling

cppmega's DSA (`cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py`)
ships the SAME tile (`block_I=64, num_stages=2, threads=256, H_per_block=64`)
and fits 99 KiB on GB10 via exactly `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`
+ `TL_DISABLE_TMA_LOWER` + direct acc_o store (no O_shared). This re-tile
mirrors that recipe and adds the `num_stages=1` / `block_H=32` fallback levers.
The fact that cppmega's num_stages=2 fits while our num_stages=2 reports 225 KiB
suggests cppmega's tilelang build (a different, non-codegen-reorg revision)
lowers the merge more effectively; on the current `merge/upstream-codegen-reorg`
tilelang the safe fit is num_stages=1 (once the tirx eager-builder bug is
fixed). cppmega's `preflight_smem_check.py` gates the 99 KiB cap at runtime via
`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`.

## Files

- `examples/deepseek_v32/sparse_mla_fwd.py` — arch-aware fwd (gb10 + static_shape)
- `examples/deepseek_v32/sparse_mla_bwd.py` — arch-aware bwd (gb10 + block_H_override)
