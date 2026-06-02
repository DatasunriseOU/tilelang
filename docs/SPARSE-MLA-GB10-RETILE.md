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

## GB10 compile result (sm_121a, real ptxas) -- MEASURED, FINAL

Environment: gb10, `/home/dave/cppmega-venv/bin/python` (py3.13), CUDA 13.3
ptxas (`release 13.3, V13.3.33`), branch `merge/upstream-codegen-reorg` @
`72e10529`+ (eager-builder PrimFunc fix landed), submodule 3rdparty/tvm @
`62cc0314`, `cap (12, 1)`, `-arch=sm_121a`.

The smem budget is dominated by the **always-live `Q_shared` tile**
(`[H_per_block=64, D=512]` bf16 = 64 KiB) which the aggressive-merge pass
canNOT overlay (it is read by the QK GEMM on every iteration). So the lowered
footprint is essentially the naive single-stage buffer sum -- the earlier
"merge overlays Q -> ~88 KiB" estimate was analytical optimism and did NOT
hold at ptxas. The fit was found empirically by shrinking the index tile.

### Forward -- ptxas budget vs `block_I` (num_stages=1, drop-O, merge on)

| block_I | smem used        | vs 99 KiB (0x18c00) |
|---------|------------------|---------------------|
| 64      | 0x26000 = 152.0 KiB | OVERFLOW          |
| 32      | 0x1c000 = 112.0 KiB | OVERFLOW          |
| **16**  | **0x173f0 = 93.0 KiB** | **FITS, compiles** |

`num_stages=2` double-buffers KV/K_tail/S -> 225 KiB (0x38400) overflow, so the
GB10 path also forces `num_stages=1`. **FINAL fwd GB10 config: `block_I=16,
num_stages=1`** (auto-applied when `gb10` resolves True and the caller left the
Hopper defaults). 93.0 KiB, full compile to a `JITKernel` -- no ptxas reject.

### Backward -- ptxas budget (block_H, block_size, threads)

`block_H` cannot drop below 32 and `block_size` cannot drop below 32 *at
threads=256* -- both trip the GEMM constraint `warp_row_tiles must be greater
than 16`. At the minimum valid `threads=256, block_H=32, block_size=32` the
kernel is **108.0 KiB (0x1b000)** -- overflow by 9 KiB. Dropping to
**`threads=128`** (4 warps) relaxes the warp-row-tiles constraint so
`block_size=16` becomes valid, halving KV_shared / P_shared_cast /
dP_shared_cast / acc_dkv_shared:

| threads | block_H | block_size | smem used        | result   |
|---------|---------|-----------|------------------|----------|
| 256     | 32      | 32        | 0x1b000 = 108.0 KiB | OVERFLOW |
| 128     | 32      | 16        | **0x163e0 = 89.0 KiB** | **FITS** |

**FINAL bwd GB10 config: `threads=128, block_size=16, block_H=32`**
(auto-applied on the gb10 path). The `acc_dkv*` accumulators stay fp32 and the
codegen emits `AtomicAddx4(...)` for both `dKV` and `dKV_tail` (verified in the
generated `tvm_kernels.cu`) -- the `atomic_addx4` path is preserved, not
degraded.

## tirx eager-builder regression -- RESOLVED

The prior blocker ("`<TVMScript> is not a callable object`" at
`inspect.signature` in the eager builder) was a real branch-wide bug: a
`tirx.PrimFunc` has no `__call__`, so feeding the already-built PrimFunc to
`tilelang.jit()` crashed. **Fixed in commit `72e10529`** (`fix(jit): handle
PrimFunc input in tilelang.jit`): `tilelang.jit`'s decorator now branches on
`isinstance(func, PrimFunc)` (lazy mode, `func.script()` source, empty
signature), mirroring `compile()`/`get_tir`. With that fix both the fwd
(lazy-builder-wrapped path) and bwd (`jit(builder)` PrimFunc path) lower all the
way through to ptxas on sm_121a. No fallback was added; the non-callable path
is unchanged.

## Parity

See the "Parity" run section appended below once the GB10 GPU is idle. The
fitting configs above (fwd block_I=16/ns=1, bwd threads=128/bs=16) build to real
`JITKernel`s, so `test_sparse_mla_fwd(..., gb10=True)` /
`test_sparse_mla_bwd(...)` (both call `assert_tensors_similar(eps=1e-2 fwd /
1e-4 bwd)`) can run for a rel-err check.

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
