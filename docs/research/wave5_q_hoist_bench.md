# Wave-5 stage-2 Q-hoist — empirical bench (BLOCKED)

Status: **BLOCKED — kernel construction crashes; budget gate also rejects realistic shapes.**

## Host

- macOS 26.4.1 (build 25E253)
- Apple M4 Max
- MLX 0.31.2 (venv `/Volumes/external/sources/cppmega.mlx/.venv`)
- Metal: available, applegpu_g16s
- TileLang dev build at `/private/tmp/tl_apache_tvm_swap/build`
- Run with `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` so `libtilelang.dylib` finds `libz3.dylib`.

## What was tested

`_bench_stage2_q_hoist_wave5(AB, AH, ASq, AD, Sk)` from
`cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py` after wave-7 #2
(commit `cac10a0`) supposedly fixed the wave-5 IR scoping bug.

## Findings

### 1. The wave-7 #2 fix (cac10a0) was incomplete.

`cac10a0` repaired the `s` IfFrame leak in stage-1 + stage-2 sparse paths
but the wave-5 stage-2 Q-cache path has the **same bug class on a different
variable** (`denom`):

```
RuntimeError: Immutable variable `denom` is used outside its defining region!
variable `denom` is defined in frame:
  script.ir_builder.tirx.IfFrame(
    stmts=(), condition=denom <= T.float32(0.0),
    then_stmts=(denom: T.float32 = T.float32(1.0),), else_stmts=None)
```

Fires on **every** budget-fits shape attempted:

| AB | AH | ASq | AD | Sk | Result |
|---|---|---|---|---|---|
| 1 | 1 | 64 | 32 | 512 | RuntimeError on `denom` |
| 1 | 1 | 128 | 64 | 2048 | RuntimeError on `denom` |
| 1 | 2 | 128 | 64 | 4096 | RuntimeError on `denom` |
| 1 | 4 | 128 | 64 | 512 | RuntimeError on `denom` |

A wave-7 follow-up needs the same `T.if_then_else(...)` rewrite that
`cac10a0` applied to `s`, applied to `denom` (and probably any other
let-binding inside an `IfFrame` in the wave-5 stage-2 path).

### 2. The wave-5 budget gate rejects most realistic shapes on Metal.

`_can_use_q_cache_v5(BLOCK_SQ=32, AH, AD, in_dtype="float16",
target="metal -thread_warp_size=32")` budget probe:

| AH \ AD | 32 | 64 | 128 |
|---|---|---|---|
| **1** | fits | fits | fits |
| **2** | fits | fits | fits |
| **4** | fits | fits | **rejected** |
| **8** | fits | **rejected** | **rejected** |
| **16** | **rejected** | **rejected** | **rejected** |
| **32** | **rejected** | **rejected** | **rejected** |

For DSA inference (AH=128, ASq=1, Sk=4096), or any production
multi-head shape with AH ≥ 8 and AD ≥ 64 (i.e. essentially every
realistic decoder-LM attention block), the budget gate refuses
wave-5 outright on Metal and the kernel falls back to wave-4 — so
the **"~2×" speedup claim cannot fire on production shapes** even
once the IR scoping bug is fixed.

### 3. Wave-4 vs wave-5 wall-clock — UNMEASURED.

Could not produce a single `(wave4_ms, wave5_ms, speedup)` data point
because every fits-budget shape crashes on `denom`, and every
non-fits shape is rejected by the budget gate before timing starts.
Wall-clocks observed are 0.03-0.04 s of pre-launch IR-construction
exception time, not kernel execution time.

## Reproduction

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  /Volumes/external/sources/cppmega.mlx/.venv/bin/python -c "
from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss \
  import _bench_stage2_q_hoist_wave5
print(_bench_stage2_q_hoist_wave5(AB=1, AH=1, ASq=64, AD=32, Sk=512))
"
```

Expected (today): `RuntimeError on denom`.

## What needs to land before bench is meaningful

1. **wave-7 follow-up** — apply the `cac10a0`-style `T.if_then_else(...)`
   rewrite to the `denom <= 0` IfFrame in stage-2 wave-5 path. Probably
   in `make_dsa_splitk_stage2_kernel` near the post-softmax denominator
   guard.
2. **wave-5 budget-gate revisit** — either lift `BLOCK_SQ` from 32, or
   tile heads inside the cache, so AH=8/AD=64 production shapes are
   reachable on Metal. As-is, wave-5 is opt-in for AH ≤ 4 + AD ≤ 64
   only — a tiny slice of real workloads.
3. **Then** re-run this bench across a 5-shape grid as originally planned.
