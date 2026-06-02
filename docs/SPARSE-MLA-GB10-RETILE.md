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

**STATUS 2026-06-02 UPDATE (measured on gb10, libtilelang.so relinked 12:38,
branch `merge/upstream-codegen-reorg` @ `ce315509`, submodule tvm `62cc0314`):
BLOCKER 1 and BLOCKER 2 are BOTH FIXED and PROVEN. The re-tiled sparse_mla_fwd
now COMPILES, LAUNCHES, and RUNS e2e on GB10/sm_121a -- the
`cuFuncSetAttribute(95216)` dynamic-smem reject is GONE (Blocker 1) and the host
LLVM `CodeGenCPU` build no longer aborts (Blocker 2). bwd's three kernels
(preprocess / `sparse_mla_bwd_kernel` with `AtomicAddx4` / postprocess) also
compile + launch clean on sm_121a. A THIRD, pre-existing bug -- a layout-
inference conflict in the `block_I=16` re-tile -- now blocks NUMERIC PARITY (see
BLOCKER 3 below); it is independent of the two codegen fixes (which cannot alter
numerics) and was simply masked until the kernel could finally launch.**

### The two fixes (commit `ce315509`, tilelang `src/`, no tvm submodule change)

**Blocker 1 fix -- `src/backend/cuda/codegen/codegen_cuda.cc`
`CodeGenTileLangCUDA::VisitStmt_(AllocBufferNode)`:** the merge pass
(`MergeSharedMemoryAllocations`) lowers the merged dynamic buffer as an
`AllocBuffer` (NOT an `Allocate`) and the var DOES keep its correct
`shared.dyn` PointerType scope -- the scope was never dropped. The bug was that
the tilelang CUDA codegen forwarded every non-barrier `AllocBuffer` to the base
`CodeGenC::VisitStmt_(AllocBufferNode)`, which ALWAYS emits a SIZED
`extern __shared__ __align__(1024) uchar buf_dyn_shmem[95216];`. Only the
`AllocateNode` path had the `scope=="shared.dyn"` -> unsized `[]` special case;
the `AllocBufferNode` path did not. Fix: special-case `scope=="shared.dyn"` in
the tilelang `AllocBufferNode` visitor to emit the UNSIZED
`extern __shared__ __align__(1024) uchar buf_dyn_shmem[];`. VERIFIED in the
generated source: the declaration is now `buf_dyn_shmem[]` (unsized), ptxas
treats the 95216 B as DYNAMIC, and the runtime `cuFuncSetAttribute(95216)`
succeeds -> `LAUNCH_OK`.

**Blocker 2 fix -- `src/transform/arg_binder.cc` compact-stride assert:** the
DLTensor "compact array" stride check built its running stride product (a
flattened element count) in `buffer->DefaultIndexType()` = int32. For large
static shapes the product exceeds 2^31, so the int32 IntImm overflowed when
`CodeGenCPU` materialized the `AssertStmt` constant via `ConstantInt::get`
(APInt `isIntN(32)` abort). Fix: accumulate the compact stride product and its
compared shape values in int64. VERIFIED: the full host JIT build now completes
with no APInt abort (fwd + all bwd kernels build on the static-shape path).

### BLOCKER 3 (numeric parity) -- ROOT CAUSE CORRECTED 2026-06-02 (gb10, measured)

**The earlier "layout-inference conflict" attribution below is WRONG and is
superseded.** A full measured bisection on gb10 sm_121 (clean lib relinked
14:0x, branch `merge/upstream-codegen-reorg`) establishes:

1. The `61x parallel.cc:619 "Layout infer conflict between sumexp and None"`
   warnings are **cosmetic, not the bug**. Instrumenting
   `ValidateCandidateAgainstFragments` shows all 61 are `throw_on_error=0 READ`
   conflicts emitted while `ChooseBestCandidate` VALIDATES (and rejects) the
   under-replicated `replicate:4` PlanLoopPartition candidate; it then correctly
   returns the `replicate:8` buffer candidate (`buf_ok && !plan_ok` branch). The
   buffer fragment ends up `replicate:8` (the acc_o/reduce layout) -- consistent.
   Pinning `sumexp` to a hand `replicate:8` layout via `T.annotate_layout`
   removed the warnings but did NOT change parity; forcing the scalars
   fully-replicated produced a *hard* `Layout may conflict with ReduceOp
   (m_i vs acc_s)` (good -- fail-loud) but still not the fix. So the softmax
   normalization is NOT corrupted by a silent layout pick.

2. The defect is **NOT `block_I`-specific.** At `H=16, threads=32` BOTH
   `block_I=16` AND `block_I=64` give the same wrong `cos=0.728` -- the campaign
   never tested `block_I=64` on gb10 (it can't fit 99 KiB at H_per_block=64, so
   it was assumed-good from Hopper, never measured). `block_I`, `num_stages`,
   `threads` (64/128/256, i.e. n_warp=1 vs 2), the aggressive-merge pass, and the
   drop-O direct store are ALL exonerated by direct A/B (each toggled
   independently, parity unchanged at ~0.7).

3. **Real root cause: the indexed KV gather feeding the attention is
   miscompiled in this kernel's full structure.** A standalone flash-attention
   fwd with the identical online-softmax + stmatrix + reduce + two-GEMM +
   tail-dim-split + masked-init pattern gives `cos=1.000000` on sm_121 when KV is
   loaded with a contiguous `T.copy`. Swapping ONLY that load for the sparse-MLA
   gather (`Ks[bi,d]=K[Idx[bi],d]`, the exact deepseek pattern) drops it to
   `cos=0.70` -- WITH IDENTICAL KEYS (arange indices). i.e.
   `gather-vs-contiguous, same keys -> cos 0.699` (reproducible). The generated
   gather store address into the swizzled `Ks` is byte-identical to the working
   `cp_async_gs` store, the cp.async sync/commit/wait is identical, and the
   4-operand `cp.async.cg ...,16` predication is NOT the cause (patching it to the
   3-operand form or to always-copy left parity unchanged). An *isolated* gather
   (gather -> swizzled shared -> single or double GEMM -> +softmax -> per-query
   multi-block) reproduces CORRECTLY (`cos=1.0`); the corruption only appears in
   the full kernel, so the trigger is the gather combined with the rest of the
   online-softmax loop (the multi-iteration `alpha`/`m_i_prev` running rescale is
   the remaining un-isolated piece). This is a TileLang gather/copy-to-swizzled-
   shared lowering bug, not a layout-inference or block-shape issue.

**STATUS: characterized, NOT yet fixed.** Parity does NOT pass. Fwd
`cos~0.70` / output direction wrong (not a scale error). No fabrication: the
fused sparse-MLA COMPILES (<=99 KiB), LAUNCHES, and RUNS on gb10 sm_121, but is
numerically WRONG due to the gather lowering above. The next step is to bisect
the gather + online-rescale interaction to a minimal repro and fix the gather
store layout in the copy/swizzle lowering.

----

#### Superseded (incorrect) Blocker-3 note kept for history

With both codegen bugs fixed, the kernel runs but the output is WRONG. Measured
on gb10 (S=1024, SKV=2048, topk=512, H=128): the fwd global similarity is only
**0.50-0.72** (target ~1.0), rel-err-mean ~4.6 -- across `gb10=True` (direct
acc_o store), `gb10=False` staged-store at the same small tile, H=64 (no head
replication) AND H=128. Compilation prints **61x** `parallel.cc:619 "Layout
infer conflict between sumexp and None in T.Parallel loop"` -- the online-softmax
`sumexp` fragment gets two incompatible layouts (`(64,)->(1,) replicate:4` from
the reduction loops vs `(64,)->(2,) replicate:8` from the GEMM-induced layout)
and the pass proceeds with an inconsistent layout instead of throwing
(`throw_on_error=false`), corrupting the softmax normalization. The bwd shows the
analogous `LayoutConflictException "delta vs acc"` during the preprocess delta
reduction. This is a TileLang layout-inference defect surfaced by the
`block_I=16` / `H_per_block=64` / `threads=256` re-tile, NOT a codegen/argbinder
bug; it must be fixed (correct the `sumexp` reduction layout for the small-BI
tile, or flip `throw_on_error` so it fails loud) before a parity rel-err can
pass `eps=1e-2`. Both Blocker-1/2 fixes are unrelated to it (pure codegen).

----

**PRIOR STATUS 2026-06-02 (superseded -- kept for the device-fit proof below;
libtilelang.so/libtvm_compiler.so relinked 11:47, branch
`merge/upstream-codegen-reorg` @ `5c21aeff`, submodule tvm `62cc0314`): fwd
compiles the DEVICE cubin through real sm_121a ptxas, but was blocked by TWO
downstream bugs before a parity number could be produced.** No fabrication --
exact measured errors below.

### Device-side: CONFIRMED GOOD (ptxas fits 99 KiB)

`tilelang.set_pass_config(TL_ENABLE_PTXAS_VERBOSE_OUTPUT)` + the auto-gb10 path
(`_select_gb10(None) -> True` on cap (12,1)) compiled `main_kernel` to a real
sm_121a cubin. Recompiling the cached `device_kernel.cu` with
`nvcc -arch=sm_121a --cubin --ptxas-options=--verbose`:

```
ptxas info : Used 248 registers, used 3 barriers, 95216 bytes smem
cuobjdump -res-usage main_kernel: REG:248 SHARED:96240 (= 95216 + 1024 driver)
```

**95216 B = 93.0 KiB dynamic smem, ptxas SUCCEEDS** (vs the num_stages=2
0x38400 = 225 KiB overflow). `shared_memory_per_block_optin` on this GB10 =
**101376 B (99 KiB)**, `reserved=1024`. The re-tile fits with 5 KiB to spare.
This is the hardware proof the device kernel budgets correctly.

### BLOCKER 1 (runtime launch) -- dynamic smem emitted as STATIC

`test_sparse_mla_fwd(..., gb10=None)` compiles, then throws at the FIRST launch:

```
tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 95216
  (3rdparty/tvm/src/runtime/cuda/cuda_module.cc:218, cuFuncSetAttribute)
```

Root-caused by replaying the exact driver call on the cached cubin
(`/tmp/replay` -> `cuModuleLoad` + `cuFuncGetAttribute`):

```
static SHARED_SIZE_BYTES = 95216    <-- the 95216 B is STATIC, not dynamic
MAX_DYNAMIC (before)     = 6160      (= 101376 - 95216, all that's left)
cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 95216) -> CUDA_ERROR_INVALID_VALUE
```

The generated `device_kernel.cu` declares
`extern __shared__ __align__(1024) uchar buf_dyn_shmem[95216];` -- WITH an
explicit compile-time `[95216]` size. ptxas therefore counts it as **static**
shared memory, consuming the full 95216 B statically; the TVM runtime then ALSO
tries to set the *dynamic* attribute to 95216, but static+dynamic would be
190432 > 101376, so the driver rejects it. (A control nvcc kernel with a TRUE
unsized `extern __shared__ char s[]` accepts `cudaFuncSetAttribute(95216..101376)`
fine on this GB10 -- so the device/driver is healthy; the bug is the sized
declaration.) Both active codegens
(`src/backend/cuda/codegen/codegen_cuda.cc:4335`, `3rdparty/tvm/.../codegen_cuda.cc:1387`)
emit `vid[];` (unsized) ONLY for `scope == "shared.dyn"` -- so the merged
`buf_dyn_shmem` reaches codegen with scope `"shared"` (static) instead of
`"shared.dyn"`, hitting the `[constant_size]` branch. The merge pass declares
the var with `shared.dyn` scope (`merge_shared_memory_allocations.cc:1793`), so
the scope is being dropped between the AllocBuffer and codegen. **Fix target:
preserve the `shared.dyn` PointerType scope on the merged dynamic buffer so
codegen emits `buf_dyn_shmem[]` (unsized) and the 95216 flows only to the
dynamic launch attribute.**

### BLOCKER 2 (host LLVM build) -- int32 IntImm overflow in AssertStmt

The full JIT/AOT build path (`get_kernel_source` / `LLVMModuleNode::Init`)
aborts in host CPU codegen (gdb backtrace):

```
python: llvm/include/llvm/ADT/APInt.h:121: APInt::APInt(...):
  Assertion `llvm::isIntN(BitWidth, val) && "Value is not an N-bit signed value"' failed.
#6  llvm::APInt::APInt
#7  llvm::ConstantInt::get(llvm::Type*, unsigned long, bool, bool)
#8  tvm::codegen::CodeGenCPU::VisitStmt_(tvm::tirx::AssertStmtNode const*)
#13 tvm::codegen::LLVMModuleNode::Init
```

A 64-bit constant in a host-wrapper **AssertStmt** (the DLTensor shape/stride
bind checks built in `src/transform/arg_binder.cc`) is materialized through
`ConstantInt::get` with an int32 width that the value overflows -- a flattened
byte/element count product computed at int32 for the static-shape path. The
prim_func body itself has NO overflowing int32 IntImm before the host split
(scanned), so the bad constant is introduced during host arg-binding /
`make_packed_api`. `tilelang.lower(mod)` alone does NOT crash; only the host
LLVM module build does. **Fix target: build the arg-binder shape/stride
byte-count asserts in int64 (or cast the IntImm to int64 before
`ConstantInt::get`) on the static-shape path.** This is independent of Blocker 1
and was the bug behind the earlier LLVM APInt crash.

### Why the 12:18 cubin compiled but launch failed

The device cubin is JIT-compiled by `tilelang_callback_cuda_compile`
(`--cubin -arch=sm_121a`) and that path does NOT run CodeGenCPU, so Blocker 2
does not fire there -- the device kernel builds clean to 93.0 KiB. Blocker 1
(static-vs-dynamic smem) only manifests at the runtime `cuFuncSetAttribute`, and
Blocker 2 only manifests on the host LLVM module build. Both must be fixed before
`assert_tensors_similar(eps=1e-2 fwd / 1e-4 bwd)` can yield a rel-err number.

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
