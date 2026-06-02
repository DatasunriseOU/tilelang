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

**STATUS 2026-06-02 (measured on gb10, libtilelang.so/libtvm_compiler.so relinked
11:47, branch `merge/upstream-codegen-reorg` @ `5c21aeff`, submodule tvm
`62cc0314`): fwd compiles the DEVICE cubin through real sm_121a ptxas, but is
blocked by TWO downstream bugs before a parity number can be produced.** No
fabrication -- exact measured errors below. Fused sparse-MLA does NOT yet RUN
e2e on GB10; the device kernel is proven to fit but the host launch path fails.

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
