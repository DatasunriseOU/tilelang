---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T01:41:55.826385+00:00
files: ['/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py', '/Volumes/external/sources/cppmega/cppmega/megatron/fp8_activations.py', '/Volumes/external/sources/cppmega/tests/test_fp8_amax_tilelang.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', '/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/_path_b_lowering.py']
---
**Code review summary (focus: correctness bugs from the TileLang port diff)**

The port in `/home/workdir/attachments/fp8_amax.py` + integration in `/home/workdir/attachments/fp8_activations.py` + test in `/home/workdir/attachments/test_fp8_amax_tilelang.py` is a solid, style-matched implementation that follows the `fp8_vecmat_path_c.py` / `_path_b_lowering.py` patterns (globals-for-decorator, `T.Kernel` + `T.Parallel`, target switching, lru_cache JIT, `tilelang_supports` gate). It correctly preserves the Triton `_amax_kernel` / `_quantize_kernel` semantics (block-reduce + `atomic_max`, scale + clamp + RNE cast to e4m3, `amax_val > 0` fallback, pre-zeroed buffer, contiguous handling).

**Critical correctness bugs introduced by this diff (will cause wrong results):**

1. **Missing shared-memory barrier after coalesced load (fp8_amax.py:205-230)**  
   ```python
   for i in T.Parallel(BLOCK):  # writes to X_shared
       ...
   for i in T.Parallel(BLOCK):  # immediately reads X_shared
       X_abs[i] = T.abs(...)
   T.reduce_max(...)
   ```
   No `T.barrier()` (or equivalent TileLang primitive — see `tilelang/language/__init__.py` for the exact name used in other kernels) between the two `T.Parallel` loops. This is a classic data-race on shared memory. The Triton reference never used shared memory; this optimization was added in the port but the sync was omitted. Will produce incorrect `amax` on Metal (and potentially on CUDA depending on scheduler).  
   **Fix:** Insert the barrier immediately after the load loop (before the abs loop), exactly as required by any GPU shared-memory pattern in the `cppmega_mlx` style guide.

2. **BLOCK_SIZE vs. threads mismatch in elementwise kernels (fp8_amax.py:170-180 for amax, ~300-310 for quantize; defaults at lines 60/70)**  
   ```python
   BLOCK = 1024
   threads = 128
   with T.Kernel(T.ceildiv(N, BLOCK), threads=threads) as bx:
       for i in T.Parallel(BLOCK):  # ← 1024 iterations
           gi = bx * BLOCK + i
           ...
   ```
   `T.Parallel(BLOCK)` with only 128 threads per block means the DSL must emit a strided inner loop (each thread handling ~8 elements). This is not the pattern used in `fp8_vecmat_path_c.py` (which either uses `T.vectorized`/`T.serial` for small fixed extents or `T.unroll` for the hot loop). The original Triton `_amax_kernel` treats `BLOCK_SIZE` as the *virtual* vector width per program ID, independent of thread count. This is likely to produce wrong indexing / partial coverage on the last block or on Metal.  
   **Fix:** Either (a) set `threads=BLOCK` (if TileLang allows) or (b) switch to the safer `T.vectorized` / local-buffer pattern shown in the vectorized branch of `fp8_vecmat_path_c.py:260-280`. Test explicitly with `N=100` (partial last block) vs. `torch.abs().amax()`.

**Medium-priority issues (no immediate breakage but fragile / regressions):**

3. **globals() mutation side-effect in every kernel build (fp8_amax.py:165, 260)**  
   ```python
   g = globals()
   g.update(_FP8_AMAX_BLOCK_SIZE=block_size, ...)
   ```
   Matches the `fp8_vecmat_path_c.py` style but is unnecessary here (you only ever read the *local* `N`/`BLOCK`/`FP8_MAX` variables inside the `@T.prim_func`). This mutates module-level state on every call to `make_fp8_*_kernel` (and therefore on every cache miss). Low risk today because `_amax_kernel_for` is cached and defaults are fixed, but introduces a subtle race if two different shapes/specializations are built concurrently.  
   **Recommendation:** Remove the `g.update` (pure locals suffice) to keep the file cleaner than the vecmat reference.

4. **Overly broad `except Exception` in integration (fp8_activations.py:145-150)**  
   ```python
   try:
       from cppmega_mlx.nn._tilelang.fp8_amax import ...
       ...
   except Exception:  # ← swallows everything
       has_tilelang = False
   ```
   This was already present for the optional dep, but the new import now hides real errors (e.g., TileLang version mismatch, malformed PrimFunc, missing intrinsics). The `fp8_amax_path_c_status()` exists precisely to give a clean reason string — use it.  
   **Fix:** Narrow to `except (ImportError, ModuleNotFoundError, AttributeError):` + `logger.debug(...)`.

5. **Minor API fragility in quantize path (fp8_amax.py:380)**  
   When `out` is supplied: `out_flat = out.reshape(-1)` (no `.contiguous()`). If the caller passes a non-contiguous `out`, the kernel write may be incorrect (TileLang Metal/CUDA lowering expects contiguous storage). The hot `fp8_pack_tilelang` path is safe (always new empty tensor), but the public `fp8_quantize_tilelang(..., out=...)` contract is now weaker than the Triton path.  
   **Fix:** `out_flat = out.reshape(-1).contiguous()` (or document the requirement).

**Edge cases handled correctly (no regressions):**
- `N == 0` (early return in wrapper, `fp8_amax.py:357`).
- `amax_val == 0` → `scale=1.0`, `inv_scale=1.0` (exact match to Triton).
- Last partial block masking (pad-0 for amax, skip-write for quantize).
- Clamp path (pre-clamp before amax + inner clamp in quantize kernel) — preserves original semantics.
- dtype gating + `_resolve_in_dtype` + target selection (cuda / metal).
- TE still takes precedence on CUDA (`fp8_activations.py:650`); TileLang only for MPS / non-TE CUDA.

**Test file notes (`test_fp8_amax_tilelang.py`):**  
Very good parity coverage. The Triton vs. TileLang exact match (`rtol=0, atol=0`) on CUDA is the gold standard. The Torch reference test uses a somewhat loose tolerance (`1e-3`); tighten to `1e-6` (or `rtol=0, atol=0`) once the shared-memory barrier is fixed — max reduction should be bit-identical.

**No other correctness regressions found:** No off-by-one in `gi = bx * BLOCK + i` or `ceildiv`, no swallowed exceptions on hot paths, no None/empty mishandling, no type mismatches in the PrimFunc signatures vs. the `float8_e4m3` / Torch `float8_e4m3fn` boundary, atomic_max + pre-zeroed buffer contract is identical to Triton.

**Performance (secondary, correctness first):**  
Fixed `BLOCK=1024` (vs. Triton's dynamic `_pick_block_size` based on `n//256`) is fine for correctness but can under-utilize SMs / increase atomic contention on very large tensors. Consider exposing/adapting the dynamic picker from `fp8_activations.py:480` into `_amax_kernel_for` once the kernel is solid.

**Overall verdict:** The port is ~90% there and matches the `cppmega_mlx` quality bar once the two kernel bugs above are fixed. After adding the barrier and auditing the `T.Parallel` distribution (or switching to the vecmat-style local-buffer pattern), it will be numerically identical on both CUDA and Metal with zero behavior change for existing CUDA users. The wrapper integration and status/test scaffolding are already production-ready.

Run the exact commands from the original request on Apple Silicon after the fixes to confirm parity. Let me know if you want a patched `fp8_amax.py` with the barrier + cleaner style.