# NVFP4 on MLX/Metal — implementation pointers (compact)

Pinned to MLX merge commit **`1eef1d155c96cdb87fa8e7ff995facf8ca26d369`** (PR #2946 "Metal/CPU nvfp4 and mxfp8", released in **MLX v0.30.3**, Jan 13 2026). MLX docs reference v0.31.2 with no API change.

## TL;DR

NVFP4 on Apple Silicon is **pure software** — no FP4 tensor core exists on M3/M4/M5. MLX exposes it through the regular quantization API: `mlx.core.quantize(w, group_size=16, bits=4, mode="nvfp4")` returns a 2-tuple `(packed_e2m1: uint32, scales: uint8)`; the optional FP32 per-tensor `global_scale` is *only* a `dequantize` keyword argument, not a tuple element. Scales are stored in a **separate buffer** (one FP8-E4M3 byte per 16 packed FP4 values), not interleaved.

---

## 1. MLX import path

`mlx.core.quantize` / `mlx.core.dequantize` / `mlx.core.quantized_matmul` with `mode="nvfp4"`. There is **no** `mx.fast.nvfp4_*` namespace. Layer-level helpers are `mlx.nn.quantize(model, mode="nvfp4", quantize_input=True, ...)`. `quantize_input=True` is gated to `nvfp4` and `mxfp8` only.

Source: [`python/mlx/nn/layers/quantized.py` line 11–17](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/python/mlx/nn/layers/quantized.py#L11):

```python
mode_defaults = {
    "affine": (64, 4),
    "mxfp4":  (32, 4),
    "nvfp4":  (16, 4),
    "mxfp8":  (32, 8),
}
```

## 2. Quantize / dequantize signatures

```python
# Returns a 2-tuple for nvfp4: (packed_e2m1, scales).  No biases, no global_scale.
w_q, scales = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")

# Dequantize accepts an optional FP32 per-tensor global_scale.
w_hat = mx.dequantize(
    w_q, scales,
    biases=None,
    group_size=16, bits=4,
    mode="nvfp4",
    global_scale=None,        # mx.array, fp32 scalar — optional
    dtype=None,
)
```

- **Input dtype to `mx.quantize`**: any float (bf16/fp16/fp32). Output `w_q` is `uint32` packed (8 × 4-bit values per word — see `_extra_repr` `in_dims *= 32 // self.bits`).
- **`scales`**: `uint8` array, one byte per 16-value group, holding an FP8-E4M3 (signed; spec compliance issue #2962 — see §3) for `nvfp4`. For `mxfp4`/`mxfp8` (group 32) it is FP8-E8M0 instead.
- **`global_scale`**: optional FP32 scalar fed into `dequantize` / `quantized_matmul`. MLX's own `mx.quantize` does **not** emit it (test confirmed: `python/tests/test_quantized.py:146` `w_q, scales = mx.quantize(w, mode="nvfp4")`). It's intended for ingesting NVIDIA-pre-quantized weights.

Sources: [`python/tests/test_quantized.py:113-161`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/python/tests/test_quantized.py#L113); [`mlx.core.dequantize` doc](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html).

## 3. Apple Silicon hardware: any FP4 tensor core?

**No.** Apple's M5 announcement (Oct 2025) advertises a "Neural Accelerator in each GPU core" exposed via Metal 4 Tensor APIs, but **does not** mention FP4. Public materials for M3 and M4 likewise have no FP4 claim. NVFP4 on Apple is implemented as Metal compute shaders that do an FP8-scale multiply and a 16-entry FP4 LUT, dispatched from `quantized.cpp`.

Source: Perplexity/Apple newsroom synthesis; corroborated by MLX issue #2962 stating MLX's nvfp4 *uses signed E4M3* (an MLX choice — Apple GPU has no opinion on it because there is no FP4 ALU). Bug filed against MLX itself, not the silicon.

## 4. Scale layout in memory

**Two separate row-major buffers**, not interleaved:

| Buffer    | dtype       | Shape                 | Comment                                                      |
|-----------|-------------|-----------------------|--------------------------------------------------------------|
| `weight`  | `uint32`    | `[..., K/8]`          | 8 packed E2M1 nibbles per word                               |
| `scales`  | `uint8`     | `[..., K/16]`         | one FP8-E4M3 byte per 16-value group                         |

The Metal kernel reads them with separate `device` pointers; see [`fp_quantized.h:30-38`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/kernels/fp_quantized.h#L30):

```cpp
template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return T(*(thread fp8_e4m3*)(&s));   // nvfp4
  } else {
    return T(*(thread fp8_e8m0*)(&s));   // mxfp4 / mxfp8
  }
}
```

PR #2979 ("Swizzle scales") in v0.30.3 reorders the scale buffer for matmul throughput but the *logical* layout is still "one byte per 16 weights". If a caller wants a global FP32, they pass it independently to `dequantize` / `quantized_matmul`.

## 5. Github permalink to MLX nvfp4 Metal kernel

Commit-pinned MSL sources:

- [`mlx/backend/metal/kernels/fp_quantized.h@1eef1d1`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/kernels/fp_quantized.h) — host-side dequant/qdot/qmm templates.
- [`mlx/backend/metal/kernels/fp_quantized.metal@1eef1d1`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/kernels/fp_quantized.metal) — kernel entry points.
- [`mlx/backend/metal/kernels/fp4.h@1eef1d1`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/kernels/fp4.h) — `fp4_e2m1` struct, packing/unpacking.
- [`mlx/backend/metal/kernels/fp8.h@1eef1d1`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/kernels/fp8.h) — `fp8_e4m3` and `fp8_e8m0` structs.
- [`mlx/backend/metal/kernels/fp_quantized_nax.{h,metal}@1eef1d1`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/kernels/fp_quantized_nax.h) — Apple "NAX" matrix-coprocessor variant (M4/M5 Neural Accelerator path).
- Host launcher: [`mlx/backend/metal/quantized.cpp@1eef1d1`](https://github.com/ml-explore/mlx/blob/1eef1d155c96cdb87fa8e7ff995facf8ca26d369/mlx/backend/metal/quantized.cpp).

PR landing page: <https://github.com/ml-explore/mlx/pull/2946>.

## 6. Recommended Frag representation for `tilelang/language/extern.py`

The user's proposed shape is **almost right**; correct it as:

```python
@dataclass
class NVFP4Frag:
    packed_e2m1:  Buffer   # dtype=uint32 (or uint8 packed), shape [..., K // 8]
    fp8_scale:    Buffer   # dtype=uint8 holding fp8_e4m3 bits, shape [..., K // 16]
    fp32_global: Optional["mx.array"] = None   # optional fp32 scalar, dequant-time only
```

Corrections vs. the user's initial draft:

1. **Packed buffer is `uint32` words of 8 nibbles** in MLX (see `_extra_repr` math `in_dims *= 32 // bits`). `int8`-as-storage works too (cast equivalent on Apple GPU) but `uint32` matches the kernel's load signature `const device uint8_t* w` (cast to `const device uint16_t*` for the 4-bit path).
2. **Scale dtype is bit-cast `uint8`**, *not* a true `float8_e8m0`. For `group_size=16` (nvfp4) MLX interprets it as `fp8_e4m3` (signed E4M3 — note issue #2962, but that is the current implementation). Use `float8_e8m0` only for `mxfp4`/`mxfp8` (group 32).
3. **Global FP32 is optional and not produced by `mx.quantize`.** It's a degree of freedom for ingesting NVIDIA NVFP4 weights; for Apple-side quantize→matmul it stays `None`.
4. **No interleaving.** Two parallel row-major buffers, both row-contiguous (PR #2941 enforces this).

A minimal Frag wrapper just needs the two buffers; expose `fp32_global` as a kw-only attribute, defaulting to `None`, threaded into `dequantize`/`quantized_matmul` calls only when present.

---

## Citations

- PR #2946 (`Metal/CPU nvfp4 and mxfp8`): <https://github.com/ml-explore/mlx/pull/2946> — merge commit `1eef1d1`.
- v0.30.3 release notes: <https://github.com/ml-explore/mlx/releases/tag/v0.30.3>.
- `mlx.nn.quantize` doc: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html>.
- `mlx.core.dequantize` doc: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html>.
- Issue #2962 (E4M3 vs UE4M3 scale spec): <https://github.com/ml-explore/mlx/issues/2962>.
- mlx-lm issue #717 (nvfp4 support): <https://github.com/ml-explore/mlx-lm/issues/717>.
- NVIDIA NVFP4 spec: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html>.
- Apple M5 / Neural Accelerator (Metal 4 Tensor APIs, no public FP4 claim): Apple Newsroom Oct 2025.
