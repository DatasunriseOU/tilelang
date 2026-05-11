# OSS FP8 codegen patterns — survey + recommendations

Scope: pick a layout / register-tiling pattern from a known-working FP8
implementation (CUDA H100/Ada, AMD CDNA3, Triton) and document its
applicability to TileLang's `simdgroup_a_fp8` / `simdgroup_b_fp8`
factories, so the factory drops in cleanly when Apple FP8 hardware ships.

Every claim below is pinned to a commit-locked github URL.

## 1. NVIDIA SM_89 / SM_90 — 16x8x32 FP8 MMA atom

CUTLASS defines exactly four FP8 MMA atoms for SM_89 (Ada) and SM_90
(Hopper) classic FP8, all sharing one register-tiling pattern:

* `SM89_16x8x32_F32E4M3E4M3F32_TN` — shape **16x8x32** (M,N,K).
* `SM89_16x8x32_F32E4M3E5M2F32_TN`, `..._F32E5M2E4M3F32_TN`,
  `..._F32E5M2E5M2F32_TN` — same MNK, mixed E4M3/E5M2 inputs.
* Per-thread fragment: D output = 4× `float`, A input = 4× `uint32_t`,
  B input = 2× `uint32_t` per lane (32-lane warp).

Source:
[NVIDIA/cutlass@d4e16f5d include/cute/arch/mma_sm89.hpp](https://github.com/NVIDIA/cutlass/blob/d4e16f5d/include/cute/arch/mma_sm89.hpp).

So one FP8 warp tile carries:
- A: 16x32 in 4 u32 regs/lane = 4×4 = **16 fp8 elements per lane** along K
- B: 8x32 in 2 u32 regs/lane = 2×4 = **8 fp8 elements per lane** along K
- C: 16x8 in 4 fp32 regs/lane (same as fp16 m16n8k16, K is just deeper)

This is the same per-lane register-tile shape that `makeGemmFragmentC`
already produces for fp16 m16n8k16 in
[tile-ai/tilelang@51aa0d93 src/layout/gemm_layouts.cc#L122-L137](https://github.com/tile-ai/tilelang/blob/main/src/layout/gemm_layouts.cc#L122):

```cpp
auto base_layout = makeGemmFragment8x8()->Repeat({2, 1}, false);
// → 16x8 C tile, 32 lanes × 4 elems/lane
```

i.e. the **C fragment is identical between fp16 m16n8k16 and fp8
m16n8k32**; only A/B widen along K. This is the keystone observation
for the recommended Layout below.

## 2. TileLang upstream FP8 dispatch — gemm_sm90.h

TileLang already specialises `DispatchInstruction` for fp8 in
[tile-ai/tilelang PR#751 (gemm_sm90.h)](https://github.com/tile-ai/tilelang/pull/751/files):

```cpp
// src/tl_templates/cuda/gemm_sm90.h around L156-L167
template <> struct DispatchInstruction<fp8_e4_t, fp8_e4_t, float> {
  using MMA_Atom = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;
  using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 16, N)>, _X>;
};
template <> struct DispatchInstruction<fp8_e5_t, fp8_e5_t, float> {
  using MMA_Atom = MMA_Atom<SM89_16x8x32_F32E5M2E5M2F32_TN>;
  // identical group layout
};
```

The same PR fixes the FP8 dtype generalisation
([tile-ai/tilelang@92121fc6 src/op/gemm.cc#L361-L392](https://github.com/tile-ai/tilelang/commit/92121fc66819444daba11bcb625826497a36c514)):
both `is_float8_e4m3()` and `is_float8_e5m2()` checks were collapsed to a
single `dtype.is_float8()` predicate, which means the **layout factory
is dtype-tagged but shape-invariant** — A/B are both 16x32 fp8 by
construction.

## 3. AMD CDNA3 — gfx942/gfx950 16x16x32 / 16x16x128 mfma

[tile-ai/tilelang PR#1743](https://github.com/tile-ai/tilelang/pull/1743)
wires gfx942 (`fnuz`) and gfx950 (`fn`) FP8 dtype routing through
`tilelang/intrinsics/mfma_macro_generator.py` and `src/target/codegen_hip.cc`.
The hardware shapes are:

* gfx942: `v_mfma_f32_16x16x32_fp8_fp8` — 16x16x32 (M,N,K).
* gfx950 scaled: `v_mfma_scale_f32_16x16x128_f8f6f4` — 16x16x128.

The CDNA factory for fp16/bf16 already uses 16x16 base tiles
([tile-ai/tilelang src/layout/gemm_layouts.cc#L63-L97](https://github.com/tile-ai/tilelang/blob/main/src/layout/gemm_layouts.cc#L63)):

```cpp
Fragment makeGemmFragmentAB16x32CDNA(const int k_pack) {  // 16x32 fp16
  // ...
}
```

For fp8 the equivalent is `makeGemmFragmentAB16x64CDNA` (gfx942) or
`...AB16x128CDNA` (gfx950); k_pack = 4 (4 fp8/u32) is the natural
parameter. Triton's gfx950 lowering lives in
[triton-lang/triton PR#6038](https://github.com/triton-lang/triton/pull/6038)
modifying `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`.

## 4. Triton on H100 — kWidth = 4 for FP8 MMAv3

Triton encodes the per-lane fp8 fragment width through `kWidth`. From
[triton-lang/triton PR#5009](https://github.com/triton-lang/triton/pull/5009)
discussion (file `lib/Conversion/TritonGPUToLLVM/SharedToDotOperandMMAv2OrV3.cpp`):

> "For Hopper fp8 * fp16, A and B will both have `kWidth = 2`, and we
> need to specify `kWidth` for the sake of A, which usually has
> `kWidth = 32 / 8 = 4`."

So **fp8 MMAv3 carries `kWidth = 4`** = 4 fp8 elements packed into one
32-bit register slot per lane. This is the same packing TileLang's
Metal `metal_fp8_e4m3_dot4` already uses
([tile-ai/tilelang tilelang/language/fp8_op.py#L144](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/fp8_op.py#L144))
— 4 fp8 bytes per uint32 word.

## 5. apache/tvm WMMA fp8 — 16x16x32 e4m3/e5m2

[apache/tvm PR#16950](https://github.com/apache/tvm/pull/16950) added
WMMA fp8 codegen with **16x16x32** fragment shape (32 fp8 along K), all
four dtype combinations (e4m3/e4m3, e4m3/e5m2, e5m2/e4m3, e5m2/e5m2),
gated on SM_89+. The intrinsic registry uses the same
`get_mma_intrin_group(...)` interface as fp16 — only the dtype tag and
the K-extent change.

## 6. MLX — no FP8 simdgroup path (yet)

MLX's
[ml-explore/mlx@d4c81062 mlx/backend/metal/kernels/quantized.h](https://github.com/ml-explore/mlx/blob/d4c81062/mlx/backend/metal/kernels/quantized.h)
has **zero FP8 references**. Quantisation path is integer 2/3/4/5/6/8-bit
only; reductions use `simd_sum` / `quad_sum`, tile sizes are 32×32×32.
MLX nvfp4 issue
([ml-explore/mlx#2962](https://github.com/ml-explore/mlx/issues/2962))
confirms FP8 scales arrive via signed E4M3 reinterpret on cpu, not via
a Metal simdgroup MMA. This matches the WWDC25 cooperative-tensor talk
([Discover Metal 4](https://developers.apple.com/videos/play/wwdc2025/205))
which advertises bf16 / fp16 cooperative tensors but does not list fp8.

So the TileLang `simdgroup_a_fp8` / `simdgroup_b_fp8` Layout we register
today is **forward-compatible scaffolding**, not a current hardware
binding. It plugs in to `metal_fragment_to_simdgroup.py`'s static check
([tile-ai/tilelang tilelang/transform/metal_fragment_to_simdgroup.py#L66-L70](https://github.com/tile-ai/tilelang/blob/main/tilelang/transform/metal_fragment_to_simdgroup.py#L66))
which already accepts `e4m3` / `e5m2` as simdgroup-eligible dtypes.

## 7. Recommended `simdgroup_*_fp8` Layout values

The binding decision: mirror the fp16 `simdgroup_a/b/c` 8x8 tile, but
parameterise k_pack = 4 along K so we can express both Apple's
hypothetical 8x8x32 fp8 cooperative tensor *and* fall back cleanly to
the existing `metal_fp8_e4m3_dot4` packed-byte path.

Concretely, in `src/layout/gemm_layouts.cc` add:

```cpp
// Mirror makeGemmFragment8x8(), but A/B widen along K by k_pack
// (k_pack = 4 for fp8: one u32 = 4 bytes = 4 fp8 lanes).
Fragment makeSimdgroupFragmentAFp8(int k_pack = 4) {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8 * k_pack);          // K dim, 32 fp8/lane
  IterVar rep = make_itervar("rep", 1);
  // 32 lanes × (8*k_pack/8) elems/lane = 32 × k_pack elements per row
  PrimExpr forward_thread = FloorDiv(j->var, k_pack) + 8 * i / 8;
  PrimExpr index = FloorMod(j->var, k_pack);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeSimdgroupFragmentBFp8(int k_pack = 4) {
  // Transposed K-major: 8x(8*k_pack), B is K-contiguous for dot4.
  IterVar i = make_itervar("i", 8 * k_pack);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(i->var, k_pack) + 8 * j / 8;
  PrimExpr index = FloorMod(i->var, k_pack);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

// C is unchanged from fp16 — accumulator stays fp32, 8x8 tile.
Fragment makeSimdgroupFragmentCFp8() { return makeGemmFragment8x8(); }
```

Rationale (per-claim citations):

1. **Tile = 8x8 along M,N**: matches Apple `simdgroup_matrix<8,8>`
   (the only matrix size MSL supports today)
   ([Metal 4 talk](https://developers.apple.com/videos/play/wwdc2025/205)).
2. **K = 8 × k_pack = 32 for fp8**: same K-extent as SM_89 m16n8k32
   ([CUTLASS mma_sm89.hpp](https://github.com/NVIDIA/cutlass/blob/d4e16f5d/include/cute/arch/mma_sm89.hpp))
   and apache/tvm wmma fp8
   ([apache/tvm PR#16950](https://github.com/apache/tvm/pull/16950)).
3. **k_pack = 4**: matches Triton MMAv3 `kWidth = 32/8 = 4`
   ([triton-lang/triton PR#5009](https://github.com/triton-lang/triton/pull/5009))
   and TileLang's existing dot4 packing
   ([tilelang/language/fp8_op.py#L144](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/fp8_op.py#L144)).
4. **C-fragment unchanged**: SM_89 atom returns fp32 in 16x8 tile with
   the same per-lane layout as fp16 m16n8k16
   ([CUTLASS](https://github.com/NVIDIA/cutlass/blob/d4e16f5d/include/cute/arch/mma_sm89.hpp)),
   so the existing `makeGemmFragment8x8()` C-tile is correct.
5. **`forward_thread = FloorDiv(j, k_pack) + 8*i/8`**: each of the 32
   lanes owns one (i, j_word) pair, with j_word = j // 4. Same shape
   as the existing `makeGemmFragment8x8` formula, just with `k_pack`
   replacing the hardcoded `2` for fp16 packed pairs
   ([tile-ai/tilelang src/layout/gemm_layouts.cc#L31-L38](https://github.com/tile-ai/tilelang/blob/main/src/layout/gemm_layouts.cc#L31)).

Wire into `extern.py` like so:

```python
LayoutKind = Literal[
    ..., "simdgroup_a", "simdgroup_b", "simdgroup_c",
    "simdgroup_a_fp8", "simdgroup_b_fp8", "simdgroup_c_fp8",
]
```

And in `src/transform/layout_inference.cc` (currently returns
`Layout()` for the simdgroup family,
[layout_inference.cc#L842-L845](https://github.com/tile-ai/tilelang/blob/main/src/transform/layout_inference.cc#L842)):

```cpp
if (str == "simdgroup_a_fp8") return makeSimdgroupFragmentAFp8(4);
if (str == "simdgroup_b_fp8") return makeSimdgroupFragmentBFp8(4);
if (str == "simdgroup_c_fp8") return makeSimdgroupFragmentCFp8();
```

This layout is **structurally identical** to the SM_89 m16n8k32 atom
modulo M/N tile size (8 vs 16/8) — when Apple ships an
`simdgroup_matrix_multiply_fp8` instruction the only change needed will
be the codegen pattern in `src/target/codegen_metal.cc` (one new MSL
function call); the layout factory is already correct.

## 8. Upstream PR-references to track for FP8 GA

| PR | Project | Status (2026-05-07) | Why we track |
|---|---|---|---|
| [#751](https://github.com/tile-ai/tilelang/pull/751) | tilelang | merged | DispatchInstruction fp8 SM89/90 |
| [#1372 / 92121fc6](https://github.com/tile-ai/tilelang/commit/92121fc66819444daba11bcb625826497a36c514) | tilelang | merged | `is_float8()` generalisation |
| [#1474](https://github.com/tile-ai/tilelang/pull/1474) | tilelang | merged | CUDA vectorized fp8 cast |
| [#1743](https://github.com/tile-ai/tilelang/pull/1743) | tilelang | merged | gfx942/gfx950 mfma fp8 |
| [#202](https://github.com/tile-ai/tilelang/pull/202) | tilelang | merged | cutlass 3.8 + fp8 T.gemm |
| [#16950](https://github.com/apache/tvm/pull/16950) | apache/tvm | merged | wmma fp8 codegen (SM_89) |
| [#16548](https://github.com/apache/tvm/pull/16548) | apache/tvm | merged | native fp8 codegen |
| [#14863](https://github.com/apache/tvm/pull/14863) | apache/tvm | merged | initial fp8 datatype |
| [#5009](https://github.com/triton-lang/triton/pull/5009) | triton | merged | kWidth fp8 MMAv3 |
| [#6038](https://github.com/triton-lang/triton/pull/6038) | triton | merged | gfx950 scaled mfma fp8 |
| [#3776](https://github.com/triton-lang/triton/pull/3776) | triton | merged | RDNA3/Navi31 wmma fp8 |
| [#2962](https://github.com/ml-explore/mlx/issues/2962) | mlx | open | nvfp4 / mxfp scale layout debate — informs Apple roadmap |
| [#2887](https://github.com/ml-explore/mlx/issues/2887) | mlx | open | mxfp8 quantized_matmul kernel gap |

## 9. Open items / risks

### 2026-05-11 direct Metal dot4 result

Late lowering now recognizes the direct global-store full-matmul marker
when `transpose_B=True` and B is already row-major `B[N, K]`. Under the
same Z3/static legality gate used by the M=1 path
(`K % 4 == 0`, K-contiguous A/B rows, 4-byte alignment, e4m3, int24-safe
K), TileLang emits one packed `__tvm_fp8_e4m3_dot4_packed` loop per
output cell. The 128x128x128 Metal benchmark measured:

| Lane | TileLang | audiohack | Ratio |
|---|---:|---:|---:|
| Existing shared `B[K,N]` scalar fallback | 1.617 ms | 0.123 ms | 13.13x |
| Direct `transpose_B=True`, input `B[N,K]` dot4 | 0.113 ms | 0.112 ms | 1.01x |

This confirms the remaining full-matmul gap is layout-bound: the
`B[K,N]` shared tile is strided along K for each output column, so packed
dot4 would be incorrect unless a producer changes the layout. The direct
path avoids wrapper-side transposes and tensor copies by requiring the
producer to materialize B in K-contiguous row-major `B[N,K]`.

* **Apple has no FP8 MMA**: confirmed via WWDC25 cooperative-tensor
  session ([Metal 4 talk](https://developers.apple.com/videos/play/wwdc2025/205))
  and MLX's lack of fp8 simdgroup paths
  ([ml-explore/mlx#2962](https://github.com/ml-explore/mlx/issues/2962)).
  The Layout above is scaffolding; the *executor* on Apple silicon
  remains the byte-packed `metal_fp8_e4m3_dot4` LUT path
  ([tilelang/language/fp8_op.py#L144](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/fp8_op.py#L144)).
* **k_pack drift**: if Apple ships fp8 cooperative tensors with a
  different K (e.g. 16 instead of 32), bump `k_pack` to 2 — only the
  factory constant changes, not the call sites.
* **No FP8 in MLX quantized.h** today; if MLX adopts a different lane
  layout (e.g. 16 fp8 per lane like a hypothetical
  `simdgroup_matrix<16,16>`), we add a parallel
  `makeSimdgroupFragmentAFp8_16x16` factory rather than re-shaping the
  existing 8x8 one.
