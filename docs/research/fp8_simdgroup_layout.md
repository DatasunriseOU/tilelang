# Apple MSL FP8 simdgroup-matrix layout — authoritative survey

Author: agent research, 2026-05-07
Scope: determine whether Apple has published per-thread register-tile
layout constants for FP8 (`float8_e4m3` / `float8_e5m2`) in
`simdgroup_matrix` (or in the Metal 4 cooperative-tensor `matmul2d`
pathway) so that tilelang can implement `simdgroup_a_fp8` /
`simdgroup_b_fp8` factories analogous to the FP16 ones.

## TL;DR

**Apple has not published any FP8 simdgroup-matrix or cooperative-tensor
layout constants.** As of MSL Specification **Version 4
(2025-10-23)** [1] and Metal Feature Set Tables **2025-06-05** [2]:

- The MSL scalar-type table (Table 2.1) lists only `half`, `bfloat`,
  `float`. There is no `float8`, `float8_e4m3`, `float8_e5m2`, `fp8`,
  `e4m3`, or `e5m2` keyword anywhere in the 346-page spec [1].
- `simdgroup_matrix<T, Cols, Rows>` is defined in §2.4 with
  `T ∈ {half, bfloat, float}` and `Cols == Rows == 8` [1, p.38].
- The new Metal 4 `matmul2d` cooperative-tensor pathway (§7.x, Tables
  7.3 and 7.4) explicitly enumerates *every* supported A/B/C type
  combination — only `char`, `half`, `bfloat`, `float` appear; no FP8
  rows in any combination, including the **OS 26.1+** additions [1,
  pp.317–318].
- Mapping of matrix elements to threads in the SIMD-group is
  **explicitly unspecified**: "The mapping of matrix elements to
  threads in the SIMD-group is unspecified" [1, §2.4, p.38, line 2403
  of the extracted text].
- `metal::float8_e4m3` / `metal::float8_e5m2` are **not** public MSL
  types and do not appear in any Apple framework header that surfaces
  in MSL.

Recommendation: **do not commit numerical layout constants.** Implement
`simdgroup_a_fp8` / `simdgroup_b_fp8` as forward-compat placeholder
layouts that explicitly raise / warn at codegen time (or are gated
behind a target-version check), with a TODO referencing this document.

## Authoritative sources

### [1] Metal Shading Language Specification, Version 4 (2025-10-23)

- URL: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Cover page: "Metal Shading Language Specification — Version 4". Footer
  on every page: "2025-10-23 | Copyright © 2025 Apple Inc."
- Relevant sections verified by direct PDF extraction
  (`pdftotext` of the canonical Apple URL on 2026-05-07):

  - **§2.1 Scalar Data Types, Table 2.1** (p.25): floating-point
    scalars are `half`, `bfloat`, `float`. No FP8 entries.
  - **§2.4 SIMD-group Matrix Data Types** (p.38, lines 2383–2407
    of the extracted text, quoted verbatim):

    > "Metal supports the following SIMD-group matrix type names,
    > where T is half, bfloat (in Metal 3.1 and later) or float and
    > Cols and Rows are 8: simdgroup_half8x8, simdgroup_bfloat8x8
    > (Metal 3.1 and later), simdgroup_float8x8."
    >
    > "The mapping of matrix elements to threads in the SIMD-group is
    > unspecified."

  - **§6.7 SIMD-Group Matrix Functions / §6.7.2 Matrix Operations**
    (pp.192–193): documents `simdgroup_load`, `simdgroup_store`,
    `simdgroup_multiply`, `simdgroup_multiply_accumulate` purely in
    terms of generic `simdgroup_matrix<T, Cols, Rows>`. No
    per-thread register-tile constants are given (none exist in the
    abstract ISA — Apple deliberately keeps the lane-mapping
    opaque). No FP8 overloads.

  - **§7 Tensor Operations / Table 7.3 MatMul2D data type
    supported** (p.317): the complete list of A/B/C combinations is
    `{char,half,float}` for A and B with `{int, half, float}`
    accumulators. No FP8 rows.
  - **Table 7.4 Additional MatMul2D data types supported in OS 26.1
    and later** (p.318): adds bfloat × bfloat, bfloat × half, bfloat
    × char, etc. **Still no FP8 rows.**

A grep on the full extracted text confirms zero hits for `fp8`,
`float8`, `e4m3`, `e5m2` (case-insensitive).

### [2] Metal Feature Set Tables (2025-06-05)

- URL: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- "SIMD-scoped matrix multiply operations" row: requires Metal 4 +
  Apple7 family (i.e. M1 and later). "Tensors" and "Machine learning
  encoding" rows: Metal 4 + Apple7. **No row** for any of FP8 / float8
  / e4m3 / e5m2 / "low-precision matmul" / similar; the document is
  silent on FP8 across every Apple GPU family (Apple7 = M1 through
  Apple10 = M5/A19).

### [3] System headers — `MetalPerformancePrimitives.framework`

The `mpp::tensor_ops::matmul2d` template in
`/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2dImpl.h`
contains a hard `static_assert` enforcing exactly the type list of
Tables 7.3/7.4. The assertion has been observed to fire on Apple M5 +
macOS 26.3.1 when llama.cpp / ollama tried mismatched combinations [3a,
3b]:

> static_assert failed due to requirement
> '__tensor_ops_detail::__is_same_v<bfloat, half>'
> "Input types must match cooperative tensor types"

This independently corroborates that no FP8 path exists even
internally in the shipping framework — there is no FP8 specialization
the kernel could fall through to.

- [3a] https://github.com/ggml-org/llama.cpp/issues/17986 (closed
  2025-12-31, llama.cpp Metal4 tensor backend on M5)
- [3b] https://github.com/ollama/ollama/issues/15862 (open
  2026-04-28, same framework error on M5 / macOS 26.3.1)

### [4] Ecosystem corroboration — FP8 emulation on MPS

Multiple OSS projects ship FP8 on Apple Silicon **only** via custom
shaders that pack two `uchar` into a half / float and decode in MSL
manually, explicitly because there is no native type:

- `tashiscool/fp8-mps-metal` ("FP8 Metal compute kernels for Apple
  Silicon MPS — fixing what PyTorch doesn't support yet"):
  > "Metal Shading Language has no native 8-bit float type. PyTorch's
  > MPS backend never implemented the cast or compute kernels for
  > FP8."
  https://github.com/ramedeiros/fp8-mps-metal (fork; tested on M4
  Pro, macOS 26.2, PyTorch 2.10) — repo provides 4 hand-written Metal
  kernels: `fp8_scaled_matmul_kernel`, `fp8_scaled_vecmat_kernel`,
  `fp8_to_half_kernel`, `float_to_fp8_kernel`. No use of
  `simdgroup_matrix`; all FP8 arithmetic is done through manual
  decode → fp16 multiply.
- PyTorch MPS backend status: `torch.float8_e4m3fn` /
  `torch.float8_e5m2` raise `"mps" does not have support for that
  dtype` and `_scaled_mm not implemented for MPS`.

### [5] Slang's Metal target (cross-check)

Slang's "Metal-Specific Functionalities" page makes no mention of FP8,
float8, e4m3, e5m2, or simdgroup_matrix at all
(https://shader-slang.org/slang/user-guide/metal-target-specific.html).
Slang lowers cooperative-matrix intrinsics only to the documented
`simdgroup_matrix` set.

## Per-question answers

| Question | Answer |
|---|---|
| Per-thread register tile shape for `simdgroup_matrix<fp8_e4m3, M, K>` on M3/M4 | **Spec is silent.** The type does not exist in MSL, and even for the types that do, lane-to-element mapping is "unspecified" by §2.4 [1]. |
| Per-thread register tile shape for `simdgroup_matrix<fp8_e5m2, K, N>` | **Spec is silent.** Same reason. |
| Accumulator type for FP8 matmul | **Not defined.** No FP8 overload of `simdgroup_multiply_accumulate` exists; no FP8 row in MatMul2D Tables 7.3/7.4. |
| Apple Silicon chips supporting FP8 simdgroup matmul | **None publicly.** Feature Set Tables list no FP8 row for any family; M5 (Apple10) headers still `static_assert` against non-{half,bfloat,float}. |
| Does MSL expose `simdgroup_matrix<half, 8, 8>` directly? | Yes: `simdgroup_half8x8` is a public alias [1, §2.4]. |
| Does MSL expose `simdgroup_matrix<float8_e4m3, ?, ?>` or `metal::matmul()` for FP8? | **No.** `metal::matmul` is not an MSL builtin; the Metal 4 path uses `mpp::tensor_ops::matmul2d`, whose template parameters reject FP8 at compile time. |
| Upper bounds on (M, N, K) for simdgroup_matrix | Always 8×8×8 for the simdgroup pathway [1, §2.4]. The Metal 4 `matmul2d` cooperative-tensor pathway accepts much larger M/N/K (parameterized by the descriptor) but not for FP8 element types. |
| Is `metal::float8_e4m3` an MSL type? | **No.** Not in MSL spec, not in `MTLDataType`, not in any public Apple shader header. Apple ML frameworks (Core ML / BNNS / MPSGraph quantize-dequantize) handle FP8 only at the framework level, not as an MSL scalar type. |

## MSL code-snippet status

There is no canonical example because no public API exists. The
closest reference (FP16) from the spec for context [1, §6.7.2 example,
p.193]:

```metal
kernel void float_matmad(device float *pMatA, device float *pMatB,
                         device float *pMatC, device float *pMatR) {
    simdgroup_float8x8 sgMatA, sgMatB, sgMatC, sgMatR;
    simdgroup_load(sgMatA, pMatA);
    simdgroup_load(sgMatB, pMatB);
    simdgroup_load(sgMatC, pMatC);
    simdgroup_multiply_accumulate(sgMatR, sgMatA, sgMatB, sgMatC);
    simdgroup_store(sgMatR, pMatR);
}
```

Replacing `simdgroup_float8x8` with `simdgroup_float8_e4m3_8x8` (or any
plausible spelling) **does not compile** against any current SDK.

## Recommendation for `tilelang/language/extern.py`

1. **Do not invent layout constants.** Any number we put in for the
   per-thread register tile (e.g. `num_threads=32` with `(2, 1)`
   per-thread elements) would be a guess about hardware Apple has not
   exposed.

2. Implement `simdgroup_a_fp8` / `simdgroup_b_fp8` as **explicit
   placeholders** that:
   - share the same `Frag(... layout="simdgroup_a_fp8")` declaration
     surface as the FP16 ones (so user code is forward-compatible),
   - resolve at codegen time to a stub that **raises a clear
     `NotImplementedError`** referencing this document (path:
     `docs/research/fp8_simdgroup_layout.md`), and
   - are documented in the docstring as "reserved name awaiting Apple
     to publish FP8 simdgroup layout; current MSL spec v4 (2025-10-23)
     does not define `simdgroup_matrix<float8_*, 8, 8>` and Tables
     7.3/7.4 contain no FP8 rows".

3. When Apple does publish FP8 (likely first surfacing as additional
   rows in Table 7.4 of a future spec, **not** as a new
   `simdgroup_float8*8x8` alias), revisit:
   - The most likely first-appearance pathway is `mpp::tensor_ops::matmul2d`
     with FP8 cooperative-tensor element types, rather than the legacy
     `simdgroup_matrix` 8×8 abstraction.
   - At that point, lane-layout is *still* likely to be unspecified
     (consistent with §2.4's policy across all element types), so
     tilelang should keep the abstract Frag interface even after FP8
     lands.

4. As an interim, if a user must run FP8 GEMMs on Apple Silicon today,
   the only correct lowering is **decode-to-half in shared memory and
   issue an FP16 `simdgroup_multiply_accumulate`**, matching the
   pattern used by `tashiscool/fp8-mps-metal`. This is a separate
   codegen path from the `simdgroup_*_fp8` factories and should be
   labeled accordingly.

## Confidence and conflicting sources

No conflicting sources were found. Every primary source (Apple PDF spec
v4, Apple Feature Set Tables, MetalPerformancePrimitives system
headers as observed via compile-time `static_assert` failures) and
every secondary OSS source (Slang, fp8-mps-metal, llama.cpp, ollama,
PyTorch issues) agree: **FP8 is not a public MSL scalar type, has no
simdgroup_matrix specialization, and has no MatMul2D entry as of
2026-05-07.**

If Apple has internal FP8 paths in the Neural Engine or in private
Core ML kernels, those are not reachable from MSL and thus irrelevant
to tilelang's MSL codegen.

---

Word count: ~1,180.
