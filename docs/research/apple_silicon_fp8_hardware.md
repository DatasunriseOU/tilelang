# Apple Silicon FP8 Hardware Availability Survey

**Date:** 2026-05-07
**Author:** research agent
**Purpose:** Decide whether to ship `simdgroup_a_fp8` / `simdgroup_b_fp8` SIMDgroup matrix factories targeting Apple GPUs now, or hold them as forward-looking placeholders.

## TL;DR

As of May 2026, **no shipping Apple Silicon GPU exposes native FP8 (E4M3 / E5M2) matrix-multiply intrinsics**, neither in the GPU's new "Neural Accelerators" (introduced on M5, October 2025) nor in the Apple Neural Engine. Apple's Metal 4 / MSL toolchain has no `float8` scalar type, no FP8 entries in `MPSDataType`, and no FP8 path in MPSGraph or Core ML. MLX, llama.cpp, and PyTorch MPS all either reject FP8 dtypes outright or emulate them via `uchar` packing. The realistic Apple low-precision stack today is **BF16 + INT8 + INT4 (and MXFP4 for MoE weights via MLX)**.

By contrast, NVIDIA H100 (Hopper, 2022) and AMD MI300X (CDNA3, 2023) both ship full hardware FP8 GEMM with documented WMMA-class intrinsics and PFLOPS-scale throughput.

**Recommendation:** ship the FP8 SIMDgroup factories with `TODO("Apple FP8 silicon")` and a static `unsupported_target` diagnostic on the Metal backend. Do not gate the IR-level type on hardware availability — but do gate codegen.

---

## 1. Per-chip FP8 capability matrix

| Chip | Released | GPU family | Native FP8 GEMM | Native BF16 | Native INT8 | Native MXFP4 | Source |
|------|----------|-----------|-----------------|-------------|-------------|--------------|--------|
| M3 / M3 Pro / M3 Max | Oct 2023 | Apple9 | **No** | Yes (M3+) | Yes | No | [pmetal hw matrix](https://github.com/Epistates/pmetal/blob/main/docs/hardware-support.md) |
| M3 Ultra | Mar 2025 | Apple9 | **No** (FP8 row marked "X" in published GPU spec tables) | Yes | Yes | No | perplexity research, citation [1] |
| M4 / M4 Pro / M4 Max | Oct/Nov 2024 | Apple9 (arch gen 16) | **No** ("Metal does not expose FP8 Tensor ops") | Yes | Yes (~38 TOPS ANE INT8) | No | [awesomeagents M4 Max spec](https://awesomeagents.ai/hardware/apple-m4-max/) |
| M5 (base) | Oct 15 2025 | Apple10 (arch gen 17) — adds NAX | **No** (Apple newsroom does not list FP8; HN consensus: FP16/BF16 + INT8 only) | Yes | Yes | Yes (MXFP4 via MLX) | [Apple M5 newsroom](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/), [HN 47406531](https://news.ycombinator.com/item?id=47406531) |
| M5 Pro / M5 Max | Mar 3 2026 | Apple10 + Fusion | **No** (same NAX architecture, no FP8 disclosure) | Yes | Yes | Yes | [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5) |
| A18 / A18 Pro | Sep 2024 | iPhone 16 | **No** (no FP8 in spec sheets or ANE docs) | Partial | Yes | No | perplexity, citation [6] |
| A19 (M5-class core) | Sep 2025 | iPhone 17 | **No** (shares M5 super-core lineage; no FP8 disclosed) | Yes | Yes | Yes | [Wikipedia M5 § A19](https://en.wikipedia.org/wiki/Apple_M5) |

**Headline finding (Hacker News thread 47406531, 2026):** *"Apple's hardware does not support FP8 (neither the ANE NPU, or the new 'neural accelerator' tensor cores), though the most recent variant supports INT8."*

The M5 announcement (Oct 15 2025) — Apple's biggest AI silicon push since the original Neural Engine — explicitly markets **"4× peak GPU compute … for AI workloads"** but never names FP8 as a supported format. Apple's own MLX research blog ([machinelearning.apple.com, Nov 19 2025](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)) benchmarks BF16 dense, 4-bit quantized, and **MXFP4** (MoE) — never FP8. If FP8 were a hardware feature, Apple would have headlined it.

## 2. Software stack availability

| Layer | FP8 support | Evidence |
|-------|-------------|----------|
| Metal 4 / MSL (Xcode 26, WWDC25) | **None.** MSL scalar types remain `float`, `half`, `bfloat`, integer/bool. No `float8_e4m3` / `float8_e5m2` types. | [Apple Metal-Shading-Language-Specification.pdf](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf); [Apple "Machine learning passes" doc](https://developer.apple.com/documentation/Metal/machine-learning-passes) — explicitly says tensors work with "common weight and input data types, such as `int8` and `fp16`" |
| `MTLTensorDataType` (Metal 4) | No FP8 cases (Apple docs page exists but enumeration not surfaced; cross-checked via PyTorch `c10/metal/utils.h` `vectypes` specializations — only `float/half/bfloat/short/int/long`). | [PyTorch c10/metal/utils.h](https://github.com/pytorch/pytorch/blob/e9ebbd3b/c10/metal/utils.h) |
| `MPSDataType` (MPS / MPSGraph) | Enum lists `float32`, `float16`, `bFloat16`, `int4`, `int8`, `int16/32/64`, `uInt4/8/16/32/64`, `complexFloat16/32`. **No `float8` / `fp8` / `e4m3` / `e5m2` entries.** | [Apple MPSDataType doc](https://developer.apple.com/documentation/MetalPerformanceShaders/MPSDataType) |
| Core ML / coremltools | Compute precision documented as FP16 (GPU/ANE) and FP32 (CPU); quantization is INT8 / INT4 weight-only. No FP8. | Apple "Typed Execution" docs (perplexity citation [3]) |
| MLX (Apple's array framework) | **Not supported.** Issue [#3341 "rocm/cuda/metal — bfloat8/float8 support"](https://github.com/ml-explore/mlx/issues/3341) (closed Apr 2026). MLX maintainer @zcbenz: *"I have no idea about float8 support in metal actually."* MoE quantization on M5 uses **MXFP4**, not FP8 ([Apple ML research blog](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). |
| PyTorch MPS backend | Hard-error: `RuntimeError: "mps" does not have support for that dtype` for `Float8_e4m3fn` / `Float8_e5m2`. Open issue [#132624](https://github.com/pytorch/pytorch/issues/132624), still feature-requested as of Dec 2025. |
| Third-party emulation | [`fp8-mps-metal`](https://github.com/ramedeiros/fp8-mps-metal) packs FP8 into `uchar` and dequantizes to FP16 inside a Metal kernel — explicitly states *"Metal Shading Language has no native 8-bit float type"* and benchmarks 4–26× speedup vs CPU fallback (still not native FP8 GEMM). |

### Metal 4 ML primitives (WWDC25)

WWDC25 sessions [205 "Discover Metal 4"](https://developer.apple.com/videos/play/wwdc2025/205/) and [262 "Combine Metal 4 ML and graphics"](https://developer.apple.com/videos/play/wwdc2025/262/) introduce `MTLTensor`, `MTL4MachineLearningCommandEncoder`, `tensor_inline`, and `cooperative_tensor` MSL types. Tensor operators include convolution, matrix multiplication, and reduction — but Apple consistently illustrates them with `int8`/`fp16`. No FP8 mention in any Metal 4 session.

## 3. FP8 format coverage on Apple

| Format | Bias | Apple HW | Apple SDK type |
|--------|------|----------|----------------|
| E4M3 (FN, no infs, ±448 max) | 7 | None | None |
| E5M2 (IEEE-style, infs/NaNs) | 15 | None | None |
| E8M0 (per-block scales, MXFP4/8) | 127 | **Indirect via MXFP4 weight format on M5+** (MLX runtime handles unpacking; the E8M0 *scale byte* is consumed by software, not by a hardware FP8 tensor unit) | None as a first-class scalar |

E8M0 is interesting: M5 + MLX support `MXFP4` MoE inference ([Apple research, Nov 2025](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)), and MXFP4 uses E2M1 elements with a shared E8M0 scale per 32-element block. But the *compute* still happens at FP16/BF16 inside the NAX matrix engine; the FP4/FP8 datapath does not exist.

## 4. Cross-reference: H100 and MI300X

**NVIDIA H100 (Hopper, SM 9.0, 2022).** Native FP8 Tensor Cores supporting both E4M3 (preferred for activations / weights) and E5M2 (preferred for gradients), per the Micikevicius et al. *"FP8 Formats for Deep Learning"* spec NVIDIA co-authored with Arm and Intel. Spec sheet quotes 3,958 TFLOPS FP8 (SXM, with sparsity) — roughly 2× FP16. Programmer interface: `wgmma.mma_async` PTX intrinsics, CUTLASS/cuBLASLt FP8 kernels, Transformer Engine, and PyTorch's `torch._scaled_mm` with `torch.float8_e4m3fn` / `torch.float8_e5m2`. Mature.

**AMD MI300X (CDNA3, 2023).** AMD product page documents 2.61 PFLOPS peak FP8 (E5M2 + E4M3), 5.22 PFLOPS with structured sparsity, via 1,216 Matrix Cores across 304 CUs. Exposed in ROCm via `<hip/hip_fp8.h>`, `__hip_fp8_e4m3` / `__hip_fp8_e5m2` types, and `rocWMMA` cooperative-matrix templates analogous to NVIDIA's `wmma::fragment`. Transformer Engine for ROCm and vLLM consume it directly. Mature, though tooling lags H100 by ~1 year.

Both contrast sharply with Apple: on Hopper/CDNA3 you write `wmma.mma.sync` / `__builtin_amdgcn_mfma_*_fp8_fp8` and the silicon executes FP8 dot-products in one cycle. On Apple Silicon (M3 → M5 Max) the equivalent path does not exist in any documented form.

## 5. Recommendation

**Ship `simdgroup_a_fp8` / `simdgroup_b_fp8` factories now, but treat them as IR-level types with no Metal backend lowering.**

Concretely:

1. **TIR / TVM dialect:** introduce the FP8 element types (E4M3, E5M2) as first-class scalar dtypes and let the SIMDgroup-matrix builder accept them. This is portable and mirrors what we'd ship for CUDA / HIP backends. No harm in having the types.
2. **Metal codegen:** emit a `static_assert(false, "FP8 SIMDgroup matmul not supported on this Apple GPU; use BF16 or INT8.")` (or runtime `MTLCompileError`) when the lowering pass sees an FP8 cooperative tensor target = Metal. Add the marker `TODO(apple-fp8)` next to the lowering hook.
3. **Capability probe:** wire a `Target::HasFP8MatrixIntrinsic()` query that returns `false` for every published Apple GPU family (Apple7 through Apple10 / NAX). Flip it to `true` only when Apple ships an MSL `float8_*` type or an `MTLTensorDataType.float8E4M3` enum case.
4. **Test matrix:** the FP8 factories should land with a CUDA H100 reference test (which we can actually run) and an Apple Metal **negative test** that asserts the diagnostic fires. This keeps the contract honest without pretending we have HW we don't.

The cost of waiting for Apple FP8 silicon is high: the type-system plumbing is the slow part, and the moment Apple ships `MTLTensorDataType.float8*` (plausibly Metal 5 / M6 in WWDC27) we want the IR ready. The cost of shipping today is essentially zero — a `TODO` and a clean diagnostic.

## 6. Sources (canonical URLs)

- Apple. *"Apple unleashes M5"* press release. Oct 15 2025. https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
- Apple Machine Learning Research. *"Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"*. Nov 19 2025. https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Apple Developer. *"Machine learning passes"*. https://developer.apple.com/documentation/Metal/machine-learning-passes
- Apple Developer. `MPSDataType` enum. https://developer.apple.com/documentation/MetalPerformanceShaders/MPSDataType
- Apple Developer. `MTLTensorDataType`. https://developer.apple.com/documentation/metal/mtltensordatatype
- Apple Developer. *"What's new in Metal"*. https://developer.apple.com/metal/whats-new/
- Apple Developer. WWDC25 session 205 "Discover Metal 4". https://developer.apple.com/videos/play/wwdc2025/205/
- Apple Developer. WWDC25 session 262 "Combine Metal 4 ML and graphics". https://developer.apple.com/videos/play/wwdc2025/262/
- Wikipedia. *"Apple M5"*. https://en.wikipedia.org/wiki/Apple_M5
- ml-explore/mlx#3341 "rocm/cuda/metal — bfloat8/float8 support". https://github.com/ml-explore/mlx/issues/3341
- ml-explore/mlx#3017 "NAX Split-K for large-K GEMM stability on M5". https://github.com/ml-explore/mlx/issues/3017
- pytorch/pytorch#132624 "Add support for float8 dtypes for the MPS backend". https://github.com/pytorch/pytorch/issues/132624
- Comfy-Org/ComfyUI#5533, #6995, #12202 — `Float8_e4m3fn` MPS rejection reports.
- ramedeiros/fp8-mps-metal. *"Metal Shading Language has no native 8-bit float type."* https://github.com/ramedeiros/fp8-mps-metal
- Epistates/pmetal hardware-support matrix (Apple7–Apple10 NAX). https://github.com/Epistates/pmetal/blob/main/docs/hardware-support.md
- Hacker News 47406531. *"Apple's hardware does not support FP8."* https://news.ycombinator.com/item?id=47406531
- awesomeagents.ai. *"Apple M4 Max — FP8 Support: No (Metal does not expose FP8 Tensor ops)."* https://awesomeagents.ai/hardware/apple-m4-max/
- NVIDIA Hopper architecture whitepaper (FP8 Tensor Cores, E4M3/E5M2). https://www.nvidia.com/hopper-architecture-whitepaper
- Micikevicius et al. *"FP8 Formats for Deep Learning."* arXiv:2209.05433.
- AMD Instinct MI300 CDNA3 ISA reference. https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- ROCm Documentation, "Data types and precision support." https://rocmdocs.amd.com/en/latest/reference/precision-support.html
