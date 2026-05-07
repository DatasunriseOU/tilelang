<img src=./images/logo-row.svg />

<div align="center">

# Tile Language
[![PyPI version](https://badge.fury.io/py/tilelang.svg)](https://badge.fury.io/py/tilelang)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/TUrHyJnKPG)
[![Puzzles](https://img.shields.io/badge/🧩_Learn-TileLang_Puzzles-blueviolet)](https://github.com/tile-ai/tilelang-puzzles)
</div>

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

## Apache TVM Migration Work Note

This checkout is being migrated to the native Apache TVM API line in `3rdparty/tvm` and should be pushed as a whole to `https://github.com/DatasunriseOU/tilelang`, including the TVM submodule pin. The purpose is not to preserve TileLang-only TVM fork shims indefinitely, but to adapt TileLang's C++ passes, runtime hooks, Python lowering path, and tests to the upstream Apache TVM API drift as directly and reviewably as possible.

Reference trees such as `/private/tmp/cppmega-mlx-tilelang-stack-c` and `/private/tmp/tl_pr_c` may be used to understand the older working TileLang+TVM behavior and local patches. When the old fork disagrees with the current Apache design, the migration should prefer the new Apache API shape and keep compatibility wrappers narrow, documented, and removable.

The working bar for this migration is a clean local build plus the relevant TileLang test suite passing against the bundled Apache TVM checkout. Any remaining incompatibility should be documented with the exact failing command, error, and API boundary before the branch is considered ready.

### Vendored TVM Compatibility Surfaces

This branch keeps the Apache TVM API shape as the target and only vendors the TileLang-era pieces that are still needed to bridge old IR into the new pipeline. The temporary `src/transform/vendored/allocate.*` and `src/transform/vendored/let_stmt.*` nodes model legacy body-carrying `Allocate` and `LetStmt`; `lower_allocate.cc` and `lower_let_stmt.cc` must lower them to Apache-native `AllocBuffer + SeqStmt` and `Bind + SeqStmt` before strict Apache TIR passes run.

Because Apache `StmtFunctor` dispatch tables cannot be extended globally after finalization, local passes that may still see vendored allocation nodes use explicit guarded traversal such as `allocate_visit_passthrough.h` instead of relying on default visitors. The global-barrier compatibility path is similarly narrow: `global_barrier_builtin.cc` and `tl_runtime_symbols.h` reintroduce only the IR/codegen symbols TileLang still emits while the runtime backing is being reconciled with upstream TVM.

The broader compatibility shims should stay small and removable. `tl_attr.h` and `tl_compat.h` carry local attr keys, launch-parameter tags, old DLTensor field names, object/type aliases, and the legacy allocation namespace bridge. `CMakeLists.txt` wires the migration by building vendored sources, linking the Apache TVM compiler target, force-including `tl_compat.h`, disabling the old recursive TVM submodule pin path, and using the PyPI Z3 flow expected by current Apache TVM.

Pipeline order is part of the contract. `LowerTileLangLetStmt` must run before `LowerTileLangAllocate`, and both must execute before Apache TIR passes in lower/legalize, target optimization, and host/device codegen. If a new drift exposes `tilelang.Allocate` or `tilelang.LetStmt` inside an Apache pass, fix the lowering boundary or add a local guarded traversal; do not broaden the vendored nodes into a second permanent TVM dialect.

### 2026-05-06 — Pulled improvements from cppmega-mlx-tilelang-stack-c and tl_pr_c

Five changes brought into this branch from the `/private/tmp/cppmega-mlx-tilelang-stack-c` and `/tmp/tl_pr_c` reference trees, plus one defensive fix:

1. **Blockscaled e8m0 layout** (`tilelang/language/blockscaled_layout.py` + `testing/python/cpu/test_blockscaled_e8m0_layout.py`). New `BlockScaledLayout.e8m0_k32()` and `e8m0_to_float()` for the Sparse-MLA Path C blockscaled FP8 reducer. 14 CPU unit cases pass at HEAD.

2. **Hybrid `tilelang/language/fp8_op.py`** (431 → 938 LOC). Combines tl_pr_c's macro structure (separate `_fp8_scaled_matmul_m1_vecmat_metal_direct_macro` for direct global store and a fallback `_macro_legacy` using only `tirx.metal.simd_sum`) with stack-c's row-pointer indexing (`B_fp8[col, 0]` + `word_i` instead of flat `B_fp8[0, 0]` + computed `b_word_i`). Adds `metal_fp8_e4m3_dot4`, `_target_thread_warp_size`, `_resolve_target`, `_normalize_block_scale_layout`, `_block_scale_value`, and the trans-B direct metal macro.

3. **Conditional FP8/FP4 helper prelude** (`src/target/codegen_metal.{cc,h}`). Hybrid of tl_pr_c granular per-dtype emit methods (`EmitFp8E3M4Helper`, `EmitFp8E4M3Helper`, `EmitFp8E4M3FnAliasHelper`, `EmitFp8E4M3FnuzHelper`, `EmitFp8E4M3B11FnuzHelper`, `EmitFp8E5M2Helper`, `EmitFp8E5M2FnuzHelper`, `EmitFp8E8M0FnuHelper`, `EmitFp8Dot4Helpers`) and stack-c switch-style dispatch in `EmitFPHelperPrelude`. Reduces vecadd float32 MSL from 9.2 KB to 586–724 bytes (-92 %). Helpers are emitted only for dtypes/intrinsics actually referenced by the kernel (collected by a `StmtExprVisitor` walking BufferLoad / BufferStore / Cast / Call / Allocate / Broadcast / Ramp / Var / Let / params / buffer_map).

4. **`src/transform/lower_access_ptr.cc` correctness fix.** Pulled stack-c's `ExtractAccessPtrBaseLoad` helper which tolerates `if_then_else(cond, BufferLoad(...), fallback)` produced by `LegalizeSafeMemoryAccess`. Replaces the prior hard `Downcast<BufferLoad>` that crashed on wrapped loads.

5. **Z3Prover hooks on Apache `Analyzer`.** The submodule pin bumps `3rdparty/tvm` to a checkout that adds three callback hooks on `tvm::arith::Analyzer` (`BindExprHook`, `BindRangeHook`, `EnterConstraintHook`) so external sub-analyzers can be auto-driven from the `Analyzer::Bind` and `ConstraintContext::EnterWithScope` paths without a circular library dependency. `src/transform/vendored/z3_prover.cc` registers the three hooks at static init via a `Z3HookRegistrar`, so every TileLang `analyzer.Bind(iv->var, range)` and every `with ConstraintContext(analyzer, expr):` automatically forwards to the per-Analyzer Z3Prover instance. This restores the partial-sync optimisation path that the upstream Apache `Analyzer` would otherwise leave starved of constraints.

Defensive fix in `tilelang/jit/adapter/torch/metal.py`: extract trailing `'x'`/`'y'`/`'z'` axis only and skip silently for thread tags that do not end in a recognised axis letter (e.g. `threadIdx.__wmma_x` keeps working; `threadIdx.something_else` no longer raises `ValueError`).

Test summary at HEAD: 36 CPU + 61 Metal + 3 skipped (opt-in benchmarks) = 100 / 103 pass, 0 fail. Cold compile time 0.015 s (≈39× faster than the tl_pr_c baseline). Bench parity vs `/tmp/tl_pr_c`: swap mean 0.984× of tl_pr_c on the metal benchmarks (within MPS variance).

### 2026-05-07 — Z3 prover safety gate for CUDA / gb10

A correctness regression has been observed on the gb10 (CUDA `sm_120`) target
when the vendored Z3 prover (`src/transform/vendored/z3_prover.{h,cc}`) is
allowed to short-circuit conservative paths in `AutoDoubleBuffer`,
`DropProvableBoundChecks`, intra-warp barrier-elide, and related transforms.
The root cause has not yet been bisected. Until it is, CUDA / gb10 users
should opt out of the prover via the env gate:

```bash
export TILELANG_DISABLE_Z3=1
```

When set to any non-empty value other than `"0"`, every
`tilelang::tlz3::Z3Prover::CanProve` call returns `false`, which keeps the
slow / conservative path in every consumer pass. Default behavior (env
unset) is unchanged, so Mac / Apple-Silicon builds still benefit from the
real prover (≈39× cold-compile speedup, 0.984× swap perf parity vs
`/tmp/tl_pr_c`). The gate lives at the top of `Z3Prover::CanProve`
(`src/transform/vendored/z3_prover.cc`) and is read once per process via a
function-local `static`. Remove the env once the gb10 regression is
root-caused and a targeted fix lands.

#### Per-pass Z3 gates (granular bisection, 2026-05-07)

The blanket `TILELANG_DISABLE_Z3` is a hammer; once the regression is
narrowed to a specific pass, you can re-enable Z3 everywhere except the
suspect site by *unsetting* the global gate and setting only the per-pass
env var. The Python side honours these via `os.getenv`; the C++ side
honours them via `tilelang::tlz3::Z3PassGate::IsEnabled(...)` (declared in
`src/transform/vendored/z3_prover.h`, implemented in `z3_prover.cc`).

| Env var | Pass | Roadmap idea | Source file |
|---|---|---|---|
| `TILELANG_DISABLE_Z3_VECTORIZE` | Loop vectorize (alignment + unit-stride proofs) | #1, #12 | `src/transform/loop_vectorize.cc` |
| `TILELANG_DISABLE_Z3_PREDICATE_FUSION` | Predicate fusion (b-cond / inner-body well-defined) | #7 | `src/transform/predicate_fusion.cc` |
| `TILELANG_DISABLE_Z3_DROP_BOUND_CHECKS` | Drop provable bound checks (BV32 fallback) | #4 | `src/transform/drop_provable_bound_checks.cc` |
| `TILELANG_DISABLE_Z3_TMA_LEGALITY` | TMA cp.async stride-aligned-16 proof | #6 | `src/op/copy.cc` |
| `TILELANG_DISABLE_Z3_BARRIER_ELISION` | Intra-warp barrier elision (RAW proof) | #11 | `src/transform/thread_storage_sync.cc` |
| `TILELANG_DISABLE_Z3_AUTO_DOUBLE_BUFFER` | Auto double buffer (reserved; stub mode currently has no live Z3 call) | #2 | `src/transform/auto_double_buffer.cc` |
| `TILELANG_DISABLE_Z3_INT24` | int8 dot4 int24 overflow proof | #5 | `tilelang/analysis/int24_overflow_proof.py` |
| `TILELANG_DISABLE_Z3_DOT4_LEGALITY` | FP8 packed-dot4 legality proof | #10 | `tilelang/language/fp8_op.py` |
| `TILELANG_DISABLE_Z3_SIMDGROUP` | Metal simdgroup eligibility / simd-lift | #8, #9 | `tilelang/transform/metal_fragment_to_simdgroup.py`, `tilelang/transform/metal_simd_lift.py` |

Truthiness convention matches the global gate: an env var is "set" when its
value is non-empty AND not `"0"`. Unset / `""` / `"0"` all mean *enabled*.
The blanket `TILELANG_DISABLE_Z3` remains as a backstop — even if a future
pass forgets to call `Z3PassGate::IsEnabled`, setting the global env still
disables it via the kill-switch at `Z3Prover::CanProve`.

Example bisection workflow (CUDA / gb10):

```bash
unset TILELANG_DISABLE_Z3
# Try disabling each pass in turn until the regression disappears.
export TILELANG_DISABLE_Z3_VECTORIZE=1   # idea #1/#12
# python -m pytest …  (run the failing test)
unset TILELANG_DISABLE_Z3_VECTORIZE
export TILELANG_DISABLE_Z3_DROP_BOUND_CHECKS=1   # idea #4
# python -m pytest …
```

<img src=./images/MatmulExample.png />

## Latest News
- 02/02/2026 🧩: Check out [TileLang Puzzles](https://github.com/tile-ai/tilelang-puzzles), a fun and interactive way to learn TileLang programming with 10 progressively harder puzzles!
- 12/18/2025 🚀: Added [CuTeDSL backend](https://github.com/tile-ai/tilelang/pull/1421) support, enabling compilation to NVIDIA CUTLASS CuTe DSL! Join us in building and optimizing this exciting new backend: [Issue #1454](https://github.com/tile-ai/tilelang/issues/1454).
- 12/17/2025 🔬: Integrated [Z3 theorem prover](https://github.com/tile-ai/tilelang/pull/1367) into TVM Arith Analyzer, bringing SMT-based symbolic reasoning for enhanced optimizations and automatic correctness verification!
- 10/31/2025 🔧: Migrated to [apache-tvm-ffi](https://github.com/tile-ai/tilelang/pull/1108), significantly reducing CPU overhead!
- 10/30/2025 📦: We have released v0.1.6.post2, which is the last version compatible with Python 3.8.
- 10/07/2025 🍎: Added Apple Metal Device support, check out [Pull Request #799](https://github.com/tile-ai/tilelang/pull/799) for details.
- 09/29/2025  🎉: Thrilled to announce that ​​AscendC​​ and ​Ascend​NPU IR​​ backends targeting Huawei Ascend chips are now supported!
Check out the preview here:
🔗 [link](https://github.com/tile-ai/tilelang-ascend).
This includes implementations across two branches:
[ascendc_pto](https://github.com/tile-ai/tilelang-ascend) and
[npuir](https://github.com/tile-ai/tilelang-ascend/tree/npuir).
Feel free to explore and share your feedback!
- 07/04/2025 🚀: Introduced `T.gemm_sp` for 2:4 sparse tensor core support, check out [Pull Request #526](https://github.com/tile-ai/tilelang/pull/526) for details.
- 06/05/2025 ✨: Added [NVRTC Backend](https://github.com/tile-ai/tilelang/pull/461) to significantly reduce compilation time for cute templates!
- 04/14/2025 🚀: Added high-performance FlashMLA implementation for AMD MI300X, achieving performance parity with hand-optimized assembly kernels of Aiter! See [example_mla_amd](./examples/deepseek_mla/amd/README.md) for details.
- 03/03/2025 🚀: Added high-performance MLA Decoding support using only 80 lines of Python code, achieving performance on par with FlashMLA on H100 (see [example_mla_decode.py](./examples/deepseek_mla/example_mla_decode.py))! We also provide [documentation](./examples/deepseek_mla/README.md) explaining how TileLang achieves this.
- 02/15/2025 ✨: Added WebGPU Codegen support, see [Pull Request #86](https://github.com/tile-ai/tilelang/pull/86)!
- 02/12/2025 ✨: Excited to announce the release of [v0.1.0](https://github.com/tile-ai/tilelang/releases/tag/v0.1.0)!
- 02/10/2025 🚀: Added debug tools for TileLang—`T.print` for printing variables/buffers ([docs](https://tilelang.com/tutorials/debug_tools_for_tilelang.html)) and a memory layout plotter ([examples/plot_layout](./examples/plot_layout)).
- 01/20/2025 ✨: We are excited to announce that tile-lang, a dsl for high performance AI workloads, is now open source and available to the public!

## Tested Devices
Although tile-lang aims to be portable across a range of Devices, it has been specifically tested and validated on the following devices: for NVIDIA GPUs, this includes the H100 (with Auto TMA/WGMMA support), A100, V100, RTX 4090, RTX 3090, and RTX A6000; for AMD GPUs, it includes the MI250 (with Auto MatrixCore support) and the MI300X (with Async Copy support).

## OP Implementation Examples
**tile-lang** provides the building blocks to implement a wide variety of operators. Some examples include:

- [Matrix Multiplication](./examples/gemm/)
- [Dequantization GEMM](./examples/dequantize_gemm/)
- [Flash Attention](./examples/flash_attention/)
- [Flash Linear Attention](./examples/linear_attention/)
- [Flash MLA Decoding](./examples/deepseek_mla/)
- [Native Sparse Attention](./examples/deepseek_nsa/)

Within the `examples` directory, you will also find additional complex kernels—such as convolutions, forward/backward passes for FlashAttention, more operators will continuously be added.

## Benchmark Summary

TileLang achieves exceptional performance across a variety of computational patterns. Comprehensive benchmark scripts and settings are available at [tilelang-benchmark](https://github.com/tile-ai/tilelang-benchmark). Below are selected results showcasing its capabilities:

- MLA Decoding Performance on H100

  <div style="display: flex; gap: 10px; justify-content: center;">
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs64_float16.png" alt="mla decode performance bs64 on H100" width="100%" />
    </div>
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs128_float16.png" alt="mla decode performance bs128 on H100" width="100%" />
    </div>
  </div>

- Flash Attention Performance on H100

  <div align="center">    <img src="./images/mha_performance_h100.png" alt="operator performance on H100" width=80% />
  </div>

- Matmul Performance on GPUs (RTX 4090, A100, H100, MI300X)

  <div>
    <img src="./images/op_benchmark_consistent_gemm_fp16.png" alt="gemm fp16 performance on Gpus" />
  </div>

- Dequantize Matmul Performance on A100

  <div>
    <img src="./images/op_benchmark_a100_wq_gemv.png" alt="dequantize gemv performance on A100" />
  </div>

## Installation
### Method 1: Install with Pip

The quickest way to get started is to install the latest release from PyPI:

```bash
pip install tilelang
```

Alternatively, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/tile-ai/tilelang
```

Or install locally:

```bash
# install required system dependencies
sudo apt-get update
sudo apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

pip install -e . -v # remove -e option if you don't want to install in editable mode, -v for verbose output
```

### Method 2: Build from Source
We currently provide three ways to install **tile-lang** from source:
- [Install from Source (using your own TVM installation)](./docs/get_started/Installation.md#method-1-install-from-source-using-your-own-tvm-installation)
- [Install from Source (using the bundled TVM submodule)](./docs/get_started/Installation.md#method-2-install-from-source-using-the-bundled-tvm-submodule)
- [Install Using the Provided Script](./docs/get_started/Installation.md#method-3-install-using-the-provided-script)

### Method 3: Install with Nightly Version

For users who want access to the latest features and improvements before official releases, we provide nightly builds of **tile-lang**.

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly
# or pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
```

> **Note:** Nightly builds contain the most recent code changes but may be less stable than official releases. They're ideal for testing new features or if you need a specific bugfix that hasn't been released yet.

## Quick Start

In this section, you'll learn how to write and execute a straightforward GEMM (matrix multiplication) kernel using tile-lang, followed by techniques for layout optimizations, pipelining, and L2-cache–friendly swizzling.

### GEMM Example with Annotations (Layout, L2 Cache Swizzling, and Pipelining, etc.)

Below is an example that demonstrates more advanced features: layout annotation, parallelized copy, and swizzle for improved L2 cache locality. This snippet shows how to adapt your kernel to maximize performance on complex hardware.

```python
# @tilelang.jit(target="cuda")
# target currently can be "cuda" or "hip" or "cpu".
# if not specified, it will be inferred from the input tensors during compile time
@tilelang.jit
def matmul_relu(
    A, B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    # declare compilation shape constant
    M, N, K = T.const('M, N, K')

    # annotate input tensor shape
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]

    # allocate output tensor
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Enable rasterization for better L2 cache locality (Optional)
        # T.use_swizzle(panel_size=10, enable=True)

        # Clear local accumulation
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # Copy tile of A
            # This is a sugar syntax for parallelized copy
            T.copy(A[by * block_M, ko * block_K], A_shared)

            # Copy tile of B
            T.copy(B[ko * block_K, bx * block_N], B_shared)

            # Perform a tile-level GEMM on the shared buffers
            # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
            T.gemm(A_shared, B_shared, C_local)

        # relu
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        # Copy result back to global memory
        T.copy(C_local, C[by * block_M, bx * block_N])

    # You can write multiple cuda kernel in one function, they execute sequentially
    # with T.Kernel(...) as ...

    # Return the tensor, you can also return multiple tensors
    return C


M, N, K = 1024, 1024, 1024

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c_ref = torch.relu(a @ b)

# Call the kernel
c = matmul_relu(a, b)
torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)

# Call the kernel with overwritten compilation constants
c = matmul_relu(a, b, block_M=128, block_N=128, block_K=64)
torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)

# Retrieve the compiled kernel
kernel = matmul_relu.compile(a, b) # use torch.Tensor
kernel = matmul_relu.compile(      # use T.Tensor as placeholder
  T.Tensor((M, K), T.float16),
  T.Tensor((K, N), T.float16)
)
kernel = matmul_relu.compile(      # directly specify the shape constants
  M=M, N=N, K=K,
  block_M=128, block_N=128, block_K=64
)
print(kernel.get_kernel_source())
c = kernel(a, b)

# 5.Profile latency with kernel
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
```

### Dive Deep into TileLang Beyond GEMM

In addition to GEMM, we provide a variety of examples to showcase the versatility and power of TileLang, including:

- [Dequantize GEMM](./examples/dequantize_gemm/): Achieve high-performance dequantization by **fine-grained control over per-thread operations**, with many features now adopted as default behaviors in [BitBLAS](https://github.com/microsoft/BitBLAS), which utilizing magic layout transformation and intrins to accelerate dequantize gemm.
- [FlashAttention](./examples/flash_attention/): Enable cross-operator fusion with simple and intuitive syntax, and we also provide an example of auto tuning.
- [LinearAttention](./examples/linear_attention/): Examples include RetNet and Mamba implementations.
- [Convolution](./examples/convolution/): Implementations of Convolution with IM2Col.

## Upcoming Features

Check our [tilelang v0.2.0 release plan](https://github.com/tile-ai/tilelang/issues/79) for upcoming features.

---

TileLang has now been used in project [BitBLAS](https://github.com/microsoft/BitBLAS) and [AttentionEngine](https://github.com/microsoft/AttentionEngine).

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration!

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/TUrHyJnKPG)

## Acknowledgments

We would like to express our gratitude to the [TVM](https://github.com/apache/tvm) community for their invaluable contributions. The initial version of this project was mainly developed by [LeiWang1999](https://github.com/LeiWang1999), [chengyupku](https://github.com/chengyupku) and [nox-410](https://github.com/nox-410) with supervision from Prof. [Zhi Yang](https://yangzhihome.github.io) at Peking University. Part of this work was carried out during an internship at Microsoft Research, where Dr. Lingxiao Ma, Dr. Yuqing Xia, Dr. Jilong Xue, and Dr. Fan Yang offered valuable advice and support. We deeply appreciate their mentorship and contributions.
