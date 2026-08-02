# TileLang Technical Debt & Backlog (TODO_PLAN1.md)

## Group 1: Python Torch Dynamo Integration (`poc/torch_dynamo/`)
- [x] Task 1: Wire fusion patterns and emitters into orchestrator. `poc/torch_dynamo/_fusion_patterns.py`. Connect the dedicated emitter and fusion patterns into the orchestrator's specialized and canonical paths.
- [x] Task 2: Implement Philox RNG for training-mode dropout. `poc/torch_dynamo/_kernels/flash_attention.py` & `poc/torch_dynamo/fx_to_tilelang.py`. Verify philox_seed and philox_offset dtype/shape and emit a proper RNG path to handle training=True states.
- [x] Task 3: Verify PyTorch AOT Autograd version compatibility. `poc/torch_dynamo/aot_autograd_glue.py`. Test and verify the canonical 2.11+ path and the exact shipped PyTorch versions for AOT autograd glue logic.
- [x] Task 4: Support multi-input fused chains. `poc/torch_dynamo/fx_to_tilelang.py`. Implement routing to extern to support complex fused chains involving 3+ inputs and multiple binary operations.
- [x] Task 5: Analyze dataflow for payload handles. `poc/torch_dynamo/fx_to_tilelang.py`. Implement necessary dataflow analysis to properly resolve and track payload[0] and payload[1] handles during FX to TileLang translation.

## Group 2: Python Triton Frontend (`poc/triton_frontend/`)
- [x] Task 6: Implement `map_tt_func` PrimFunc shell. `poc/triton_frontend/__init__.py`. Implement a real `map_tt_func` that correctly builds the PrimFunc shell for the frontend.
- [x] Task 7: Add operator conformance tests. `poc/triton_frontend/conformance/__init__.py`. Implement missing conformance tests for softmax, matmul, layer_norm, fa_v2, fa_v3, and paged_attn.
- [x] Task 8: Refactor memory emitters and pointer analysis. `poc/triton_frontend/op_mapping.py`. Replace legacy load/store stubs with PtrAnalysis-derived buffers and indices, and consolidate emitters once Path C is active.
- [x] Task 9: Verify barrier and op emitter mappings. `poc/triton_frontend/op_mapping.py`. Verify individual op emitters and confirm that `barrier_arrive` and `barrier_wait` remain supported in `tilelang.language`.
- [x] Task 10: Re-enable disabled frontend tests. `poc/triton_frontend/tests/test_dot_reduce_atomic.py`. Re-enable dot reduce atomic tests by wrapping them in the `tilelang.builder` context and resolve related numeric smoke test issues.

## Group 3: Python Examples & Benchmark (`examples/`)
- [x] Task 11: Fix Warp Specialized Pass Errors. `examples/flash_decoding/example_gqa_decode.py`, `examples/gdn/example_chunk_o_bwd.py`. Resolve issues with the warp specialized pass, including errors caused by reduce_sum with clear=True.
- [x] Task 12: Fix Reduce Operation Bugs. `examples/gdn/example_wy_fast_bwd_split.py`, `examples/gdn/example_chunk_o_bwd.py`. Correct the reduce implementation to handle operations where dim != -1 and support reducing a whole buffer to a scalar.
- [x] Task 13: Optimize and Standardize BitNet 1.58b. `examples/bitnet-1.58b/modeling_bitnet.py`. Remove legacy Flash Attention checks for RoCm, fix transpose inefficiencies, handle dynamic sequence lengths, and standardize the Cache interface.
- [x] Task 14: Improve Layout Inference for Int8 Loads. `examples/dequantize_gemm/example_dequant_gemm_fine_grained.py`. Enhance the Layout Inference Pass to efficiently handle four-dimensional int8 loads.
- [x] Task 15: Fix TMA Convolution and MHA Causal Split. `examples/convolution/test_example_convolution.py`, `examples/flash_decoding/example_mha_inference.py`. Fix TMA support for convolutions and implement handling for the causal split case in MHA inference.

## Group 4: Python Language & Carver (`tilelang/`)
- [x] Task 16: Improve Infrastructure and Architecture Inference. `tilelang/autotuner/tuner.py`, `tilelang/carver/arch/__init__.py`. Create a common logger in utils and replace the temporary architecture inference solution.
- [x] Task 17: Enhance CUDA Architecture Modeling. `tilelang/carver/arch/cuda.py`. Consider input dtypes, explore static shared memory improvements, and implement a method to determine real memory bandwidth.
- [x] Task 18: Refactor Matmul Analysis Logic. `tilelang/carver/analysis.py`, `tilelang/carver/matmul_analysis.py`. Distinguish GEMV from reduction, integrate specific logic into policies, and analyze based on bits and schedule index_dtype.
- [x] Task 19: Clean up TensorCore Policy Variables. `tilelang/carver/roller/hint.py`, `tilelang/carver/roller/policy/tensorcore.py`. Rename use_tc, remove or utilize block reduction depth, and dynamically set offsets using tags.
- [x] Task 20: Optimize TensorCore Policy Algorithms. `tilelang/carver/roller/policy/tensorcore.py`. Optimize the all_steps enlarge policy to be a multiple of the original and investigate shared memory connection cases.

## Group 5: Python Testing (`testing/`)
- [x] Task 21: Resolve ROCm and HIP compatibility and precision issues. `testing/python/kernel/test_tilelang_kernel_gemm.py` etc. Fix GEMM precision on ROCm and implement missing support for variable allocation, initializers, and loop unroll factors on HIP/ROCm platforms to re-enable skipped tests.
- [x] Task 22: Improve symbolic divisible check in arithmetic logic. `testing/python/arith/test_arith_iter_affine_map.py`. Enhance the arithmetic module's symbolic divisible checking capabilities to re-enable the currently disabled affine map iter tests.
- [x] Task 23: Expand language feature support for atomics and memory reshape. `testing/python/language/test_tilelang_language_atomic.py`, `testing/python/language/test_tilelang_language_reshape.py`. Add support for half-precision types in `atomic_addx4` and enable `reshape` operations to be correctly applied to shared memory buffers.
- [x] Task 24: Complete transform pass soundness and intrinsic logic. `testing/python/transform/test_auto_double_buffer.py` etc. Implement real soundness obligations for double buffering, add the documented factory wiring for `simdgroup_*` intrinsics, and support gather loops for `im2col` to prevent silent unsafe rewrites.
- [x] Task 25: Restore Hopper intrinsic lowering verifications. `testing/python/transform/test_tilelang_transform_lower_hopper_intrin.py`. Revisit and re-enable the temporarily removed test checks for lowering Hopper intrinsics to ensure the transform operates safely and correctly.

## Group 6: C++ Transforms (`src/transform/`)
- [x] Task 26: Implement Auto Double Buffer Soundness. `src/transform/auto_double_buffer.cc`. Implement the real soundness obligation checks for auto double buffering.
- [x] Task 27: Enhance Vectorization Checks and Pass Ordering. `src/transform/loop_vectorize.cc`, `src/transform/vectorize_loop.cc`. Improve vectorization validation, add negative_ramp codegen support, and adjust pass ordering.
- [x] Task 28: Refactor Layout Inference and Validation. `src/transform/layout_inference.cc`, `src/transform/layout_reducer.cc`. Phase out buffer_map, address missing thread mappings, and implement metadata validation checks for layout reducers.
- [x] Task 29: Improve Hardware-Specific Lowering. `src/transform/lower_blackwell_2sm.cc`, `src/transform/lower_tma_to_ptr_arith.cc`, `src/transform/pipeline_planning.cc`. Add mixed 1cta/2cta tcgen5mma support, emit conv2d-with-padding gathers, and link wgmma to buffers.
- [x] Task 30: Cleanup Workarounds and Refactor Storage. `src/transform/flatten_buffer.cc`, `src/transform/pipeline_planning.cc`, `src/transform/storage_rewrite.cc`, `src/transform/thread_storage_sync.cc`. Relocate boolean handling, refactor pipeline ops, apply deferred storage rewrite checks, and remove thread count workarounds.

## Group 7: C++ Ops and Codegen (`src/op/`, `src/target/`)
- [x] Task 31: tcgen05.cp Support and Dynamic Shared Memory Buffer Remapping. `src/op/copy.cc`. Add support for tcgen05.cp in conjunction with LowerTmemCopy and address buffer remapping for shared.dyn when is_cp is true.
- [x] Task 32: Smarter Shared Memory Box Dimension Deduction. `src/op/copy.cc`. Find a more robust and intelligent method to deduce the shared memory box dimensions.
- [x] Task 33: Type Info Retention and TF32 Workaround Cleanup. `src/target/codegen_cuda.cc`, `src/target/codegen_metal.cc`. Remove the temporary type workaround for TF32 and implement a unified way to keep type information directly in the AST.
- [x] Task 34: Vectorized Reduction and Ramp Lanes Optimization. `src/target/codegen_cuda.cc`. Implement vectorized reduction for various data types and revisit the ramp lanes limit logic.
- [x] Task 35: Target-Specific Builtin and Atomic Operation Updates. `src/target/codegen_hip.cc`, `src/target/codegen_metal.cc`. Update the HIP backend to use __builtin_amdgcn_s_barrier() and implement atomic operations for floating-point datatypes on Metal via Compare-And-Swap (CAS).

## Group 8: C++ Headers (`src/op/*.h`, `src/tl_templates/*.h`, `src/transform/common/*.h`)
- [x] Task 36: Remove redundant code. `src/op/gemm_sp_py.h`. Deduplicate and remove redundant code shared with gemm.h.
- [x] Task 37: Expand TCGEN5MMA support and shapes. `src/op/tcgen5_meta.h`, `src/tl_templates/cuda/gemm_sm100.h`. Support more shapes and dtypes for TCGEN5MMA, add 2cta-preferred shapes, address saturation issues, and implement gemm_ts.
- [x] Task 38: Optimize SM80/SM90 GEMM templates. `src/tl_templates/cuda/gemm_sm90.h`, `src/tl_templates/cuda/gemm_sp_sm80.h`. Move bar.sync out of body_rs in SM90 and implement the unsupported feature in SM80 SP GEMM.
- [x] Task 39: Add ROCm shfl_sync support. `src/tl_templates/hip/common.h`. Implement support for shfl_sync using features provided in ROCm 7.1.1.
- [x] Task 40: Fix transform pass ordering and naming conflicts. `src/transform/common/loop_vectorization_utils.h`, `src/transform/common/mbarrier.h`. Move the loop vectorization pass to the correct prior stage and rename the mbarrier identifier to avoid conflicts with user-defined variables.
