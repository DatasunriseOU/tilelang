import tilelang.testing
from tilelang import carver
from tilelang import tvm
from tilelang.language import dtypes as T
from tilelang.carver.arch import CUDA, METAL, auto_infer_current_arch
from typing import List


def run_general_reduction_recommend_hints(structure: str = "SSR", shape: List[int] = None, dtype: T.dtype = T.float16, topk: int = 20):
    arch = auto_infer_current_arch()
    carve_template = carver.GeneralReductionTemplate(
        structure=structure,
        shape=shape,
        dtype=dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=topk)
    assert len(hints) > 0, "Hints length is zero"


def _arch_without_runtime(cls, target, platform, bandwidth):
    arch = object.__new__(cls)
    arch.target = tvm.target.Target(target)
    arch.platform = platform
    arch.warp_size = 32
    arch.sm_partition = 4
    arch.transaction_size = [32, 128]
    arch.smem_cap = 32 * 1024
    arch.max_smem_usage = 2 * arch.smem_cap
    arch.reg_cap = 65536
    arch.compute_max_core = 1
    arch.bandwidth = bandwidth
    arch.l2_cache_size_bytes = 0
    arch.compute_capability = "metal" if platform == "METAL" else "80"
    arch.sm_version = 80
    return arch


def _first_sr_1024_hint(arch):
    return carver.GeneralReductionTemplate(
        structure="SR",
        shape=[1024, 1024],
        dtype="float32",
    ).with_arch(arch).recommend_hints(topk=1)[0]


def test_metal_row_reduce_1024_prefers_128_reduce_threads():
    hint = _first_sr_1024_hint(
        _arch_without_runtime(METAL, "metal", "METAL", [750, 1200])
    )

    assert hint.block == [1]
    assert hint.thread == [1]
    assert hint.rstep == [1024]
    assert hint.reduce_thread == [128]


def test_cuda_row_reduce_1024_hint_order_is_unchanged():
    hint = _first_sr_1024_hint(
        _arch_without_runtime(
            CUDA, {"kind": "cuda", "arch": "sm_80"}, "CUDA", [750, 12080]
        )
    )

    assert hint.block == [128]
    assert hint.thread == [128]
    assert hint.rstep == [32]
    assert hint.reduce_thread == [1]


@tilelang.testing.requires_cuda_target
def test_general_reduction_recommend_hints():
    run_general_reduction_recommend_hints("SSR", [1024, 1024, 1024], T.float16)
    run_general_reduction_recommend_hints("SS", [1024, 1024], T.float16)
    run_general_reduction_recommend_hints("SRS", [1024, 1024, 1024], T.float16)


def run_elementwise_recommend_hints(shape: List[int] = None, dtype: T.dtype = T.float16, topk: int = 20):
    arch = auto_infer_current_arch()
    carve_template = carver.ElementwiseTemplate(
        shape=shape,
        dtype=dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=topk)
    assert len(hints) > 0, "Hints length is not topk"


@tilelang.testing.requires_cuda_target
def test_elementwise_recommend_hints():
    run_elementwise_recommend_hints([1024, 1024], T.float16)
    run_elementwise_recommend_hints([1024], T.float16)
    run_elementwise_recommend_hints([1024, 1024, 1024], T.float16)


def run_matmul_recommend_hints(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    in_dtype: T.dtype = T.float16,
    out_dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float16,
):
    arch = auto_infer_current_arch()
    carve_template = carver.MatmulTemplate(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=20)
    assert len(hints) > 0, "Hints length is not 20"


@tilelang.testing.requires_cuda_target
def test_matmul_recommend_hints():
    run_matmul_recommend_hints(1024, 1024, 1024, T.float16, T.float16, T.float16)
    run_matmul_recommend_hints(1024, 1024, 1024, T.int8, T.int32, T.int32)
    run_matmul_recommend_hints(1024, 1024, 1024, T.float16, T.float32, T.float16)


def run_gemv_recommend_hints(
    N: int = 1024, K: int = 1024, in_dtype: T.dtype = T.float16, out_dtype: T.dtype = T.float16, accum_dtype: T.dtype = T.float16
):
    arch = auto_infer_current_arch()
    carve_template = carver.GEMVTemplate(
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=20)
    assert len(hints) > 0, "Hints length is not 20"


@tilelang.testing.requires_cuda_target
def test_gemv_recommend_hints():
    run_gemv_recommend_hints(1024, 1024, T.float16, T.float16, T.float16)
    run_gemv_recommend_hints(1024, 1024, T.int8, T.int32, T.int32)
    run_gemv_recommend_hints(1024, 1024, T.float16, T.float32, T.float16)


def run_fmha_recommend_hints(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_length: int = 512,
    seq_kv_length: int = 512,
    head_dim: int = 128,
    in_dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float16,
    out_dtype: T.dtype = T.float16,
):
    arch = auto_infer_current_arch()
    carve_template = carver.FlashAttentionTemplate(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_length=seq_length,
        seq_kv_length=seq_kv_length,
        head_dim=head_dim,
        in_dtype=in_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=20)
    for hint in hints:
        print(hint)
    assert len(hints) > 0, "Hints length should be greater than 0"


@tilelang.testing.requires_cuda_target
@tilelang.testing.requires_cuda_compute_version_eq(8, 0)
def test_fmha_recommend_hints():
    run_fmha_recommend_hints(4, 32, 512, 512, 128, T.float16, T.float16, T.float16)
    run_fmha_recommend_hints(4, 32, 512, 512, 128, T.int8, T.int32, T.int32)


if __name__ == "__main__":
    tilelang.testing.main()
