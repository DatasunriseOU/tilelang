import tilelang
import tilelang.testing
import tilelang.language as T
import pytest
from tilelang import tvm
from tilelang.engine.phase import PreLowerSemanticCheck


def _run_fragment_loop_checker(jit_func):
    func = jit_func.get_tir()
    PreLowerSemanticCheck(tvm.IRModule({func.attrs["global_symbol"]: func}))


@tilelang.jit
def simple_invalid_loop(dtype: T.dtype = T.bfloat16, accum_dtype: T.dtype = T.float32, num_threads: int = 128):
    A = T.dynamic("A")

    @T.prim_func
    def main(
        data: T.Tensor((128, A), dtype),  # type: ignore
    ):
        with T.Kernel(128, threads=num_threads) as (tid,):
            data_frag = T.alloc_fragment([128], accum_dtype)

            for i in T.Parallel(128):
                if i < A:
                    data_frag[i] = data[tid, i]

            for i in T.Parallel(A):
                data_frag[i] = 0

    return main


@tilelang.jit
def nested_invalid_loop(dtype: T.dtype = T.bfloat16, accum_dtype: T.dtype = T.float32, num_threads: int = 128):
    A = T.dynamic("A")

    @T.prim_func
    def main(
        data: T.Tensor((128, A), dtype),  # type: ignore
    ):
        with T.Kernel(128, threads=num_threads) as (tid,):
            data_frag = T.alloc_fragment([128], accum_dtype)

            for i in T.Parallel(128):
                if i < A:
                    data_frag[i] = data[tid, i]

            for i in T.Parallel(A // 64):
                for j in T.Parallel(64):
                    data_frag[i * 64 + j] = 0

    return main


@tilelang.jit
def invalid_loop_with_complex_dataflow(dtype: T.dtype = T.bfloat16, accum_dtype: T.dtype = T.float32, num_threads: int = 128):
    A = T.dynamic("A")

    @T.prim_func
    def main(
        data: T.Tensor((128, A), dtype),  # type: ignore
    ):
        with T.Kernel(128, threads=num_threads) as (tid,):
            data_frag = T.alloc_fragment([128], accum_dtype)

            for i in T.Parallel(128):
                if i < A:
                    data_frag[i] = data[tid, i]

            for i in T.Parallel(A):
                data_frag[64 // 2 + i % 64] = 0

    return main


@tilelang.jit
def valid_loop_not_use_loop_var(dtype: T.dtype = T.bfloat16, accum_dtype: T.dtype = T.float32, num_threads: int = 128):
    A = T.dynamic("A")

    @T.prim_func
    def main(
        data: T.Tensor((128, A), dtype),  # type: ignore
    ):
        with T.Kernel(128, threads=num_threads) as (tid,):
            data_frag = T.alloc_fragment([128], accum_dtype)

            for i in T.Parallel(128):
                if i < A:
                    data_frag[i] = data[tid, i]

            for i in T.Parallel(A):  # noqa: B007
                for j in T.Parallel(64):
                    data_frag[j] = 0  # This is valid because we don't use i

    return main


@tilelang.jit
def valid_loop_not_frag(dtype: T.dtype = T.bfloat16, accum_dtype: T.dtype = T.float32, num_threads: int = 128):
    A = T.dynamic("A")

    @T.prim_func
    def main(
        data: T.Tensor((128, A), dtype),  # type: ignore
    ):
        with T.Kernel(128, threads=num_threads) as (tid,):
            data_shared = T.alloc_shared([128], accum_dtype)

            for i in T.Parallel(128):
                if i < A:
                    data_shared[i] = data[tid, i]

            for i in T.Parallel(A):
                data_shared[i] = 0  # Valid because this is shared memory

    return main


@tilelang.jit
def valid_loop_serial(dtype: T.dtype = T.bfloat16, accum_dtype: T.dtype = T.float32, num_threads: int = 128):
    A = T.dynamic("A")

    @T.prim_func
    def main(
        data: T.Tensor((128, A), dtype),  # type: ignore
    ):
        with T.Kernel(128, threads=num_threads) as (tid,):
            data_shared = T.alloc_shared([128], accum_dtype)

            for i in T.Parallel(128):
                if i < A:
                    data_shared[i] = data[tid, i]

            for i in T.serial(A):
                data_shared[i] = 0  # Valid because this is serial

    return main


def test_invalid_loop():
    with pytest.raises(ValueError):
        _run_fragment_loop_checker(simple_invalid_loop)
    with pytest.raises(ValueError):
        _run_fragment_loop_checker(nested_invalid_loop)
    with pytest.raises(ValueError):
        _run_fragment_loop_checker(invalid_loop_with_complex_dataflow)


def test_valid_loop():
    _run_fragment_loop_checker(valid_loop_not_use_loop_var)
    _run_fragment_loop_checker(valid_loop_not_frag)
    _run_fragment_loop_checker(valid_loop_serial)


if __name__ == "__main__":
    tilelang.testing.main()
