import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.phase import PreLowerSemanticCheck


def _run_fragment_loop_checker(jit_func):
    func = jit_func.get_tir()
    PreLowerSemanticCheck(tvm.IRModule({func.attrs["global_symbol"]: func}))


@tilelang.jit
def _dynamic_parallel_fragment_elem_offset():
    limit = T.dynamic("limit")

    @T.prim_func
    def main(out: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((128,), T.float32)
            for i in T.Parallel(limit):
                frag_view = T.FragmentBuffer((1,), T.float32, data=frag.data, elem_offset=i)
                frag_view[0] = T.float32(0)
            out[0] = frag[0]

    return main


@tilelang.jit
def _dynamic_parallel_fragment_index():
    limit = T.dynamic("limit")

    @T.prim_func
    def main(out: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((128,), T.float32)
            for i in T.Parallel(limit):
                frag[i] = T.float32(0)
            out[0] = frag[0]

    return main


def test_dynamic_parallel_fragment_elem_offset_allowed():
    _run_fragment_loop_checker(_dynamic_parallel_fragment_elem_offset)


def test_dynamic_parallel_fragment_index_rejected():
    with pytest.raises(ValueError, match="symbolic range"):
        _run_fragment_loop_checker(_dynamic_parallel_fragment_index)


if __name__ == "__main__":
    tilelang.testing.main()
