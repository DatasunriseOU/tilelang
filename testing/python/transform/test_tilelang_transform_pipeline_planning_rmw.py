"""Regression tests for dependent shared-memory RMW pipeline stages."""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target


def _copy_then_dependent_rmw(iters=4, block=32):
    @T.prim_func
    def main(
        source: T.Buffer((iters, block), "float32"),
        output: T.Buffer((iters, block), "float32"),
    ):
        with T.Kernel(1, threads=block):
            shared = T.alloc_shared((block,), "float32")
            for k in T.Pipelined(iters, num_stages=2):
                T.copy(source[k, 0], shared, disable_tma=True)
                for i in T.Parallel(block):
                    shared[i] = T.exp(shared[i])
                T.copy(shared, output[k, 0], disable_tma=True)

    return main


def _copy_then_independent_write(iters=4, block=32):
    @T.prim_func
    def main(
        first: T.Buffer((iters, block), "float32"),
        second: T.Buffer((iters, block), "float32"),
        output: T.Buffer((iters, block), "float32"),
    ):
        with T.Kernel(1, threads=block):
            shared = T.alloc_shared((block,), "float32")
            for k in T.Pipelined(iters, num_stages=2):
                T.copy(first[k, 0], shared, disable_tma=True)
                T.copy(second[k, 0], shared, disable_tma=True)
                T.copy(shared, output[k, 0], disable_tma=True)

    return main


def _copy_then_partial_read_full_write(iters=4, block=32):
    @T.prim_func
    def main(
        source: T.Buffer((iters, block), "float32"),
        output: T.Buffer((iters, block), "float32"),
    ):
        with T.Kernel(1, threads=block):
            shared = T.alloc_shared((block,), "float32")
            for k in T.Pipelined(iters, num_stages=2):
                T.copy(source[k, 0], shared, disable_tma=True)
                for i in T.Parallel(block):
                    shared[i] = shared[i // 2]
                T.copy(shared, output[k, 0], disable_tma=True)

    return main


def _run_pipeline_planning(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    target = determine_target("cuda -arch=sm_90", return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    return tilelang.transform.PipelinePlanning()(mod)


def test_pipeline_planning_accepts_only_copy_dependent_rmw():
    planned = _run_pipeline_planning(_copy_then_dependent_rmw())
    assert "software_pipeline_stage" in planned["main"].script()

    with pytest.raises(
        tvm.error.InternalError,
        match="Multiple writes to overlapping buffer regions",
    ):
        _run_pipeline_planning(_copy_then_independent_write())

    with pytest.raises(
        tvm.error.InternalError,
        match="Multiple writes to overlapping buffer regions",
    ):
        _run_pipeline_planning(_copy_then_partial_read_full_write())
