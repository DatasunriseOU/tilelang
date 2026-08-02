import re

import pytest

import tilelang.language as T
from tilelang.engine.lower import lower
from tilelang import tvm
from tvm.target import Target

pytest.importorskip("tilelang")


def test_cuda_alloc_buffer_special_scopes_use_scalar_declarations():
    if tvm.ffi.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang CUDA source builder is not enabled in this build.")

    @T.prim_func
    def prog(out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=32):
            state = T.alloc_var(T.float32)
            desc = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
            state[0] = T.float32(1)
            desc[0] = T.uint64(0)
            out[0] = state[0]

    target = Target({"kind": "cuda", "arch": "sm_90a", "keys": ["cuda", "gpu"]})
    source = lower(prog.with_attr("global_symbol", "main"), target=target).kernel_source

    assert re.search(r"\bfloat\s+state(?:_\d+)?\s*=", source), source
    assert re.search(r"\btl::GmmaDescriptor\s+desc(?:_\d+)?;", source), source
    assert not re.search(r"\b(?:float|uint64_t)\s+(?:state|desc)(?:_\d+)?\[1\];", source), source
