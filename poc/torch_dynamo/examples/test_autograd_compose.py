"""Wave-2 #09 regression: torch.func.grad composability + view-aliasing guard.

Skipped end-to-end on hosts without a working torch+CUDA. The structural
imports verify the API surface is wired correctly even when execution would
fail.
"""
from __future__ import annotations

import warnings

import pytest


def test_register_double_backward_symbol_imports():
    from poc.torch_dynamo.aot_autograd_glue import (
        DoubleBackwardUnsupportedError,
        autotune_select,
        register_double_backward,
    )
    assert callable(register_double_backward)
    assert callable(autotune_select)
    assert issubclass(DoubleBackwardUnsupportedError, NotImplementedError)


def test_autotune_select_cpu_fallback_returns_first_candidate():
    from poc.torch_dynamo.aot_autograd_glue import (
        autotune_select,
        _AUTOTUNE_SHORTLIST,
    )

    class _Fake:
        shape = (256, 64)
        dtype = "float16"

    chosen = autotune_select("tilelang::probe_fwd", [_Fake()], kind="fa")
    assert chosen == _AUTOTUNE_SHORTLIST["fa"][0]


def test_autotune_select_bench_picks_fastest():
    from poc.torch_dynamo.aot_autograd_glue import autotune_select

    class _Fake:
        shape = (256, 64)
        dtype = "float16"

    timings = {(64, 64, 4): 0.5, (128, 64, 8): 0.1, (128, 128, 8): 0.3}

    chosen = autotune_select(
        "tilelang::pickfastest_fwd",
        [_Fake()],
        kind="fa",
        bench_fn=lambda cfg: timings[cfg],
    )
    assert chosen == (128, 64, 8)


def test_view_aliased_input_warned_and_replaced():
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo.custom_op_wrapper import _ensure_contiguous_inputs

    base = torch.randn(8, 8)
    view = base[2:6]  # ``view._base is base``
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _ensure_contiguous_inputs("tilelang::view_probe_fwd", [view])
    assert any("view-aliased" in str(w.message) for w in caught)
    assert out[0].data_ptr() != view.data_ptr()
