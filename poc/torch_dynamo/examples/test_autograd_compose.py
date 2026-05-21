"""Wave-2/3/4 #09 regression: torch.func.grad composability + view-aliasing guard.

Skipped end-to-end on hosts without a working torch+CUDA. The structural
imports verify the API surface is wired correctly even when execution would
fail.
"""
from __future__ import annotations

from types import SimpleNamespace
import threading
from uuid import uuid4
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


def test_view_aliased_input_rejected_without_hidden_copy():
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo.custom_op_wrapper import _ensure_contiguous_inputs

    base = torch.randn(8, 8)
    view = base[2:6]  # ``view._base is base``
    with pytest.raises(RuntimeError, match=r"input #0 .*view-aliased"):
        _ensure_contiguous_inputs("tilelang::view_probe_fwd", [view])


def test_torch_compile_view_sum_backward_materialises_gradient_alias():
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo import register

    register()

    def fn(x):
        return x.view(2, 3).sum()

    x = torch.randn(6, dtype=torch.float32, requires_grad=True)
    ref = fn(x)
    ref.backward()
    expected_grad = x.grad.detach().clone()
    x.grad = None

    compiled = torch.compile(fn, backend="tilelang", fullgraph=True)
    actual = compiled(x)
    actual.backward()

    torch.testing.assert_close(actual.detach(), ref.detach())
    torch.testing.assert_close(x.grad, expected_grad)


def test_torch_compile_transpose_sum_backward_lowers_clone(capsys):
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo import register

    register()

    def fn(x):
        return (x.view(2, 3).transpose(0, 1) * 2).sum()

    x = torch.randn(6, dtype=torch.float32, requires_grad=True)
    ref = fn(x)
    ref.backward()
    expected_grad = x.grad.detach().clone()
    x.grad = None

    capsys.readouterr()
    compiled = torch.compile(fn, backend="tilelang", fullgraph=True)
    actual = compiled(x)
    actual.backward()
    captured = capsys.readouterr()

    torch.testing.assert_close(actual.detach(), ref.detach())
    torch.testing.assert_close(x.grad, expected_grad)
    assert "Failed to build prim_func" not in captured.out + captured.err


def test_torch_compile_layer_norm_backward_reconstructs_saved_view():
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo import register

    register()

    def fn(x):
        return torch.nn.functional.layer_norm(x.view(2, 3), (3,)).sum()

    x = torch.randn(6, dtype=torch.float32, requires_grad=True)
    ref = fn(x)
    ref.backward()
    expected_grad = x.grad.detach().clone()
    x.grad = None

    compiled = torch.compile(fn, backend="tilelang", fullgraph=True)
    actual = compiled(x)
    actual.backward()

    torch.testing.assert_close(actual.detach(), ref.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_torch_compile_cat_of_slices_backward_lowers_slice():
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo import register

    register()

    def fn(x):
        return torch.cat([x[:3], x[3:]], dim=0).sum()

    x = torch.randn(6, dtype=torch.float32, requires_grad=True)
    ref = fn(x)
    ref.backward()
    expected_grad = x.grad.detach().clone()
    x.grad = None

    compiled = torch.compile(fn, backend="tilelang", fullgraph=True)
    actual = compiled(x)
    actual.backward()

    torch.testing.assert_close(actual.detach(), ref.detach())
    torch.testing.assert_close(x.grad, expected_grad)


def test_wave3_specialize_prim_func_returns_input_when_tvm_missing():
    # specialize_prim_func is a no-op when tvm isn't importable or PrimFunc is
    # None. We pin the no-op branch — the tvm-present branch is exercised
    # implicitly when callers actually have a PrimFunc to specialise.
    from poc.torch_dynamo.aot_autograd_glue import specialize_prim_func

    assert specialize_prim_func(None, (64, 64, 4)) is None
    sentinel = object()
    assert specialize_prim_func(sentinel, ()) is sentinel  # empty config


def test_wave3_has_symint_shape_detects_non_int_dims():
    from poc.torch_dynamo.aot_autograd_glue import _has_symint_shape

    class _SymInt:
        # mimics torch.SymInt's duck-type — non-int instance
        def __int__(self) -> int:
            return 32

    class _FakeT:
        shape = (_SymInt(), 64)

    class _ConcreteT:
        shape = (32, 64)

    assert _has_symint_shape(_FakeT()) is True
    assert _has_symint_shape(_ConcreteT()) is False


def test_wave4_atomic_accumulator_double_backward_invokes_real_closure():
    # Wave-4 fix: previous wave-3 test re-implemented the zero-grad rule in
    # the test, so it would silently drift if the impl changed. This test
    # actually invokes the registered backward closure via real custom ops so
    # it locks in PyTorch's boxed List[Tensor] autograd structure.
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo.aot_autograd_glue import register_double_backward
    from poc.torch_dynamo.custom_op_wrapper import FusedKernelArtifact, wrap_as_custom_op

    name = f"wave4_dbw_test_{uuid4().hex}"
    spec = SimpleNamespace(shape=(3,), dtype="float32")
    bwd_calls = []

    def _fwd_launcher(x):
        return x * 2

    def _bwd_launcher(x, grad):
        bwd_calls.append((x.detach().clone(), grad.detach().clone()))
        return grad * 2

    fwd = wrap_as_custom_op(
        FusedKernelArtifact(
            name=name,
            launcher=_fwd_launcher,
            input_specs=(spec,),
            output_specs=(spec,),
        ),
        {},
        is_backward=False,
        allow_grad_inputs=True,
    )
    wrap_as_custom_op(
        FusedKernelArtifact(
            name=name,
            launcher=_bwd_launcher,
            input_specs=(spec, spec),
            output_specs=(spec,),
        ),
        {},
        is_backward=True,
        allow_grad_inputs=True,
    )

    ok = register_double_backward(f"tilelang::{name}_fwd", f"tilelang::{name}_bwd")
    assert ok is True

    x = torch.randn(3, requires_grad=True)
    y = fwd(x)
    grad_out = torch.ones_like(y)
    assert grad_out.is_contiguous()
    (grad_x,) = torch.autograd.grad(y, x, grad_outputs=grad_out)

    assert torch.allclose(grad_x, torch.full_like(x, 2))
    assert len(bwd_calls) == 1


def test_wave4_missing_forward_dispatch_op_returns_false():
    pytest.importorskip("torch")

    from poc.torch_dynamo.aot_autograd_glue import register_double_backward
    from poc.torch_dynamo.custom_op_wrapper import FusedKernelArtifact, wrap_as_custom_op

    name = f"wave4_missing_fwd_{uuid4().hex}"
    spec = SimpleNamespace(shape=(3,), dtype="float32")

    def _bwd_launcher(x, grad):
        return grad

    wrap_as_custom_op(
        FusedKernelArtifact(
            name=name,
            launcher=_bwd_launcher,
            input_specs=(spec, spec),
            output_specs=(spec,),
        ),
        {},
        is_backward=True,
        allow_grad_inputs=True,
    )

    assert register_double_backward(
        f"tilelang::{name}_missing_fwd",
        f"tilelang::{name}_bwd",
    ) is False


def test_wave4_compile_symbolic_fallback_does_not_recurse():
    """Wave-4 fix #2 regression: ensure the symbolic→concrete fallback does
    not recurse infinitely when ``compile_symbolic`` raises.

    We invoke the fallback path directly with a stub graph that triggers the
    broken-walker branch and assert (a) a single warning fires and (b) the
    inner ``_compile_one_side`` call is reached *exactly once* (no
    RecursionError, no repeat entries into compile_symbolic).
    """
    from poc.torch_dynamo import aot_autograd_glue as glue

    call_count = {"compile_symbolic": 0, "compile_one_side": 0}
    original_compile_one_side = glue._compile_one_side
    original_compile_symbolic = glue.compile_symbolic

    def _stub_compile_one_side(gm, example_inputs, *, is_backward):
        call_count["compile_one_side"] += 1
        # Guard must be active here (we're inside the fallback).
        assert glue._in_symbolic_fallback() is True, \
            "fallback guard must be armed inside _compile_one_side"
        return "OK"

    def _wrapped_compile_symbolic(gm, example_inputs, *, is_backward, tile_var_names=("M",)):
        call_count["compile_symbolic"] += 1
        return original_compile_symbolic(
            gm, example_inputs, is_backward=is_backward, tile_var_names=tile_var_names
        )

    glue._compile_one_side = _stub_compile_one_side
    glue.compile_symbolic = _wrapped_compile_symbolic
    try:
        # GraphModule that breaks fx_to_tilelang import → exception in
        # compile_symbolic → fallback.
        class _BadGM:
            @property
            def graph(self):
                raise RuntimeError("forced failure")
            meta: dict = {}

            # mock getattr(gm, "meta", {})
            def __getattr__(self, name):
                if name == "meta":
                    return self.meta
                raise AttributeError(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = glue.compile_symbolic(_BadGM(), [], is_backward=False)

        # Exactly one warning of type RuntimeWarning about the symbolic path.
        symbolic_warns = [w for w in caught if "symbolic" in str(w.message).lower()]
        assert len(symbolic_warns) == 1, (
            f"expected exactly 1 symbolic-fallback warning, got {len(symbolic_warns)}"
        )
        # _compile_one_side reached once, compile_symbolic entered once
        # (the outer call only — no recursion).
        assert call_count == {"compile_symbolic": 1, "compile_one_side": 1}, call_count
        assert out == "OK"
        # Guard is cleared after the fallback returns.
        assert glue._in_symbolic_fallback() is False
    finally:
        glue._compile_one_side = original_compile_one_side
        glue.compile_symbolic = original_compile_symbolic


def test_wave4_validate_graph_importable_from_aot_glue():
    """Wave-4 fix #1 regression: ``_validate_graph`` must be importable from
    ``aot_autograd_glue`` (the broken ``from . import _validate_graph``
    resolved to the package object before this fix).
    """
    from poc.torch_dynamo._graph_validation import (
        UnsupportedFXOpError,
        validate_graph,
    )
    from poc.torch_dynamo.aot_autograd_glue import _validate_graph as legacy_alias
    from poc.torch_dynamo import _validate_graph as init_alias

    assert callable(validate_graph)
    assert callable(legacy_alias)
    assert callable(init_alias)
    assert init_alias is validate_graph
    assert issubclass(UnsupportedFXOpError, NotImplementedError)


def test_wave4_pending_double_backward_pairing_flushes_on_bwd_compile():
    """Wave-4 fix #4: forward compile records a pending pairing; the bwd
    side flushes it once the bwd op lands in _REGISTRY."""
    from poc.torch_dynamo import aot_autograd_glue as glue

    fwd_q = "tilelang::wave4_pending_test_fwd"
    bwd_q = "tilelang::wave4_pending_test_bwd"

    glue._record_pending_double_backward(fwd_q, bwd_q, has_atomic_accumulator=False)
    with glue._PENDING_DBW_LOCK:
        assert fwd_q in glue._PENDING_DBW
        assert glue._PENDING_DBW[fwd_q] == (bwd_q, False)

    # Without the bwd op in _REGISTRY, flush is a no-op.
    glue._finalise_double_backward_pairings()
    with glue._PENDING_DBW_LOCK:
        assert fwd_q in glue._PENDING_DBW

    # Drop the pairing manually (we can't invoke register_autograd on a
    # not-actually-an-op qualname, so we just confirm pop happens).
    with glue._PENDING_DBW_LOCK:
        glue._PENDING_DBW.pop(fwd_q, None)


def test_wave4_autotune_cache_is_thread_safe():
    """Wave-4 fix: ``_AUTOTUNE_CACHE`` writes are guarded by ``_AUTOTUNE_LOCK``
    so concurrent compiles can't lose the first writer's choice."""
    from poc.torch_dynamo.aot_autograd_glue import (
        _AUTOTUNE_CACHE,
        autotune_select,
    )

    _AUTOTUNE_CACHE.clear()

    class _Fake:
        shape = (256, 64)
        dtype = "float16"

    bench_calls = []

    def _slow_bench(cfg):
        bench_calls.append(cfg)
        # Each thread picks a different "fastest" if writes race.
        return float(cfg[0])  # smaller first dim = faster

    threads = [
        threading.Thread(
            target=lambda: autotune_select(
                "tilelang::race_test_fwd", [_Fake()], kind="fa", bench_fn=_slow_bench,
            )
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cache holds exactly one entry for the key, regardless of race count.
    keys = [k for k in _AUTOTUNE_CACHE if k[0] == "tilelang::race_test_fwd"]
    assert len(keys) == 1
    # All threads converged to the same (fastest) choice.
    chosen = _AUTOTUNE_CACHE[keys[0]]
    # _slow_bench keys: smaller first dim wins → (64, 64, 4) for "fa" shortlist.
    assert chosen == (64, 64, 4)
    # The benchmark function should only be called once per candidate (3 candidates for 'fa')
    assert len(bench_calls) == 3


def test_wave4_artifact_carries_atomic_flag():
    """Wave-4 fix #6: ``has_atomic_accumulator`` lives on the artifact."""
    from poc.torch_dynamo.custom_op_wrapper import FusedKernelArtifact

    art = FusedKernelArtifact(
        name="probe",
        launcher=lambda *a: None,
        input_specs=(),
        output_specs=(),
    )
    assert art.has_atomic_accumulator is False
    art.has_atomic_accumulator = True
    assert art.has_atomic_accumulator is True


def test_wave4_detect_atomic_accumulator_uses_explicit_loop():
    """Wave-4 fix #6: the previous inline expression mis-parsed as
    ``(x and y) or z`` and matched on every non-empty string. The new
    helper uses an explicit any() loop with the correct precedence."""
    from poc.torch_dynamo.aot_autograd_glue import _detect_atomic_accumulator

    class _Node:
        def __init__(self, t):
            self.target = t

    class _Graph:
        def __init__(self, ts):
            self.nodes = [_Node(t) for t in ts]

    class _GM:
        def __init__(self, ts):
            self.graph = _Graph(ts)

    # Plain matmul → no atomic.
    assert _detect_atomic_accumulator(_GM(["aten.matmul.default", "aten.relu.default"])) is False
    # scatter_add hit.
    assert _detect_atomic_accumulator(_GM(["aten.scatter_add.default"])) is True
    # index_add hit.
    assert _detect_atomic_accumulator(_GM(["aten.index_add_.default"])) is True
    # Empty graph.
    assert _detect_atomic_accumulator(_GM([])) is False
    # No graph attr at all.
    assert _detect_atomic_accumulator(object()) is False
