"""Tests for ``poc.triton_frontend.mlir_walker``.

These tests cover both the happy path (mlir.ir importable -> real walk
populating value_map/buffers) and the degraded fallback path (mlir.ir
absent -> exactly-once UserWarning). The regex walker is *not* deleted
per ``feedback_no_silent_delete``, so the fallback path is exercised
explicitly here too.
"""
from __future__ import annotations

import importlib
import warnings

import pytest


# A minimal TTIR fragment: one func with tt.load + arith.addf + tt.store.
# Triton dialect ops aren't registered in a generic LLVM install, so the
# parser must accept unregistered dialects (parse_ttir does that).
_SIMPLE_TTIR = """
module {
  "tt.func"() ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>):
    %0 = "tt.load"(%arg0) : (!tt.ptr<f32>) -> f32
    %1 = "tt.load"(%arg1) : (!tt.ptr<f32>) -> f32
    %2 = "arith.addf"(%0, %1) : (f32, f32) -> f32
    "tt.store"(%arg2, %2) : (!tt.ptr<f32>, f32) -> ()
    "tt.return"() : () -> ()
  }) {sym_name = "add", function_type = (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>) -> ()} : () -> ()
}
"""


def test_walker_imports_or_warns_once():
    """Either mlir.ir is importable (MLIR_WALKER_AVAILABLE=True) or a single
    UserWarning matching DEGRADED_WARNING_MESSAGE was emitted on probe.
    """
    # Force a fresh import so the module-load probe runs again under
    # an active catch_warnings.
    import poc.triton_frontend.mlir_walker as mw  # noqa: WPS433
    # Reset the one-shot warned flag so try_import_mlir can re-warn.
    mw._WARNED_ONCE = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ir = mw.try_import_mlir()
    if ir is not None:
        assert mw.MLIR_WALKER_AVAILABLE is True
        # No degraded warning when bindings are present.
        assert not any(
            mw.DEGRADED_WARNING_MESSAGE in str(w.message) for w in caught
        )
    else:
        # Fallback path: exactly one warning, matching the constant.
        matches = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and str(w.message) == mw.DEGRADED_WARNING_MESSAGE
        ]
        assert len(matches) == 1, (
            f"expected exactly one DEGRADED warning, got {len(matches)}: "
            f"{[str(w.message) for w in caught]}"
        )


def test_walk_simple_ttir():
    """Parse a tiny TTIR module and walk it; visitor should see N+1 ops
    in pre-order (where N = body ops + the func op + the module op).
    """
    from poc.triton_frontend.mlir_walker import (  # noqa: WPS433
        MLIR_WALKER_AVAILABLE,
        TTIRWalker,
        parse_ttir,
        walk_module,
    )

    if not MLIR_WALKER_AVAILABLE:
        pytest.skip(
            "mlir.ir not importable in this environment; the fallback "
            "regex walker is exercised by the coverage suite already."
        )

    module = parse_ttir(_SIMPLE_TTIR)
    if module is None:
        pytest.skip("parse_ttir failed (likely missing tt.* dialect TD).")

    seen: list[str] = []

    class Recorder:
        def visit_op(self, op):  # noqa: D401
            from poc.triton_frontend.mlir_walker import _op_name  # noqa: WPS433
            name = _op_name(op)
            if name:
                seen.append(name)

    walk_module(module, Recorder())

    # Expect at minimum: tt.func + (tt.load, tt.load, arith.addf,
    # tt.store, tt.return) = 6 ops; plus the top-level module op.
    # We're loose here because parse_ttir may wrap/skip the outer
    # builtin.module depending on the bindings shape.
    assert len(seen) >= 6, f"expected >=6 ops, saw {seen}"
    # Ensure pre-order: func appears before its body.
    func_idx = next(
        (i for i, n in enumerate(seen) if n.endswith(".func")), None
    )
    addf_idx = next(
        (i for i, n in enumerate(seen) if n == "arith.addf"), None
    )
    assert func_idx is not None and addf_idx is not None
    assert func_idx < addf_idx, (
        f"pre-order violated: func at {func_idx}, arith.addf at {addf_idx}"
    )


def test_value_map_populated():
    """After walking, ctx.buffers should have an entry per func argument
    and ctx.value_map should reference each block argument SSA value.
    """
    from poc.triton_frontend.mlir_walker import (  # noqa: WPS433
        MLIR_WALKER_AVAILABLE,
        TTIRWalker,
        parse_ttir,
        walk_module,
    )

    if not MLIR_WALKER_AVAILABLE:
        pytest.skip(
            "mlir.ir not importable; value_map population is mlir.ir-only."
        )

    module = parse_ttir(_SIMPLE_TTIR)
    if module is None:
        pytest.skip("parse_ttir failed (likely missing tt.* dialect TD).")

    walker = TTIRWalker()
    walk_module(module, walker)

    # The TTIR fragment has 3 func args (%arg0, %arg1, %arg2) -- each
    # should land in ctx.buffers keyed by the printed arg name.
    assert len(walker.ctx.buffers) >= 3, (
        f"expected >=3 buffer entries, got {list(walker.ctx.buffers)}"
    )
    # value_map should have at least one entry per block argument.
    # (Emitters may add more for tt.load / arith.addf results; we only
    # require the block-arg seeding here, since OP_TABLE emitters
    # require TVM and may be no-ops in the bindings-only test env.)
    assert len(walker.ctx.value_map) >= 3, (
        f"expected >=3 value_map entries, got {len(walker.ctx.value_map)}"
    )


def test_bootstrap_jaxlib_alias_idempotent():
    """``bootstrap_jaxlib_alias`` must be safe to call repeatedly.

    The function is called at package import time (via
    ``_mlir_path_setup.probe_and_wire_mlir``) and again at module load
    inside ``mlir_walker._bootstrap_and_probe``. Both call sites assume
    the second invocation is a no-op when the alias is already in
    place. We assert that explicitly here.
    """
    from poc.triton_frontend._mlir_path_setup import (  # noqa: WPS433
        bootstrap_jaxlib_alias,
    )

    # First call: may or may not produce the alias depending on host;
    # we just assert it returns a bool and does not raise.
    first = bootstrap_jaxlib_alias()
    assert isinstance(first, bool)

    # Second call: must not raise and must return the same bool.
    second = bootstrap_jaxlib_alias()
    assert second == first

    # Third call (paranoid): still no raise.
    third = bootstrap_jaxlib_alias()
    assert third == first


def test_wrap_module_for_walker_passthrough_for_non_jaxlib():
    """``wrap_module_for_walker`` returns the module unchanged when its
    class doesn't live under ``jaxlib.`` and ``module.body`` already
    carries ``regions`` (the upstream MLIR shape).
    """
    from poc.triton_frontend.mlir_walker import wrap_module_for_walker  # noqa: WPS433

    class _FakeBlock:
        regions = ()

    class _FakeModule:
        body = _FakeBlock()
        operation = None

    fm = _FakeModule()
    assert wrap_module_for_walker(fm) is fm


def test_walker_skips_regions_of_self_handling_ops():
    """The walker MUST NOT descend into regions of ops that walk their
    own regions (``tt.reduce`` / ``tt.scan`` / ``scf.for`` / ``scf.if``
    / ``scf.while``). If it did, inner combiner / loop-body ops would be
    dispatched out of program order -- their operands are block
    arguments of the inner block, never bound by the parent emitter --
    and ``WalkerCtx.get`` would raise ``KeyError: SSA value not yet
    mapped`` on the next downstream op.

    Regression for the softmax / layer_norm e2e failures where
    ``tt.reduce``'s combiner region contained ``arith.maximumf`` whose
    operands were the combiner block arguments. The previous walker
    descended into the combiner, dispatched ``arith.maximumf``, and
    immediately tried to look up the unbound block-arg SSA values.
    """
    from poc.triton_frontend.mlir_walker import (  # noqa: WPS433
        OPS_THAT_HANDLE_OWN_REGIONS,
        walk_module,
    )

    # ``tt.reduce`` / ``tt.scan`` and the scf.* trio must all be in the
    # allowlist; this is the contract the test pins.
    for op_name in ("tt.reduce", "tt.scan", "scf.for", "scf.if", "scf.while"):
        assert op_name in OPS_THAT_HANDLE_OWN_REGIONS, (
            f"{op_name} must be in OPS_THAT_HANDLE_OWN_REGIONS so the "
            f"walker doesn't dispatch its inner region ops out of order."
        )

    # Build a fake op tree with a tt.reduce that has an inner arith.addf.
    # The walker should visit tt.reduce but NOT recurse into its region.
    class _FakeBlockArg:
        pass

    class _FakeOp:
        def __init__(self, name, regions=()):
            self.name = name
            self.regions = regions

    class _FakeBlock:
        def __init__(self, ops):
            self.operations = ops
            self.arguments = ()

    class _FakeRegion:
        def __init__(self, blocks):
            self.blocks = blocks

    # tt.reduce { arith.addf }
    inner_addf = _FakeOp("arith.addf")
    reduce_region = _FakeRegion([_FakeBlock([inner_addf])])
    reduce_op = _FakeOp("tt.reduce", regions=[reduce_region])

    # scf.for { tt.load }
    inner_load = _FakeOp("tt.load")
    for_region = _FakeRegion([_FakeBlock([inner_load])])
    for_op = _FakeOp("scf.for", regions=[for_region])

    # tt.func body: tt.reduce, scf.for, plus a sibling arith.mulf
    sibling = _FakeOp("arith.mulf")
    func_region = _FakeRegion([_FakeBlock([reduce_op, for_op, sibling])])
    func_op = _FakeOp("tt.func", regions=[func_region])

    class _FakeModule:
        body = None
        operation = func_op

    seen: list[str] = []

    class Recorder:
        def visit_op(self, op):  # noqa: D401
            seen.append(op.name)

    walk_module(_FakeModule(), Recorder())

    # Parent ops must be visited.
    assert "tt.reduce" in seen, seen
    assert "scf.for" in seen, seen
    # Sibling at func body level must be visited.
    assert "arith.mulf" in seen, seen
    # CRITICAL: inner ops of self-handling regions must NOT be visited.
    assert "arith.addf" not in seen, (
        f"walker descended into tt.reduce's combiner region: {seen}"
    )
    assert "tt.load" not in seen, (
        f"walker descended into scf.for's body region: {seen}"
    )


def test_wrap_module_for_walker_adapts_jaxlib_shape():
    """When ``module.body`` is a Block (no ``regions``), the wrapper
    produces an adapter that forwards ``operation`` and hides ``body``.
    """
    from poc.triton_frontend.mlir_walker import (  # noqa: WPS433
        _JaxlibModuleAdapter,
        wrap_module_for_walker,
    )

    class _FakeBlock:
        # No ``regions`` attribute -- jaxlib's Module.body shape.
        pass

    class _FakeOperation:
        regions = ()

    class _FakeModule:
        body = _FakeBlock()
        operation = _FakeOperation()

    fm = _FakeModule()
    wrapped = wrap_module_for_walker(fm)
    assert isinstance(wrapped, _JaxlibModuleAdapter)
    # The adapter must NOT expose ``body`` (otherwise the walker would
    # take the body branch and silently bottom out).
    assert getattr(wrapped, "body", None) is None
    assert wrapped.operation is fm.operation
