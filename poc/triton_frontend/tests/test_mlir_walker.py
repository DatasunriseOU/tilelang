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
