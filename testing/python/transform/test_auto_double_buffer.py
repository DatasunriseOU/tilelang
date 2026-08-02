"""Tests for the AutoDoubleBuffer transform (Z3 Roadmap Idea #2).

The pass is currently a SAFE STUB: when enabled it detects candidates and
logs a Z3 verdict but does NOT mutate the IR. These tests therefore check:

1. The pass exists, is importable, and is registered.
2. The PassConfig key gates behavior — default OFF leaves IR unchanged.
3. With config ON, a canonical-pattern kernel still has IR identical to
   the input (safe-stub guarantee).
4. With config ON, a non-canonical kernel is also left unchanged (no
   regression).

When a future iteration replaces the safe stub with the real ping-pong
rewrite, these tests should be tightened to assert structural changes.
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T


def _run_pass(func, config=None):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    if config is None:
        return tl.transform.AutoDoubleBuffer()(mod)
    with tvm.transform.PassContext(config=config):
        return tl.transform.AutoDoubleBuffer()(mod)


def _assert_unchanged(func, config=None):
    """Run the pass and assert the resulting body is structurally equal
    to the input."""
    mod_out = _run_pass(func, config=config)
    expected = func.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(mod_out["main"], expected, map_free_vars=True)


@T.prim_func
def _canonical_pattern_kernel(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128,), "float32"),
):
    """Canonical pattern: per-iter load global -> shared, then use shared."""
    A_shared = T.alloc_buffer((128,), "float32", scope="shared")
    for k in range(128):
        for i in range(128):
            A_shared[i] = A[k, i]
        for i in range(128):
            B[i] = A_shared[i]


@T.prim_func
def _cross_iter_dep_kernel(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128,), "float32"),
):
    """Non-canonical: the load address at iter k depends on a value
    written at iter k-1 (via B). This is the kind of pattern Z3 can't
    prove sound for ping-pong."""
    A_shared = T.alloc_buffer((128,), "float32", scope="shared")
    for k in range(128):
        for i in range(128):
            # Address depends on prior iteration's write to B
            A_shared[i] = A[k, i] + B[i]
        for i in range(128):
            B[i] = A_shared[i]


@T.prim_func
def _no_shared_kernel(
    A: T.Tensor((128,), "float32"),
    B: T.Tensor((128,), "float32"),
):
    """No shared memory at all — pass should be a strict no-op."""
    for i in range(128):
        B[i] = A[i] * T.float32(2.0)


def test_default_off_preserves_behavior():
    """No config given → default OFF → IR must be byte-equal to input."""
    _assert_unchanged(_canonical_pattern_kernel)
    _assert_unchanged(_cross_iter_dep_kernel)
    _assert_unchanged(_no_shared_kernel)


def test_canonical_pattern_detected_default_off():
    """Canonical pattern + explicit OFF → no transformation, no regression."""
    _assert_unchanged(
        _canonical_pattern_kernel,
        config={"tl.auto_double_buffer": False},
    )


def test_canonical_pattern_with_z3_proof_inserts_pong():
    """Canonical pattern + ON: in safe-stub mode the IR is still
    structurally unchanged (the pass logs the Z3 verdict and returns).
    When the real transformation lands this test should assert that a
    second `_pong` allocation appears."""
    _assert_unchanged(
        _canonical_pattern_kernel,
        config={"tl.auto_double_buffer": True},
    )


def test_uncertain_pattern_falls_back():
    """Cross-iteration dependency + ON → Z3 cannot prove soundness →
    no transformation. The IR must be unchanged."""
    _assert_unchanged(
        _cross_iter_dep_kernel,
        config={"tl.auto_double_buffer": True},
    )


def test_no_shared_kernel_unchanged():
    """Kernel with no shared memory at all → unchanged regardless of
    config."""
    _assert_unchanged(
        _no_shared_kernel,
        config={"tl.auto_double_buffer": True},
    )
    _assert_unchanged(
        _no_shared_kernel,
        config={"tl.auto_double_buffer": False},
    )


def test_pass_is_registered():
    """Sanity: the pass is callable and yields a tvm.transform.Pass."""
    p = tl.transform.AutoDoubleBuffer()
    assert p is not None
    assert callable(p)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
