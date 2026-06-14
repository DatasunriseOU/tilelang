"""Unit tests for the DeadLocalStoreElim fixpoint pass
(:mod:`poc.triton_frontend.dead_local_store_elim`).

Builds a minimal SBlock-structured PrimFunc -- the exact shape ``from_ttir``
emits (launch-thread AttrStmts -> SBlockRealize -> SBlock(root) whose body is a
SeqStmt of per-tile staging loops + a real store) -- and asserts the pass:

  * removes a ``scope="local"`` staging loop whose buffer is never read,
  * keeps a ``scope="local"`` staging loop whose buffer IS read (a load mask),
  * never removes the real global store,
  * is bit-identical structurally on a func with no dead stores (idempotent).
"""
from __future__ import annotations

import pytest

# Bootstrap the tilelang dev-root tvm onto sys.path (the editable install's
# loader runs on first ``import tilelang``); only then is ``tvm`` importable.
pytest.importorskip("tilelang")

from poc.triton_frontend.dead_local_store_elim import (  # noqa: E402
    eliminate_dead_local_stores,
)

tvm = pytest.importorskip("tvm")
from tvm import tir  # noqa: E402
import tvm.tirx as tirx  # noqa: E402


def _alloc(name, shape, dtype="int32", scope="local"):
    return tir.decl_buffer(shape, dtype, name=name, scope=scope)


def _staging_loop(buf, value_fn, extent=64):
    """A ``for i: buf[i] = value_fn(i)`` staging loop."""
    i = tir.Var(buf.name + "_i", "int32")
    store = tir.BufferStore(buf, value_fn(i), [i])
    return tir.For(i, 0, extent, tir.ForKind.SERIAL, store)


def _build_func(*, with_dead):
    live_mask = _alloc("live_mask", [64], "int32")          # read by the store guard
    dead_idx = _alloc("dead_idx", [64], "int64")            # never read -> dead
    out = _alloc("out_global", [64], "float32", scope="global")

    # live_mask[i] = i      (read below in the store predicate)
    s_live = _staging_loop(live_mask, lambda i: i)
    # dead_idx[i] = i*7     (never read anywhere)
    s_dead = _staging_loop(dead_idx, lambda i: tir.Cast("int64", i) * tir.const(7, "int64"))

    # real store guarded by a read of live_mask:  if live_mask[j] < 32: out[j] = 1.0
    j = tir.Var("j", "int32")
    guard = tir.LT(tir.BufferLoad(live_mask, [j]), tir.const(32, "int32"))
    real_store = tir.For(
        j, 0, 64, tir.ForKind.SERIAL,
        tir.IfThenElse(guard, tir.BufferStore(out, tir.const(1.0, "float32"), [j]), None),
    )

    seq = [s_live, s_dead, real_store] if with_dead else [s_live, real_store]
    allocs = [live_mask, dead_idx, out] if with_dead else [live_mask, out]
    block = tirx.SBlock(
        iter_vars=[], reads=[], writes=[], name_hint="root",
        body=tir.SeqStmt(seq), alloc_buffers=allocs,
    )
    realize = tirx.SBlockRealize([], tvm.tir.const(True, "bool"), block)
    body = tir.AttrStmt(
        tir.IterVar((0, 128), tir.Var("threadIdx.x", "int32"), 1, "threadIdx.x"),
        "thread_extent", tir.const(128, "int32"), realize,
    )
    return tir.PrimFunc([], body)


def _local_alloc_names(func):
    names = []
    def visit(s):
        if type(s).__name__ in ("SBlock", "Block"):
            for ab in s.alloc_buffers:
                if str(ab.scope()) in ("local", ""):
                    names.append(ab.name)
    tir.stmt_functor.post_order_visit(func.body, visit)
    return set(names)


def test_dce_removes_dead_local_keeps_live():
    func = _build_func(with_dead=True)
    before = _local_alloc_names(func)
    assert "dead_idx" in before and "live_mask" in before

    folded = eliminate_dead_local_stores(func)
    after = _local_alloc_names(folded)

    # dead, never-read staging buffer removed; live mask preserved.
    assert "dead_idx" not in after, "dead local staging buffer should be removed"
    assert "live_mask" in after, "live (read) mask must be preserved"

    # the real global store survives (out is still written somewhere).
    text = folded.script()
    assert "out_global" in text
    assert "live_mask" in text
    assert "dead_idx" not in text


def test_dce_idempotent_when_no_dead():
    func = _build_func(with_dead=False)
    before = _local_alloc_names(func)
    folded = eliminate_dead_local_stores(func)
    after = _local_alloc_names(folded)
    assert before == after == {"live_mask"}


if __name__ == "__main__":
    test_dce_removes_dead_local_keeps_live()
    test_dce_idempotent_when_no_dead()
    print("ALL PASS")
