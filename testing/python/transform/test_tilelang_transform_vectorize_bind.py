import tilelang
from tilelang import tvm


def _lower_vectorized(mod):
    vectorized = tilelang.transform.VectorizeLoop()(mod)
    return tilelang.transform.LowerTileLangLetStmt()(vectorized)


def _collect_binds(mod):
    binds = []
    tvm.tirx.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: binds.append(node) if isinstance(node, tvm.tirx.Bind) else None,
    )
    return binds


def _assert_single_bind_and_valid_ssa(mod):
    binds = _collect_binds(mod)
    assert len(binds) == 1
    assert tvm.tirx.analysis.verify_ssa(mod["main"])
    assert not tvm.tirx.analysis.undefined_vars(mod["main"].body, mod["main"].params)


def test_vectorize_bind_remaps_var_and_its_uses():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    b = tvm.tirx.decl_buffer((4,), "float32", name="B")
    i = tvm.tirx.Var("i", "int32")
    x = tvm.tirx.Var("x", "float32")
    alpha = tvm.tirx.Var("alpha", "float32")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(x, a[i] + (alpha - alpha)),
            tvm.tirx.BufferStore(b, x, [i]),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [a.data, b.data, alpha],
        loop,
        buffer_map={a.data: a, b.data: b},
    ).with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})

    vectorized = tilelang.transform.VectorizeLoop()(mod)
    vectorized_body = vectorized["main"].body
    assert isinstance(vectorized_body, tvm.tirx.SeqStmt)
    bind, store = vectorized_body.seq
    assert isinstance(bind, tvm.tirx.Bind)
    assert isinstance(store, tvm.tirx.BufferStore)
    assert str(bind.var.dtype) == "float32x4"
    assert bind.value.dtype == bind.var.dtype
    assert store.value.same_as(bind.var)

    tilelang.transform.Simplify()(vectorized)


def test_vectorize_bind_whole_body_scalarization_keeps_single_definition():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    b = tvm.tirx.decl_buffer((4,), "float32", name="B")
    i = tvm.tirx.Var("i", "int32")
    x = tvm.tirx.Var("x", "float32")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(x, a[i]),
            tvm.tirx.BufferStore(
                b,
                tvm.tirx.if_then_else(
                    i < 2,
                    x,
                    tvm.tirx.FloatImm("float32", 0),
                ),
                [i],
            ),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [a.data, b.data],
        loop,
        buffer_map={a.data: a, b.data: b},
    ).with_attr("global_symbol", "main")

    _assert_single_bind_and_valid_ssa(_lower_vectorized(tvm.IRModule({"main": func})))


def test_vectorize_bind_local_scalarization_keeps_single_definition():
    b = tvm.tirx.decl_buffer((4,), "float32", name="B")
    i = tvm.tirx.Var("i", "int32")
    x = tvm.tirx.Var("x", "float32")
    alpha = tvm.tirx.Var("alpha", "float32")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(x, alpha),
            tvm.tirx.IfThenElse(
                i < 2,
                tvm.tirx.BufferStore(b, x, [i]),
                None,
            ),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [b.data, alpha],
        loop,
        buffer_map={b.data: b},
    ).with_attr("global_symbol", "main")

    _assert_single_bind_and_valid_ssa(_lower_vectorized(tvm.IRModule({"main": func})))


def test_vectorize_bind_changed_same_lane_preserves_var_identity():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    i = tvm.tirx.Var("i", "int32")
    ptr = tvm.tirx.Var("ptr", "handle")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(ptr, tvm.tirx.address_of(a[i])),
            tvm.tirx.Evaluate(ptr),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [a.data],
        loop,
        buffer_map={a.data: a},
    ).with_attr("global_symbol", "main")

    vectorized = tilelang.transform.VectorizeLoop()(tvm.IRModule({"main": func}))
    bind, evaluate = vectorized["main"].body.seq
    assert isinstance(bind, tvm.tirx.Bind)
    assert isinstance(evaluate, tvm.tirx.Evaluate)
    assert bind.var.same_as(ptr)
    assert evaluate.value.same_as(ptr)


def test_vectorize_bind_changed_same_lane_scalarization_uses_fresh_var():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    i = tvm.tirx.Var("i", "int32")
    ptr = tvm.tirx.Var("ptr", "handle")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(ptr, tvm.tirx.address_of(a[i])),
            tvm.tirx.IfThenElse(i < 2, tvm.tirx.Evaluate(ptr), None),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [a.data],
        loop,
        buffer_map={a.data: a},
    ).with_attr("global_symbol", "main")

    vectorized = tilelang.transform.VectorizeLoop()(tvm.IRModule({"main": func}))
    binds = _collect_binds(vectorized)
    assert len(binds) == 2
    assert sum(bind.var.same_as(ptr) for bind in binds) == 1

    restored_bind = next(bind for bind in binds if not bind.var.same_as(ptr))
    restored_value_vars = []
    tvm.tirx.stmt_functor.post_order_visit(
        restored_bind.value,
        lambda node: restored_value_vars.append(node) if isinstance(node, tvm.tirx.Var) else None,
    )
    scalar_loops = []
    tvm.tirx.stmt_functor.post_order_visit(
        vectorized["main"].body,
        lambda node: scalar_loops.append(node) if isinstance(node, tvm.tirx.For) and node.loop_var.name.endswith("_s") else None,
    )
    assert len(scalar_loops) == 1
    assert any(var.same_as(scalar_loops[0].loop_var) for var in restored_value_vars)
    assert tvm.tirx.analysis.verify_ssa(vectorized["main"])
    assert not tvm.tirx.analysis.undefined_vars(
        vectorized["main"].body,
        vectorized["main"].params,
    )


def test_vectorize_bind_scalarization_restores_binding_dependencies():
    a = tvm.tirx.decl_buffer((4,), "float32", name="A")
    b = tvm.tirx.decl_buffer((4,), "float32", name="B")
    i = tvm.tirx.Var("i", "int32")
    x = tvm.tirx.Var("x", "float32")
    y = tvm.tirx.Var("y", "float32")
    alpha = tvm.tirx.Var("alpha", "float32")

    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Bind(x, a[i]),
            tvm.tirx.Bind(y, x + alpha),
            tvm.tirx.IfThenElse(
                i < 2,
                tvm.tirx.BufferStore(b, y, [i]),
                None,
            ),
        ]
    )
    loop = tvm.tirx.For(i, 0, 4, tvm.tirx.ForKind.VECTORIZED, body)
    func = tvm.tirx.PrimFunc(
        [a.data, b.data, alpha],
        loop,
        buffer_map={a.data: a, b.data: b},
    ).with_attr("global_symbol", "main")

    vectorized = tilelang.transform.VectorizeLoop()(tvm.IRModule({"main": func}))
    # Native tirx.Bind restoration stays printable and traversable before the
    # compatibility lowering pass. A vendored LetStmt here crashes the printer.
    vectorized["main"].script()

    lowered = tilelang.transform.LowerTileLangLetStmt()(vectorized)
    restored_binds = [bind for bind in _collect_binds(lowered) if bind.var.name in {"x", "y"} and bind.var.dtype.lanes == 1]
    assert [bind.var.name for bind in restored_binds] == ["x", "y"]
    assert tvm.tirx.analysis.verify_ssa(lowered["main"])
    assert not tvm.tirx.analysis.undefined_vars(
        lowered["main"].body,
        lowered["main"].params,
    )


if __name__ == "__main__":
    tvm.testing.main()
