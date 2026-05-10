import re

def rewrite():
    with open('poc/torch_dynamo/fx_to_tilelang.py', 'r') as f:
        content = f.read()

    # Find the start of _emit_sequential_region
    start_str = '    def _emit_sequential_region('
    start_idx = content.find(start_str)
    
    # Find the end (next def)
    end_str = '    def _emit_sequential_reduction('
    end_idx = content.find(end_str)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find start or end")
        return
        
    old_method = content[start_idx:end_idx]
    
    new_method = """    def _emit_sequential_region(
        self, T: Any,
        region: List[Tuple[str, Tuple[Any, ...]]],
    ) -> Any:
        ops_only = [op for op, _ in region]

        compute_ops: List[Tuple[str, Tuple[Any, ...]]] = [
            (op, payload) for (op, payload) in region
            if op not in self._SEQUENTIAL_VIEW_OPS
        ]
        if not compute_ops:
            raise NotImplementedError(
                "sequential region: only view-like ops, nothing to compile")

        if len(compute_ops) == 1:
            sole_op, sole_payload = compute_ops[0]
            if sole_op in self._SEQUENTIAL_REDUCTION_OPS:
                return self._emit_sequential_reduction(
                    T, sole_op, sole_payload)
            if sole_op in self._SEQUENTIAL_MATMUL_OPS:
                return self._emit_sequential_matmul(
                    T, sole_op, sole_payload)

        for op, _ in compute_ops:
            if op not in self._SEQUENTIAL_UNARY_OPS and op not in self._SEQUENTIAL_BINARY_OPS:
                raise FxToTileLangUnsupported(
                    f"sequential region: op trace {ops_only!r} "
                    "contains unsupported ops; falling back to extern is intentional")

        first_payload = compute_ops[0][1]
        src_spec = first_payload[1] if len(first_payload) > 1 else None
        if not isinstance(src_spec, _TensorSpec):
            raise NotImplementedError("sequential region: cannot resolve source tensor spec")

        shape = src_spec.shape
        dtype = src_spec.dtype
        n_elem = 1
        for s in shape:
            n_elem *= int(s)
        if n_elem <= 0:
            raise NotImplementedError(f"sequential region: degenerate numel from shape {shape}")

        BLOCK = 128 if n_elem >= 128 else (64 if n_elem >= 64 else max(n_elem, 1))

        # Dataflow analysis
        node_map = {n.name: n for n in self.gm.graph.nodes}
        internal_nodes = {payload[0] for _, payload in compute_ops}
        
        external_inputs: List[str] = []
        external_names_set = set()
        
        for op_name, payload in compute_ops:
            node_name = payload[0]
            if node_name not in node_map:
                continue
            fx_node = node_map[node_name]
            for arg in fx_node.args:
                if isinstance(arg, type(fx_node)):
                    if arg.name not in internal_nodes and arg.name not in external_names_set:
                        arg_spec = self.ctx.value_map.get(arg)
                        if isinstance(arg_spec, _TensorSpec):
                            if arg_spec.shape != shape or arg_spec.dtype != dtype:
                                raise NotImplementedError(
                                    f"sequential region: broadcast / mixed-dtype not yet supported "
                                    f"({arg_spec.shape}|{arg_spec.dtype} vs {shape}|{dtype})")
                        external_inputs.append(arg.name)
                        external_names_set.add(arg.name)
                        
        if len(external_inputs) > 6:
            raise FxToTileLangUnsupported(
                f"sequential region: too many external inputs ({len(external_inputs)})")
                
        def _apply_unary_local(T_mod: Any, op_name: str, v: Any) -> Any:
            if op_name == "relu":
                return T_mod.max(v, T_mod.cast(0, dtype))
            if op_name == "tanh":
                return T_mod.tanh(v)
            if op_name == "sigmoid":
                one = T_mod.cast(1, dtype)
                return one / (one + T_mod.exp(-v))
            if op_name == "silu":
                one = T_mod.cast(1, dtype)
                return v / (one + T_mod.exp(-v))
            if op_name == "gelu":
                return (T_mod.cast(0.5, dtype) * v *
                        (T_mod.cast(1.0, dtype) +
                         T_mod.tanh(T_mod.cast(0.7978845608028654, dtype) *
                                    (v + T_mod.cast(0.044715, dtype) *
                                     v * v * v))))
            if op_name == "exp":
                return T_mod.exp(v)
            if op_name == "log":
                return T_mod.log(v)
            if op_name == "sqrt":
                return T_mod.sqrt(v)
            if op_name == "rsqrt":
                return T_mod.rsqrt(v)
            if op_name == "neg":
                return -v
            if op_name == "abs":
                return T_mod.abs(v)
            raise FxToTileLangUnsupported(f"sequential unary op {op_name} has no TIR builder")

        def _apply_binary_local(T_mod: Any, op_name: str, a: Any, b: Any) -> Any:
            if op_name == "add":
                return a + b
            if op_name == "sub":
                return a - b
            if op_name == "mul":
                return a * b
            if op_name == "div":
                return a / b
            if op_name == "maximum":
                return T_mod.max(a, b)
            if op_name == "minimum":
                return T_mod.min(a, b)
            raise FxToTileLangUnsupported(f"sequential binary op {op_name} has no TIR builder")

        def _compose_chain(T_mod: Any, ext_vals: dict) -> Any:
            computed = dict(ext_vals)
            last_val = None
            for op_name, payload in compute_ops:
                node_name = payload[0]
                fx_node = node_map.get(node_name)
                if not fx_node:
                    continue
                if op_name in self._SEQUENTIAL_UNARY_OPS:
                    arg_name = fx_node.args[0].name
                    v = computed.get(arg_name)
                    v_out = _apply_unary_local(T_mod, op_name, v)
                    computed[node_name] = v_out
                    last_val = v_out
                elif op_name in self._SEQUENTIAL_BINARY_OPS:
                    name1 = fx_node.args[0].name
                    name2 = fx_node.args[1].name
                    v_out = _apply_binary_local(T_mod, op_name, computed.get(name1), computed.get(name2))
                    computed[node_name] = v_out
                    last_val = v_out
            return last_val

        # Generate branches for up to 6 external inputs
        ext_names = external_inputs
        if len(ext_names) == 1:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx]})
            return kernel
        elif len(ext_names) == 2:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx]})
            return kernel
        elif len(ext_names) == 3:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx]})
            return kernel
        elif len(ext_names) == 4:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), X3: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    X3_flat = T.Buffer((n_elem,), dtype, data=X3.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx], ext_names[3]: X3_flat[idx]})
            return kernel
        elif len(ext_names) == 5:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), X3: T.Tensor(shape, dtype), X4: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    X3_flat = T.Buffer((n_elem,), dtype, data=X3.data)
                    X4_flat = T.Buffer((n_elem,), dtype, data=X4.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx], ext_names[3]: X3_flat[idx], ext_names[4]: X4_flat[idx]})
            return kernel
        elif len(ext_names) == 6:
            @T.prim_func
            def kernel(X0: T.Tensor(shape, dtype), X1: T.Tensor(shape, dtype), X2: T.Tensor(shape, dtype), X3: T.Tensor(shape, dtype), X4: T.Tensor(shape, dtype), X5: T.Tensor(shape, dtype), Y: T.Tensor(shape, dtype)):
                if False: _ = shape; _ = dtype
                with T.Kernel(T.ceildiv(n_elem, BLOCK), threads=BLOCK) as bx:
                    X0_flat = T.Buffer((n_elem,), dtype, data=X0.data)
                    X1_flat = T.Buffer((n_elem,), dtype, data=X1.data)
                    X2_flat = T.Buffer((n_elem,), dtype, data=X2.data)
                    X3_flat = T.Buffer((n_elem,), dtype, data=X3.data)
                    X4_flat = T.Buffer((n_elem,), dtype, data=X4.data)
                    X5_flat = T.Buffer((n_elem,), dtype, data=X5.data)
                    Y_flat = T.Buffer((n_elem,), dtype, data=Y.data)
                    for i in T.Parallel(BLOCK):
                        idx = bx * BLOCK + i
                        if idx < n_elem:
                            Y_flat[idx] = _compose_chain(T, {ext_names[0]: X0_flat[idx], ext_names[1]: X1_flat[idx], ext_names[2]: X2_flat[idx], ext_names[3]: X3_flat[idx], ext_names[4]: X4_flat[idx], ext_names[5]: X5_flat[idx]})
            return kernel
        else:
            raise FxToTileLangUnsupported(
                f"sequential region: unsupported number of external inputs ({len(ext_names)})")

"""
    
    with open('poc/torch_dynamo/fx_to_tilelang.py', 'w') as f:
        f.write(content[:start_idx] + new_method + content[end_idx:])

rewrite()
