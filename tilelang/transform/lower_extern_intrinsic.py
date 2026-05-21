import tvm
from tvm import tir
from tilelang.language.extern_registry import lookup
from tilelang.language.extern import EXTERN_CALL_PREFIX

@tvm.tir.transform.prim_func_pass(opt_level=0)
class LowerExternIntrinsic:
    """A pass to inject per-target body strings for tl.extern_intrinsic calls.

    This pass scans the PrimFunc for `tirx.call_extern` nodes that invoke
    registered `tl.extern_intrinsic` kernels. For each unique intrinsic found,
    it looks up the body source string for the specified target (e.g. "cuda", "metal").
    The gathered body strings are combined and appended to the function's
    "pragma_import_c" attribute, allowing the backend code generator to emit them.
    """
    def __init__(self, target_name: str):
        self.target_name = target_name

    def transform_function(self, func: tir.PrimFunc, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tir.PrimFunc:
        bodies = []
        seen_bodies = set()

        def _mutate(node):
            if not (
                isinstance(node, tir.Call)
                and getattr(node.op, "name", None) == "tirx.call_extern"
                and node.args
                and isinstance(node.args[0], tir.StringImm)
            ):
                return None

            name = node.args[0].value
            if not name.startswith(EXTERN_CALL_PREFIX):
                return None

            intrinsic_name = name[len(EXTERN_CALL_PREFIX):]
            intrinsic = lookup(intrinsic_name)
            if intrinsic is None:
                raise ValueError(
                    f"extern_intrinsic '{intrinsic_name}' is not registered; "
                    "register it before lowering the TileLang kernel."
                )
            body = intrinsic.bodies.get(self.target_name)
            if body is None:
                targets = ", ".join(sorted(intrinsic.bodies))
                raise ValueError(
                    f"extern_intrinsic '{intrinsic_name}' has no body for target "
                    f"'{self.target_name}'. Registered targets: [{targets}]."
                )
            if body not in seen_bodies:
                seen_bodies.add(body)
                bodies.append(body)

            new_args = list(node.args)
            new_args[0] = tir.StringImm(intrinsic_name)
            return tir.Call(node.dtype, node.op, new_args, node.span)

        new_body = tir.stmt_functor.ir_transform(func.body, None, _mutate)

        if bodies:
            combined_body = "\n\n".join(bodies)
            # Find if there is an existing pragma_import_c AttrStmt at the top level
            # In TVM, we wrap the body with an AttrStmt with key "pragma_import_c"
            new_body = tir.AttrStmt(
                node=tir.StringImm("pragma_import_c"),
                attr_key="pragma_import_c",
                value=tir.StringImm(combined_body),
                body=new_body
            )
            return func.with_body(new_body)

        if not new_body.same_as(func.body):
            return func.with_body(new_body)

        return func
