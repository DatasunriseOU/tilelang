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
        bodies = set()

        def _mutate(node):
            if isinstance(node, tir.Call) and hasattr(node.op, "name") and node.op.name == "tirx.call_extern":
                if isinstance(node.args[0], tir.StringImm):
                    name = node.args[0].value
                    if name.startswith(EXTERN_CALL_PREFIX):
                        intrinsic_name = name[len(EXTERN_CALL_PREFIX):]
                        intrinsic = lookup(intrinsic_name)
                        if intrinsic is not None:
                            body = intrinsic.bodies.get(self.target_name)
                            if body is not None:
                                bodies.add(body)
                            
                            new_args = list(node.args)
                            new_args[0] = tir.StringImm(intrinsic_name)
                            return tir.Call(node.dtype, node.op, new_args, node.span)
            return None

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
