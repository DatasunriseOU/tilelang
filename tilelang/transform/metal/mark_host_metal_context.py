from tvm.ir import Op
from tvm.tir import (
    PyStmtExprMutator,
    functor,
    Evaluate,
    AttrStmt,
)
from tvm.tir.transform import prim_func_pass

"""
Transformation pass to mark host-side kernel calls for Metal interop synchronization.

To execute TVM-generated Metal kernels inside another framework's scheduling
scope, the TVM runtime must be able to reuse the owner's active Metal command
buffer. This keeps execution ordering and memory consistency without copying
or staging tensor storage.

This pass identifies calls to `tir.tvm_call_packed_lowered` occurring within a
`compute_scope` and wraps them with a `metal_context` attribute. This attribute
signals the downstream host C codegen to inject runtime logic that:
1. Reuses an owner-provided Metal command buffer/event when one is registered.
2. Otherwise uses an explicit TVM Metal stream when one is active.
3. Otherwise lets TVM's Metal runtime use its own default scheduling path.
"""


tvm_call_packed_lowered = Op.get("tir.tvm_call_packed_lowered")


@functor.mutator
class MarkHostMetalContextMutator(PyStmtExprMutator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_in_compute_scope = False

    def visit_attr_stmt_(self, stmt):
        switch = stmt.attr_key == "compute_scope"
        old_value = False
        if switch:
            assert not self.is_in_compute_scope
            old_value, self.is_in_compute_scope = self.is_in_compute_scope, True
        s = self.visit_stmt(stmt.body)
        if switch:
            self.is_in_compute_scope = old_value
        return s

    def visit_evaluate_(self, op: Evaluate):
        if self.is_in_compute_scope and op.value.op.same_as(tvm_call_packed_lowered):
            return AttrStmt(0, "metal_context", "", op)
        return op


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        mutator = MarkHostMetalContextMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
