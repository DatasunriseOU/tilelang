import tvm
from tvm import tir
a = tir.Var("a", "int32")
b = tir.const(3, "int32")
expr = tir.FloorMod(a, b) == tir.const(1, "int32")

import tvm._ffi
func = tvm._ffi.get_global_func("tl.z3.bv_can_prove")
func(a, -5, -4, expr, 32)
