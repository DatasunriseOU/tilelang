import tvm
from tvm import tir

k = tir.Var("k", "int32")
outer = tir.Var("outer", "int32")
vec = 4
ramp = tir.Ramp(0, 1, vec)
expr = k - outer * vec - ramp
analyzer = tvm.arith.Analyzer()
print("Simplified:", analyzer.simplify(expr))
