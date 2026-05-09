import tvm
from tvm import arith

analyzer = arith.Analyzer()
v = tvm.tir.Var("v", "int32")
expr = tvm.tir.Min(v, 512)
bound = analyzer.const_int_bound(expr)
print(f"min_value: {bound.min_value}, max_value: {bound.max_value}")
