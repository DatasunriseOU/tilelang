from tvm import tir
from tvm.arith import Analyzer

analyzer = Analyzer()
k = tir.Var("k", "int32")
k_next = k + 1
obligation = tir.floormod(k, 2) != tir.floormod(k_next, 2)
print("Can prove:", analyzer.can_prove(obligation))
