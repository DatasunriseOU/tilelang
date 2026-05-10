import z3
# Prove that bvsrem(-5, 3) < 0 is True
s = z3.Solver()
a = z3.BitVecVal(-5, 32)
b = z3.BitVecVal(3, 32)
s.add(z3.SRem(a, b) < 0)
print("SRem < 0 is SAT?", s.check())
# Prove that bvsmod(-5, 3) < 0 is False
s2 = z3.Solver()
s2.add(z3.z3core.Z3_mk_bvsmod(s2.ctx.ref(), a.ast, b.ast) < 0)
print("SMod < 0 is SAT?", s2.check())
