import sys
sys.path.append(".")
from poc.triton_frontend.op_emitters.control import _func_sym_name

op_str = 'tt.func private @triton.language.standard.max__fp32S128S_c0_cFalse_cTrue_cFalse() {}'
class DummyOp:
    def __str__(self): return op_str

print(_func_sym_name(DummyOp()))
