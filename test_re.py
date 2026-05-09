import sys
sys.path.append(".")
from poc.triton_frontend.op_emitters.control import _func_sym_name
import inspect
print(inspect.getsource(_func_sym_name))
