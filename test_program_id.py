import sys
sys.path.append(".")
from poc.triton_frontend import from_triton_kernel
from poc.triton_frontend.tests.test_vector_add import _vector_add_kernel

func = from_triton_kernel(_vector_add_kernel, constexprs={"BLOCK": 128}, target="cuda")
print("func type:", type(func))
print("func body:", func.body)
