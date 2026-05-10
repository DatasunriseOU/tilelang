import numpy as np
import tvm
import tvm.testing
from poc.triton_frontend import from_triton_kernel
import triton
import triton.language as tl

@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def test_run():
    N = 1024
    BLOCK = 128
    func = from_triton_kernel(_vector_add_kernel, constexprs={"BLOCK": BLOCK}, target="llvm")
    print("Lowered func!")
    
    rt_mod = tvm.build(func, target="llvm")
    print("Built func!")
    
    x_np = np.random.rand(N).astype("float32")
    y_np = np.random.rand(N).astype("float32")
    out_np = np.zeros(N, dtype="float32")
    
    dev = tvm.cpu()
    x_tvm = tvm.nd.array(x_np, dev)
    y_tvm = tvm.nd.array(y_np, dev)
    out_tvm = tvm.nd.array(out_np, dev)
    
    rt_mod(x_tvm, y_tvm, out_tvm, N)
    print("Ran func!")
    tvm.testing.assert_allclose(out_tvm.numpy(), x_np + y_np)
    print("Passed!")

if __name__ == "__main__":
    test_run()
