import tvm
import tilelang as tl
import tilelang.language as T
from tilelang.language.extern import extern_intrinsic, Frag

# define intrinsic
@extern_intrinsic(
    name="my_add",
    signature=lambda: (
        Frag("a", (16,), "shared", "float32"),
        Frag("b", (16,), "shared", "float32"),
        Frag("out", (16,), "shared", "float32", is_output=True),
    ),
    bodies={
        "cuda": r"""
__device__ void my_add(const float* a, const float* b, float* out) {
    for (int i=0; i<16; ++i) {
        out[i] = a[i] + b[i];
    }
}
"""
    }
)
def my_add(): ...

@T.prim_func
def my_kernel(
    A: T.Buffer((16,), "float32"),
    B: T.Buffer((16,), "float32"),
    C: T.Buffer((16,), "float32")
):
    with T.Kernel(1, 1):
        my_add(A, B, C)

print(my_kernel)
