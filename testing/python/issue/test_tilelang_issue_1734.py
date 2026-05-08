import tilelang
import tilelang.testing
from tilelang import language as T


def _device_kernel_source(source):
    for marker in ("kernel void", "__global__ void"):
        pos = source.find(marker)
        if pos >= 0:
            return source[pos:]
    return source


def test_issue_1734():
    """Test that loop-invariant if statements are hoisted out of loops."""

    @tilelang.jit()
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor[(2, 512), T.float32],
            B: T.Tensor[(2, 512), T.float32],
            C: T.Tensor[(2,), T.float32],
        ):
            with T.Kernel(1, threads=256):
                A_local = T.alloc_fragment((2, 512), T.float32)
                B_local = T.alloc_fragment((2, 512), T.float32)
                C_local = T.alloc_fragment((2,), T.float32)

                T.copy(A, A_local)
                T.copy(C, C_local)

                for i, j in T.Parallel(2, 512):
                    if C_local[i] >= 0:
                        B_local[i, j] = A_local[i, j]

                T.copy(B_local, B)

        return main

    mod = tilelang.lower(kernel.get_tir())
    source = mod.kernel_source
    # Verify that the if statement is hoisted outside the for loop
    # After hoisting, we should see "if" before "for" pattern
    kernel_source = _device_kernel_source(source)
    if_pos = kernel_source.find("if (")
    for_pos = kernel_source.find("for (")
    assert if_pos >= 0 and for_pos >= 0, f"Expected hoisted if and loop in generated source.\n{kernel_source}"
    assert if_pos < for_pos, "Loop-invariant if should be hoisted outside the loop"


if __name__ == "__main__":
    tilelang.testing.main()
