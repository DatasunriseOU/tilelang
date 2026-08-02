import contextlib
import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang.language.extern import extern_intrinsic, Frag
from tilelang.language import extern_registry

_SCALE_HIP = r"""
__device__ void vector_scale_2_hip(const float* in_vec, float* out_vec) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out_vec[i] = in_vec[i] * 2.0f;
    }
}
"""


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Drop our test entry between tests to avoid cross-test pollution."""
    yield
    with contextlib.suppress(KeyError):
        extern_registry.unregister("vector_scale_2_hip")


def run_hip_extern_intrinsic():
    # 1. Define a @tl.extern_intrinsic with a hip body
    extern_intrinsic(
        name="vector_scale_2_hip",
        signature=lambda: (
            Frag("in_vec", (16,), "shared", "float32"),
            Frag("out_vec", (16,), "shared", "float32", is_output=True),
        ),
        bodies={"hip": _SCALE_HIP},
    )

    # 2. Create a @T.prim_func kernel that calls the intrinsic
    @T.prim_func
    def main(
        A: T.Buffer((32, 16), "float32"),
        B: T.Buffer((32, 16), "float32"),
    ):
        with T.Kernel(32, threads=1) as bx:
            # Allocate shared memory
            a_shared = T.alloc_shared((16,), "float32")
            b_shared = T.alloc_shared((16,), "float32")

            # Copy from global to shared
            for i in T.Parallel(16):
                a_shared[i] = A[bx, i]

            # Call the intrinsic
            T.evaluate(
                T.call_extern("handle", "tl.extern_intrinsic.vector_scale_2_hip", a_shared.access_ptr("r"), b_shared.access_ptr("rw"))
            )

            # Copy back to global
            for i in T.Parallel(16):
                B[bx, i] = b_shared[i]

    # 3. Build the kernel for the "hip" target
    kernel = tilelang.compile(main, out_idx=[1], target="hip")

    # 4. Run the kernel and check the output numerically using PyTorch
    # PyTorch on ROCm uses device="cuda"
    a_torch = torch.randn((32, 16), dtype=torch.float32, device="cuda")
    b_torch = torch.empty_like(a_torch)

    kernel(a_torch, b_torch)

    torch.testing.assert_close(b_torch, a_torch * 2.0)


@tilelang.testing.requires_rocm
def test_hip_extern_intrinsic():
    run_hip_extern_intrinsic()


if __name__ == "__main__":
    tilelang.testing.main()
