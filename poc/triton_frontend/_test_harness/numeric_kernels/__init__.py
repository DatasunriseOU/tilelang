"""Numeric verification kernels for the Triton -> TileLang -> Metal -> MLX pipeline.

Each module in this package exposes a uniform contract consumed by
``poc.triton_frontend._test_harness.numeric_smoke``:

    TRITON_KERNEL : Callable           # a ``@triton.jit`` decorated function
    LAUNCH_GRID   : tuple[int, ...]    # e.g. (1,) or (32,)
    META_ARGS     : dict[str, int]     # constexpr bindings
    ATOL          : float              # absolute tolerance for np.allclose
    RTOL          : float              # relative tolerance for np.allclose

    def make_inputs(*, dtype="float32", seed=0)
        -> tuple[list[np.ndarray], np.ndarray]:
        '''Returns (kernel_args_as_arrays, expected_output).'''

    def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
        '''Pure numpy compute that reproduces the kernel's output.'''

The harness only imports these modules lazily (after triton has been
import-checked), so importing this package has zero runtime deps beyond
numpy.
"""

from __future__ import annotations

__all__ = ["KERNEL_MODULES"]

# Names of the kernel sub-modules in the order the harness should run
# them. We start with the smallest (no dot/reduce) and escalate.
KERNEL_MODULES = (
    "vector_add",
    "softmax",
    "row_sum",
    "gather_rows_3d",
    "matmul",
    "layer_norm",
    "flash_attention",
    "fa_v3",
    "paged_attention",
    "dot_reduce_atomic",
    "dot_reduce_atomic_trans_b",
    "atomic_hist",
    "tma_descriptor_copy",
    "fla_dot_exp2",
    "histogram",
    "split_join",
    "where_broadcast",
)
