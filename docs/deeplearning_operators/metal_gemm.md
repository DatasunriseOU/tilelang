# Metal GEMM

TileLang provides two Metal GEMM paths targeting Apple Silicon GPUs.

## Overview

| Path | `T.gemm` instruction | micro tile | hardware | status |
|------|---------------------|------------|----------|--------|
| simdgroup | `metal.simdgroup` | 8 x 8 x 8 | M1-M5 | stable |
| cooperative tensor | `metal.cooperative_tensor` | 16 x 32 x 16 | M5+ | experimental |

Both paths are available through the same `T.gemm(A_shared, B_shared, C)` call. The compiler
automatically selects the cooperative tensor path when the tile shape and C scope permit;
otherwise it falls back to the simdgroup path.

## Cooperative Tensor Path

On M5+ devices the cooperative tensor path uses Apple's **Metal 4 MPP**
(`mpp::tensor_ops::matmul2d`) to access the Neural Accelerator hardware. This can
deliver significantly higher throughput than the 8x8 simdgroup path.

### Selection rules

TileLang picks the cooperative tensor path when **all** of the following hold:

- `C` is placed in **shared memory** (not `local.fragment` / `metal.simdgroup`).
- M % 16 == 0, N % 32 == 0, K % 16 == 0 (16x32x16 micro tile).
- The number of warps can be evenly partitioned into `M/16 x N/32` tile groups.

If any condition fails the compiler falls back to `metal.simdgroup` without user action.

### Current limitations

- **Shared-C only.** The `local.fragment` C path always uses simdgroup today. Direct
  fragment-C cooperative tensor requires a fragment layout that `tirx.IndexMap.inverse`
  cannot yet represent; this is being worked on.
- **float32 accumulation only in MPP path.** MPP matmul2d loads `half`/`bfloat` inputs
  but the destination `cooperative_tensor` is always `float32`.
- **No software pipelining (`num_stages=0`).** Cooperative tensor shared-memory loads
  are emitted inside the inner loop; the pipeline planner does not yet interleave them.
- **No transpose flags.** The lowering assumes `trans_A=False, trans_B=False`.

### Compile-time integration on M5

The MSL output of the cooperative-tensor path uses `-std=metal4.0` and
`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, both of which
require Xcode 16 (or newer) and Apple M5+ silicon. To opt in:

```python
from tilelang.engine.callback import register_default_metal_compile_callback
register_default_metal_compile_callback(override=True)
```

This is **not** auto-registered — on M1-M4 hardware the resulting compile
command would fail.

## Future Work

- Support fragment-C cooperative tensor (requires inverse layout work or a
  `metal.cooperative_tensor` scope for accumulators).
- Enable software pipelining for cooperative tensor loads.
- Extend to transposed operand layouts.
- Add direct global-to-global GEMM (GG) lowering through MPP.
- Performance tuning: auto-select inner-K steps, persistent cooperative tensor
  accumulators.
