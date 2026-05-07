# `torch.compile(backend="tilelang")` — POC

**Status:** forward path is real. Backward is integration #10 (stubbed).
See [`RFC_unified_fused_kernel.md`](../../RFC_unified_fused_kernel.md) §7
Phase 2 for the surrounding plan.

This directory is intentionally **outside** the production `tilelang/` tree
until Phase 2 stabilises. Do not import from production code.

## Op-naming choice (PyTorch 2.11+)

Each FX subgraph gets its own `tilelang::fused_<content_hash>` op
(scheme **A** from the design discussion). Hashing covers the FX op trace
plus input/output specs — so identical recompiles hit a process-local cache
inside `custom_op_wrapper._REGISTRY` and avoid the duplicate-registration
error that `torch.library.custom_op` raises in 2.11+. The scheme dovetails
with TileLang's existing JIT cache (`tilelang/jit/__init__.py::cached`).

## Integration shape

```
+-------------------------+
| user model (nn.Module)  |
+-----------+-------------+
            |  torch.compile(backend="tilelang")
            v
+-------------------------+
| TorchDynamo             |   <- captures FX GraphModule + example_inputs
+-----------+-------------+
            |
            v
+-------------------------+
| aot_autograd (Phase 2.3)|   <- splits joint graph into fwd / bwd
+-----+-------------+-----+
      |             |
      v             v
+-----------+ +-----------+
| fw_compiler bw_compiler|
| (FXToTileLang)         |   <- poc/torch_dynamo/fx_to_tilelang.py
+-----+-------------+----+
      |             |
      v             v
+--------------------------+
| TileLang TIR PrimFunc    |   <- ATEN_DISPATCH per FX node
+-----------+--------------+
            |
            v
+--------------------------+
| TileLang codegen         |   <- existing tilelang/ pipeline
| (CUDA / HIP / Metal)     |      (Phase 2 reuses, does not modify)
+-----------+--------------+
            |
            v
+--------------------------+
| torch.library.custom_op  |   <- poc/torch_dynamo/custom_op_wrapper.py
| (+ FakeTensor meta)      |
+--------------------------+
```

## Files

| File | Role | Status |
|---|---|---|
| `__init__.py` | Registers `tilelang` with `torch._dynamo.register_backend`; validates FX graphs fail-fast | **done (forward)** |
| `fx_to_tilelang.py` | FX walker + `ATEN_DISPATCH` table; emits a fused TileLang `PrimFunc` (matmul+relu pattern), eager FX replay otherwise | **partial (12 ops real / 10 documented stubs)** |
| `custom_op_wrapper.py` | `torch.library.custom_op` registration with FakeTensor meta + autograd guard | **done (forward)** |
| `aot_autograd_glue.py` | `aot_autograd` fw/bw compilers | **stub — integration #10** |
| `examples/torch_compile_smoke.py` | End-to-end forward smoke; pytest auto-skips if torch / tilelang missing | **runnable, passing on host** |

## Op coverage (forward only)

| Status | Ops |
|---|---|
| Real lowering | `add`, `sub`, `mul`, `div`, `relu`, `gelu`, `silu`, `matmul`, `mm`, `bmm`, `softmax` / `_softmax`, `layer_norm` / `native_layer_norm` |
| Documented stub (raises with recipe) | `addmm`, `tanh`, `rms_norm`, `log_softmax`, `sum`, `mean`, `where`, `masked_fill`, `_scaled_dot_product_flash_attention`, `scaled_dot_product_attention` |

The "documented stub" emitters carry a 5-10 line recipe in their docstring
so the next contributor can pattern-match against the existing emitters.

## Backward path

Out of scope for this POC. `_check_no_grad` in `custom_op_wrapper.py`
raises `NotImplementedError` if any input tensor has `requires_grad=True`.
Integration #10 will replace this guard with a real backward via
`aot_autograd`.

## Canonical 2026 API used here

```python
# Backend registration
from torch._dynamo import register_backend
register_backend(name="tilelang", compiler_fn=tilelang_backend)

# Backend signature
def tilelang_backend(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> Callable[..., list[torch.Tensor]]: ...

# Backward via aot_autograd
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
backend = aot_autograd(fw_compiler=..., bw_compiler=...)

# Custom op wrapping
from torch.library import custom_op, register_fake
```

## How to extend (Phase 2.2 work order)

1. **Pick an FX op category.** Start with elementwise (`aten.add`, `aten.mul`,
   `aten.relu`) — emits `T.copy` + scalar lambdas.
2. **Implement the emitter** in `fx_to_tilelang.py`'s `ATEN_DISPATCH`.
   Replace the `_stub` factory call with a real function that mutates
   `LoweringContext.value_map`.
3. **Wire the per-node-op handler** (`on_call_function` etc.) to call the
   emitter and stash the resulting TIR fragment.
4. **Assemble in `FXToTileLang.run`.** Build a `tvm.tir.PrimFunc`, hand it to
   the existing TileLang codegen pipeline.
5. **Wrap in custom_op.** Replace the static `FusedKernelArtifact` fields in
   `custom_op_wrapper.py` with values derived from the `PrimFunc` signature.
6. **Run `examples/torch_compile_smoke.py`.** It should now print the output
   shape instead of the expected `NotImplementedError`.

## Op coverage (Phase 2.2 targets)

`fx_to_tilelang.ATEN_DISPATCH` ships entries for the inductor-coverage set
called out in RFC §7 Phase 2.2:

- matmul family: `matmul`, `mm`, `bmm`, `addmm`
- elementwise: `add`, `sub`, `mul`, `div`
- activations: `relu`, `gelu`, `silu`, `tanh`
- norms / reductions: `layer_norm`, `native_layer_norm`, `rms_norm`,
  `softmax`, `_softmax`, `log_softmax`, `sum`, `mean`
- attention: `_scaled_dot_product_flash_attention`,
  `scaled_dot_product_attention`

All emitters currently raise. Adding a new op is a one-line edit to the
dispatch table plus an emitter function.

## Out of scope for the POC

- Dynamic shapes (RFC §8 risk #8).
- Autograd through the fused region (RFC §8 risk #5; deferred to Phase 2.3).
- Cross-CTA reductions on Metal (RFC §8 risk #7).
- TTGIR / CuTeDSL bridges (RFC Phase 4).
