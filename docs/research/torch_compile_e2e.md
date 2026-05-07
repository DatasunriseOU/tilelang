# torch.compile(backend="tilelang") — empirical end-to-end (wave-7 retry-3)

Empirical run on 2026-05-07 of the torch.compile-with-tilelang-backend smoke
chain after wave-7 #4 redo (`99e6e638`, drop `-> Any` return annotation in
`_impl/_fake` so `torch.library.infer_schema` accepts the registration).

## Environment

| Field | Value |
|---|---|
| Host | macOS Darwin 25.4.0 (Apple Silicon) |
| Python | 3.12 (from `python3.12` Frameworks install) |
| Torch | 2.8.0 |
| `torch.backends.mps.is_available()` | `True` |
| Worktree | `/private/tmp/tl_poc_review` (`poc-integrations-review`) |
| Predecessors | `99e6e638` (wave-7 #4 redo), `4e487af7` (wave-7 #6 test fix), `bce67cf9`, `cac10a0`, `27392ded`, `a439df0`, `d8ef9558`, `c1bf9bc9`, `b05f9e65`, `48951199`, `3078b95d`, `f045f92d`, `c2216f9c`, `2cfc7c94`, `4e2f79d6` |

`poc.torch_dynamo.register()` succeeds → `tilelang` becomes a recognised
`torch.compile` backend.

## Per-case verdicts

| # | Case | Status | Result |
|---|---|---|---|
| (a) | `relu(x)` — single unary | ✅ **PASS** | `max_abs_diff = 0.000e+00` |
| (b) | `x + y` — single binary | ✅ **PASS** | `max_abs_diff = 0.000e+00` |
| (c) | `relu(addmm(b, x, w))` — fused chain | ✅ **PASS** | `max_abs_diff = 0.000e+00` |
| (d) | `scaled_dot_product_attention(q, k, v)` — multi-region | ❌ **FAIL** | `UnsupportedFXOpError: tilelang backend cannot lower _scaled_dot_product_flash_attention_for_cpu` |

**3 / 4 PASS.** Headline: the FX → TileLang → custom_op → eval pipeline
*works end-to-end on CPU* for unary, binary, and fused linear+activation
chains.

Inputs were random `torch.randn` tensors on default (CPU) device:

* (a) `x = (8, 8)`
* (b) `x, y = (8, 8)`
* (c) `b = (16,)`, `x = (8, 32)`, `w = (32, 16)`
* (d) `q = k = v = (1, 4, 16, 32)`

`out` and `ref` are bit-identical for cases (a)-(c) — the custom_op artifact
is calling the eager kernel under the hood (FXToTileLang's MVP path), not
emitting MSL/CUDA. Once the engine path is wired (wave-8) we expect a small
non-zero diff vs eager (compute-order changes).

## Failure detail — case (d)

```
BackendCompilerFailed: backend='tilelang' raised:
UnsupportedFXOpError: tilelang backend cannot lower the following FX nodes:
  - %_scaled_dot_product_flash_attention_for_cpu :
      [num_users=1] = call_function[target=torch._C._nn.scaled_dot_product_attention_for_cpu]
```

This op is the CPU-FA-style aten kernel that the dynamo trace inserts when
`torch.nn.functional.scaled_dot_product_attention` is called on CPU. It is
*not yet* in `ATEN_DISPATCH` (`fx_to_tilelang.py`). Two paths to fix:

1. **Wave-8 cheap fix**: add a passthrough emitter that calls the original
   `torch._C._nn.scaled_dot_product_attention_for_cpu` (no fusion, but the
   chain at least lowers without raising).
2. **Wave-8 proper fix**: wire the hand-written FA kernel from
   `poc/torch_dynamo/_kernels/flash_attention.py` (already exists) into
   `_emit_fused_flash_attention_region` and dispatch to it when an
   `_scaled_dot_product_flash_attention*` FX node is seen.

Recommend (2) — the kernel is already written from wave-1 #02 work.

## Next-blocker fixes for full coverage

1. **`aten._scaled_dot_product_flash_attention_for_cpu`** in `ATEN_DISPATCH`
   (wave-8). Required for (d) and any `nn.functional.sdpa` call.
2. **`aten.layer_norm` tuple unpacking** — second remaining fail in
   `test_torch_compile_chain.py` per Test #1's matrix (commit `7dc9ed58`).
3. **`requires_grad` guard misfire** — first remaining fail (also from Test
   #1). Both block training-mode usage.
4. **MPS device coverage** — current 4-case run was on CPU; need MPS rerun
   to confirm device-affinity invariants in `_ensure_contiguous_inputs`.

## Why this is good news

Before wave-7 #4 redo, *zero* `torch.compile(backend="tilelang")` invocations
succeeded — every call raised `RuntimeError: argument types must be one of
[Tensor, List[Tensor], int, ...]` at registration time because
`infer_schema` rejected the `-> Any` return annotation.

After 99e6e638, **forward-only** unary/binary/fused-linear+activation
graphs go end-to-end with bit-exact eager parity. The remaining gaps are
expected ATEN_DISPATCH coverage gaps, not pipeline integrity bugs.
