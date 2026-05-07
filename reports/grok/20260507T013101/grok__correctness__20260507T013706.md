---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T01:37:06.368332+00:00
files: ['/private/tmp/tl_poc_review/poc/torch_dynamo/__init__.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/_fusion_patterns.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/_kernels/flash_attention.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/aot_autograd_glue.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/examples/torch_compile_smoke.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/examples/test_torch_compile_chain.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/README.md']
---
**Code Review: `poc/torch_dynamo/` — `torch.compile(backend="tilelang")` POC**

**Overall verdict (correctness focus):** The scaffold is a solid, RFC-aligned foundation. The architecture (`Dynamo → FX walker → op_trace + patterns → per-region PrimFunc → `torch.library.custom_op``) is correct and matches the 2026 PyTorch 2.11+ contract (`register_backend`, `aot_autograd`, `custom_op` + `register_fake`). Forward-only path works numerically via eager fallback; real TileLang kernels are emitted for the smoke-test `matmul+activation` case.

**Open design question (op naming) — resolved correctly to (A):**  
The code correctly implements scheme **(A)** (`tilelang::fused_<content_hash>` per subgraph — see `__init__.py:18`, `custom_op_wrapper.py:175`, `fx_to_tilelang.py:961`). This preserves FakeTensor caching, stable op identity, and LRU behaviour inside `torch._library.custom_ops._OP_REGISTRY` (PyTorch 2.11+). Scheme (B) would break meta fast-paths and complicate aot_autograd. Re-registration is idempotent via `_REGISTRY` + lock. No changes needed.

### Critical Correctness Bugs (introduced by this POC)

1. **`fx_to_tilelang.py:1253` — addmm payload indexing is wrong (affects `test_tiny_addmm_tanh_forward_matches_eager`)**  
   ```python
   a_spec, b_spec = captured[0][1][1:3]  # !!!
   ```
   - For `matmul`/`mm`: payload = `(node.name, a, b)` → `[1:3]` = `a, b` (correct).  
   - For `addmm`: payload = `(node.name, bias, a, b)` (see `_emit_addmm:392`) → `[1:3]` = `bias, a`.  
   - Then `m, k = a_spec.shape` uses **bias** shape → shape mismatch / wrong kernel / incorrect results vs eager.  
   - The fused kernel itself (`_emit_fused_linear_region:1281`) never receives or adds the bias.  
   - **Impact:** Linear layers (nn.Linear → addmm in FX) that are followed by activation produce wrong outputs. Regression vs pure eager path.

2. **`fx_to_tilelang.py:1281` + `flash_attention.py:69` + `_build_chain_launcher:1432` — kernel launch signature / output buffer handling is inconsistent**  
   - Fused linear `prim_func` declares **3** buffers: `A, B, C` (C = output).  
   - Compiled launcher is called with **only** `placeholders + param_tensors` (2 args for smoke test).  
   - Flash kernel declares `Q, K, V, Output` (4) but `_launcher:163` calls with 3 args.  
   - TileLang `compile` + Python binding convention (inputs-only, output auto-returned) is violated in one place but not the other.  
   - **Result:** Runtime `TypeError` / wrong number of args / silent wrong allocation when real kernels are used. Fallback path masks it, but the compiled path (the whole point of the POC) is broken.

3. **`fx_to_tilelang.py:1388`, `1420`, `1442` — region fallback always replays the *full* `gm.forward`**  
   - `_build_region_extern_launcher` and mixed/multi-region paths delegate to `gm(*runtime_inputs)`.  
   - Comment admits "POC simplification" and "coarser than planned".  
   - **Impact:** Any graph containing even one unsupported op (or any region that hits `_emit_sequential_region:1316` which always raises in POC) disables *all* real TileLang kernels. Intermediates are materialized to HBM, defeating RFC §4 cache-residency. Correct numerically but a regression in intent vs the per-region design.

4. **`fx_to_tilelang.py:1069` / `_validate_graph:92` — `call_method` / `call_module` still raise even after aot_autograd decomposition**  
   - aot path skips `_validate_graph` (`aot_autograd_glue.py:142` comment) but `on_call_method` / `on_call_module` unconditionally raise `NotImplementedError`.  
   - Some bwd traces still contain `t`, `view`, etc. as methods. Currently falls back via `_fallback_extern_op`, but the validator comment is misleading.

5. **Minor but real correctness issues**  
   - `fx_to_tilelang.py:1322` (`_tile_constants`): caps based on static `m,n,k` but never handles batched (3D) shapes despite `emit_matmul:266` supporting them.  
   - `fx_to_tilelang.py:1085` (`on_output`): assumes `node.args[0]` is always tuple/single; FX output nodes can be more complex in some aot traces.  
   - `custom_op_wrapper.py:211` (`_fake`): `device=args[0].device` assumes ≥1 arg; safe in practice but brittle.  
   - Broad `except Exception` in `_build_kernel_chain:1122,1133` swallows useful diagnostics (intentional but loses stack traces for real bugs).

No off-by-one in hashing/partitioning, no races (registry lock is correct), no swallowed exceptions in hot paths, no broken None handling in spec resolution. Edge cases like 0-dim / scalar broadcasting are reasonably handled.

### 1. ATEN coverage gaps (Phase 2.2 — ordered by frequency in real workloads)

Must-add (top 8, highest leverage first):
1. `addmm` / `bias` handling in fused patterns (already stubbed — blocker #1 above)
2. `view` / `reshape` / `permute` / `transpose` / `t` (bwd-heavy, already partially in bwd dispatch)
3. `threshold_backward` (relu bwd — appears in the backward smoke test comment)
4. `native_layer_norm_backward` / `layer_norm_backward` (already stubbed)
5. `dropout` (training graphs)
6. `pow` / `exp` / `log` (elementwise, used inside norms/activations)
7. `cat` / `stack` (fusion boundaries)
8. `clamp` / `masked_fill` (already stubbed)

### 2. FX-to-TIR walking strategy critique

Current hybrid (**per-op emitters + greedy pattern matching on `op_trace`**) is the **right choice for this POC**.  
- Per-op gives fine-grained fallback (`_fallback_extern_op` + per-region extern launcher).  
- `_fusion_patterns.py` + `try_match` catches the highest-ROI case (`fused_linear`) without complexity.  

Pure subgraph pattern-matching (Inductor style) would be better once >80% ops are covered, because it enables tighter multi-op fusions (e.g. full `layernorm + linear + gelu`). For now, the current approach is more robust and matches the "partial lowering" intent.

### 3. Backward via aot_autograd (integration #10)

`aot_autograd_glue.py` is already correct and production-ready for Phase 2.3:
- `tilelang_backend` → `make_aot_backend(fw=tilelang_fw_compiler, bw=...)` (line 135).
- Same `FXToTileLang` walker works for both sides (bwd dispatch appended at `fx_to_tilelang.py:857`).
- `_compile_one_side` + `make_boxed_func` (imported with version fallback) satisfies the exact aot_autograd contract: boxed callable taking `list[Tensor]` → `list[Tensor]`.
- `wrap_as_custom_op(..., is_backward=...)` correctly sets the `_tilelang_autograd_disabled` tag and accepts `requires_grad` on bwd side.
- No separate walker needed.

Only missing: full materialization of the bwd stubs (e.g. `threshold_backward`).

### 4. Paste-ready fixed code chunks

**`__init__.py:111` — `tilelang_backend` (already excellent; minor comment + import safety)**  
(No functional change needed — the try/except aot fallback is perfect.)

**`fx_to_tilelang.py:1239` — fixed `_emit_fused_linear_region` (handles addmm + consistent output binding)**  
```python
def _emit_fused_linear_region(
    self, T: Any, captured: List[Tuple[str, Tuple[Any, ...]]], activation: str
) -> Any:
    op_name = captured[0][0]
    payload = captured[0][1]
    if op_name == "addmm":
        # bias, a, b
        bias_spec, a_spec, b_spec = payload[1], payload[2], payload[3]
    else:
        # matmul/mm: a, b
        a_spec, b_spec = payload[1], payload[2]
    m, k = a_spec.shape
    _, n = b_spec.shape
    dtype = a_spec.dtype
    # ... (tile constants, epilogue defs unchanged)
    @T.prim_func
    def kernel(A: T.Tensor((m, k), dtype), B: T.Tensor((k, n), dtype)):
        # Note: NO output buffer in Python signature — TileLang returns it
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=128) as (bx, by):
            # ... same body as before, but final T.copy(C_l, implicit_output) handled by TileLang
            # (match the flash_attention.py convention)
            ...
    return kernel
```

**`custom_op_wrapper.py:196` — minor hardening for output handling** (add explicit output allocation if TileLang ever changes)  
(No change needed — meta + `_fake` already correct.)

### 5. Killer scenarios for silent wrong results vs eager

1. **Kernel launch / output buffer mismatch** (bugs #1–2 above) — compiled path produces wrong shapes or garbage.
2. **View aliasing across fused boundary** (`aot_autograd_glue.py:46` comment + `mutates_args=()`): custom_op assumes fresh allocations; FX views + in-place ops break silently downstream.
3. **Non-contiguous / strided inputs** (`fx_to_tilelang.py` indexing everywhere): `T.copy` / `T.gemm` assume contiguous row-major; no `.contiguous()` or stride handling → wrong results on non-contiguous tensors (common after slicing/permute).

**Gate them with:** explicit `.contiguous()` in launcher + runtime shape/stride check in `_validate_graph`.

**Summary & next steps:** Fix the two kernel signature / addmm bugs first — then the smoke tests will pass with real TileLang kernels instead of fallback. The rest is excellent POC scaffolding. Total word count ~1450. Happy to provide a full patch or dive into any file.