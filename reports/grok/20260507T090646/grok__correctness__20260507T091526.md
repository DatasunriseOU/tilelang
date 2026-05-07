---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T09:15:26.029887+00:00
files: ['/private/tmp/tl_poc_review/poc/torch_dynamo/aot_autograd_glue.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/__init__.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/examples/test_autograd_compose.py']
---
**WAVE-3 FINAL REGRESSION REVIEW (aot_autograd_glue.py, __init__.py, custom_op_wrapper.py, test_autograd_compose.py)**

Wave-1 + wave-2 fixes have **not** introduced regressions or half-fixes. Key examples (all still correct post-wave-3):
- `custom_op_wrapper.py:81-120` (`_ensure_contiguous_inputs`): view-aliasing + non-contiguous handling + warn-once cache work exactly as documented (including the aliased+already-contiguous `clone()` optimization).
- `aot_autograd_glue.py:130` + `_compile_one_side:210ish` (`_validate_graph` call): now correctly covers both fwd and bwd paths (no regression to the legacy forward-only path in `__init__.py:120`).
- `_REGISTRY_LOCK` + `_REGISTRY` idempotency in `custom_op_wrapper.py:220-250` and `_bind_runtime`: solid.
- Lazy imports, `_check_no_grad(allow_grad=...)`, `_fake`, contiguity lock, and test surface in `test_autograd_compose.py` are all clean.
- No new off-by-one errors, mismatched types, broken `None`/edge-case handling, or critical swallowed exceptions in core paths (the many `try/except Exception` blocks are intentional best-effort as documented for wave-3 symbolic/autotune/double-bwd).

**However, the package is NOT ready to ship.** Three (possibly four) **HIGH-severity correctness bugs** remain. These appear to be either introduced by or exposed by the wave-3 changes (symbolic tiles, `register_double_backward`, atomic heuristic, etc.). They break the claimed double-backward / `torch.func.grad(grad(f))` composability and dynamic-shape paths.

### HIGH-SEVERITY BUGS (reference exact locations in current code)

1. **Infinite recursion on symbolic-shape fallback** (aot_autograd_glue.py: compile_symbolic + _compile_one_side)  
   ```python
   # aot_autograd_glue.py: ~265 (inside compile_symbolic)
   except Exception as exc:
       ...
       return _compile_one_side(gm, example_inputs, is_backward=is_backward)  # !!!
   ```
   and
   ```python
   # aot_autograd_glue.py: ~230 (inside _compile_one_side)
   if any(_has_symint_shape(ex) for ex in example_inputs):
       return compile_symbolic(...)   # calls back into the above
   ```
   If the wave-3 symbolic path in `FXToTileLang` raises (explicitly documented as "best-effort" / "walker integration isn't ready"), the fallback recurses infinitely → `RecursionError`. Affects any SymInt-bearing graph (dynamic shapes). The one-shot `RuntimeWarning` never fires reliably.

2. **Double-backward registration is dead (never wires `register_autograd`)** (aot_autograd_glue.py: register_double_backward + caller)  
   ```python
   # aot_autograd_glue.py: ~175-180 (register_double_backward)
   bwd_op = _REGISTRY.get(bwd_op_qualname)
   if bwd_op is None:  # pragma: no cover
       return
   ```
   and the call site:
   ```python
   # aot_autograd_glue.py: ~320-330 (in _compile_one_side, only when not is_backward)
   if not is_backward:
       ...
       register_double_backward(fwd_qualname, bwd_qualname, ...)
   ```
   `aot_autograd` (and `torch.func`) calls `fw_compiler` **before** `bw_compiler`. The bwd custom_op is therefore never in `_REGISTRY` when the fwd path runs → registration is silently skipped every time. The entire wave-3 double-backward feature (including the atomic zero-grad heuristic) is non-functional. The comment claiming "by the time we see the bwd graph the partner is already registered" is inverted.

3. **Bwd-op call inside double-bwd has arity / saved-tensor mismatch** (aot_autograd_glue.py: backward closure)  
   ```python
   # aot_autograd_glue.py: ~205-230 (inner backward fn)
   saved = list(ctx.saved_tensors)          # ALL fwd inputs (from setup_context)
   ...
   return bwd_op(saved + list(tangents))    # !!!
   ```
   `setup_context` does `ctx.save_for_backward(*inputs[0])` (all original fwd inputs). But the bwd `GraphModule` (produced by aot_autograd) only saves the *subset* of tensors actually needed by the partitioned backward + any intermediates. The bwd custom_op therefore expects a **different** argument list (aot-saved tensors + tangents). Passing the full saved list either:
   - causes TypeError (wrong arg count), or
   - feeds wrong values → incorrect second derivatives.
   The `len(saved) == 1` atomic special-case only masks one narrow scenario and is not a general fix.

4. **backward closure signature does not support multi-output fused ops** (aot_autograd_glue.py: register_double_backward)  
   ```python
   # aot_autograd_glue.py: ~200
   def backward(ctx, grad) -> Any:   # single 'grad' param
   ```
   PyTorch's `register_autograd` calls the backward function with `*grad_outputs` (one positional arg per fwd output). Multi-output fwd ops (very common: matmul+add, attention, etc.) will raise `TypeError: backward() takes 2 positional arguments but 3 were given` (or similar). The later `tangents = grad if isinstance...` line never runs for these cases.

These four bugs make the wave-3 autograd claims (double-backward, symbolic tiles, atomic accumulator handling) non-functional in practice. No other high-severity correctness issues (race conditions, null handling, etc.) were found in the reviewed files.

**Performance notes (correctness-adjacent)**:  
- Autotune cache + `_shape_bucket` + `specialize_prim_func` are fine (best-effort as documented).  
- Contiguity guard in `custom_op_wrapper.py` is correct and does not regress numerics.  

Fix the four items above (plus the dead `ctx._tilelang_has_atomic` and brittle `str(node.target)` atomic detection) and this package will be green for shipping. Current state is **not** ready.