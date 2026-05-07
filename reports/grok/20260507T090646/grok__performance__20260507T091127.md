---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T09:11:27.465326+00:00
files: ['/private/tmp/tl_poc_review/poc/torch_dynamo/aot_autograd_glue.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/__init__.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/examples/test_autograd_compose.py']
---
**WAVE-3 FINAL REGRESSION REVIEW — STATUS: NOT SHIPPABLE (2 high-severity items)**

I reviewed the exact current state of the four attached files (`aot_autograd_glue.py`, `__init__.py`, `custom_op_wrapper.py`, `test_autograd_compose.py`). Wave-1/2/3 fixes (contiguity guard improvements, symbolic branch, atomic double-bwd handling, `_has_params` fast-path, `_validate_graph` unification, etc.) have been applied cleanly with no half-fixes or new correctness regressions in the provided test surface. The `aot_autograd` wiring, `_REGISTRY` caching, and forward-only fallback path behave as intended when torch is present.

However, two **HIGH-severity issues** remain in the *current code* (runtime hot path + critical regression). Both are concrete, quantifiable, and directly traceable to wave-2/3 additions. No O(n²), I/O, N+1, async blocking, or unbounded memory growth elsewhere.

### 1. CRITICAL BUG (recursion → stack overflow on SymInt graphs) — `aot_autograd_glue.py`
**Location**: `aot_autograd_glue.py:compile_symbolic` (the `except Exception` block, immediately after the `try: ... lowerer.run()` block, ~lines 280-300 in the pasted source).

```python
except Exception as exc:  # pragma: no cover - best effort
    _w.warn(...)
    return _compile_one_side(gm, example_inputs, is_backward=is_backward)  # !!!
```

`_compile_one_side` (lines ~220-230) does:

```python
if any(_has_symint_shape(ex) for ex in example_inputs):
    return compile_symbolic(...)  # loops back
```

**Impact**: Any graph with `SymInt` dims that hits the (explicitly documented "best-effort") walker failure path in `compile_symbolic` → infinite recursion. This is a *new regression* introduced by the wave-3 symbolic path. Previous waves had no such branch. Dynamic-shape workloads (common with `torch.compile`) will now crash hard instead of falling back gracefully.

### 2. HIGH-SEVERITY PERF REGRESSION (hot-path allocations on *every* fused kernel call) — `custom_op_wrapper.py`
**Location**: `custom_op_wrapper.py:_ensure_contiguous_inputs` (full function body, lines ~85-125) + its call site in `_impl` (line ~170) + `_runner` (lines ~200-205).

Concrete problems in the *absolute runtime hot path* (executed once per fused kernel invocation during training/inference):

- `new_seen = set(_CONTIGUITY_WARN_SEEN)` **on every single call** (set copy from the global frozenset, even when no warning ever fires). After a few unique ops appear, this becomes O(#previously-seen-warnings) work per call.
- Always does `fixed: list = []`, `enumerate(tensors)`, `append(...)` for *every* input, then `return tuple(fixed)`. No early-exit "all good" path.
- Multiple `try/except` per tensor (`hasattr`, `is_contiguous`, `getattr(t, "_base")`).
- Aliased+contiguous case (common with views/slices) still forces `.contiguous().clone()` every call (full memcpy + extra alloc).

**Quantified impact**: For a typical model with 50-200 fused kernels per forward pass this is hundreds-to-thousands of small Python allocations + set copies + potential memcpys *per inference/training step*. This is a clear regression vs. the pre-wave-2 direct-launcher path (which had no such guards/allocs). The "Wave-3 perf fix" comment improved the clone branch but left the structural allocs untouched.

Compounding this (same file):

- `_bind_runtime._runner` (lines ~200+): `op(list(runtime_inputs))` → extra `list()` alloc from the `*args` tuple on *every* call.
- `_impl` then does a second `_check_no_grad` (even though `_runner` already did it for the fwd path).

These are the only hot-path concerns. Everything else (caches, registry lookups, `_AUTOTUNE_CACHE`, `_REGISTRY_LOCK` only on change, compile-time `_validate_graph`/`has_atomic`, `_import_make_boxed_func`, `specialize_prim_func`, etc.) is either once-per-compile or properly cached and introduces no measurable overhead.

**Verdict**: Not green / not ship-ready yet. The two items above must be fixed (recursion → proper one-shot fallback; hot-path allocs → early-exit + avoid set-copy-every-call) before we can mark this clean. No other high-severity issues. The rest of the wave-1/2/3 changes landed solidly.