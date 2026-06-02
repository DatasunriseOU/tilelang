# cpython-3.13 / tirx: `tilelang.jit(PrimFunc)` -> "... is not a callable object"

## Symptom

Building `examples/deepseek_v32/sparse_mla_fwd.py` (and any TileLang kernel that
feeds an already-built `PrimFunc` to `tilelang.jit`) raised, at TIR construction
time (before lowering):

```
TypeError: '# from tvm.script import tirx as T
@T.prim_func
def main(... ) ...
    T.copy(T.region(acc_o[0, 0], 1, 64, 512), ...)
    ...' is not a callable object
```

The error text embeds the printed TVMScript of the kernel (including the
`T.copy(...)` lines), which is why a prior investigator described it as
"`T.copy(...)` is not a callable object". The `T.copy` text is a red herring: it
is just part of `repr(PrimFunc)`. The real object that is "not a callable" is the
**`PrimFunc` itself**.

## Exact failure site

```
tilelang/jit/__init__.py:603   decorator()  ->  prim_func(func, eager_jit=True)
tilelang/language/eager/builder.py:1256  impl()  ->  sig = inspect.signature(func)
.../python3.13/inspect.py:2519  raise TypeError(f'{obj!r} is not a callable object')
```

`tilelang.jit`'s public signature is

```python
def jit(func: Callable[_P, _T] | PrimFunc | None = None, *, ...)
```

i.e. it explicitly accepts an already-built `PrimFunc` (the lazy result of a
builder). But the inner `decorator(func)` never branched on that: it
unconditionally ran `prim_func(func, eager_jit=True)`, which calls
`inspect.signature(func)` / `inspect.getsource(func)` assuming `func` is a Python
callable. When `func` is a `PrimFunc`, `inspect.signature` raises
`TypeError(f'{obj!r} is not a callable object')`.

## Root cause: tir -> tirx migration, NOT cpython 3.13 per se

This is a latent bug exposed by the `[TIR][IR] Update to use tirx` migration
(commit 1b66dbfb), not by cpython 3.13's inspect changes:

* Before the migration, `tvm.tir.PrimFunc` was a callable tvm-FFI `Object`
  (`callable(pf) is True`), so `inspect.signature(pf)` succeeded and the missing
  branch in `jit` went unnoticed.
* After the migration, `tvm.tirx.function.PrimFunc` (to which both
  `tvm.tir.PrimFunc` and `tvm.tirx.PrimFunc` are now aliased) has **no
  `__call__`**, so `callable(pf) is False`. `inspect.signature` then takes the
  non-callable branch and raises.

cpython only enters the picture because 3.13's `inspect._signature_from_callable`
is where the `TypeError('... is not a callable object')` is raised; the same
non-callable `PrimFunc` reaches that line on 3.12 too. The trigger is the tirx
`PrimFunc` losing `__call__`, surfacing the never-handled `jit(PrimFunc)` path.

Reproduce (python3.13, this repo):

```python
import tilelang
import examples.deepseek_v32.sparse_mla_fwd as m
pf = m._build_sparse_mla_fwd(128,512,64,2048,1,None,True,True,64,2,256,False,None)
assert type(pf).__name__ == "PrimFunc" and callable(pf) is False
tilelang.jit(pf)        # -> TypeError: '<TVMScript>' is not a callable object
```

## Fix

`tilelang/jit/__init__.py` — handle the `PrimFunc` input explicitly, mirroring
`compile()` and `JITImpl.get_tir`, both of which already branch on
`isinstance(..., PrimFunc)`:

1. `decorator(func)`: if `isinstance(func, PrimFunc)`, build a `JITImpl` in
   `mode="lazy"` with `func_source=func.script()` and an empty
   `inspect.Signature()`, instead of running the eager `prim_func(...)` /
   `inspect.signature(func)` path.
2. `JITImpl.initialize_jit_mode`: only call `self.func.set_mode(...)` when
   `self.func` is a `JITFunc` (a raw `PrimFunc` has no `set_mode`; it is
   implicitly lazy).
3. `JITImpl.__call__`: short-circuit for a `PrimFunc` input — the compiled kernel
   does not depend on call args, so use a constant cache key, call
   `self.compile()`, and return the kernel (no `set_mode`/`parse_args`, which live
   on `JITFunc`).

This is a real, fail-loud fix: it does not swallow errors or add a degraded
fallback. The non-PrimFunc (`Callable`) path is unchanged, so eager-builder and
lazy-builder kernels behave exactly as before.

## Verification (python3.13, nanochat venv)

* `jit(PrimFunc)`, `jit(out_idx=...)(PrimFunc)`, and `JITImpl.get_tir()` all
  succeed for the deepseek sparse-MLA fwd across `gb10 in {False, True}` x
  `static_shape in {dynamic, static}` — TIR construction passes; the only
  remaining failure is the expected Metal/CUDA target-codegen issue at lowering
  (out of scope on Mac: no CUDA copy-impl registered / Metal GEMM-policy guard).
* No regression: eager-style, lazy-style, and direct-PrimFunc jit paths verified;
  `testing/python/metal/test_metal_codegen.py` (3), `test_metal_reduce.py` and
  `test_metal_local_var.py` (56) all pass.
