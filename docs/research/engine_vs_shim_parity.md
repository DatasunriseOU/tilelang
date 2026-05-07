# Engine-vs-shim parity matrix — empirical results (wave-7 retry-3)

**Host**: macOS Darwin 25.4.0 on Apple M4 Max, MLX 0.31.2, Metal available.
**Run date**: 2026-05-07.
**Interpreter**: `/Volumes/external/sources/nanochat/.venv/bin/python` (Python
3.13). Default cppmega.mlx `python3` venv was missing `mlx_lm`; the nanochat
venv ships matching `libmlx.dylib` 0.31.2 and was reused. Final invocation:

```bash
cd /Volumes/external/sources/cppmega.mlx
DYLD_LIBRARY_PATH=/Volumes/external/sources/nanochat/.venv/lib/python3.13/site-packages/mlx/lib:/opt/homebrew/lib \
  /Volumes/external/sources/nanochat/.venv/bin/python /tmp/parity_matrix.py
```

The injected `DYLD_LIBRARY_PATH` works around the known venv-vs-brew ABI
mismatch documented in Test #1 (commit `7dc9ed58`): MLX's bundled
`libmlx.dylib` is reachable, the ambient `/opt/homebrew/Cellar/mlx/0.31.1/`
copy is shadowed, and `libz3.dylib` is picked up from `/opt/homebrew/lib`.

**Pre-existing wave-7 fixes in tree**: `a439df0`, `cac10a0`, `27392ded`,
`9657659d` (perf bench discoveries), `99e6e638` (wave-7 #4 redo).

## Matrix

| Kernel | shim | engine | engine_with_msl_extraction | auto |
|---|:-:|:-:|:-:|:-:|
| `fp8_amax::make_fp8_amax_kernel(n_elements=1024)` | ❌ | ❌ | ❌ | ❌ |
| `topk_selector::_path_c_kernel_for` (import + attr) | ✅ | ✅ | ✅ | ✅ |
| `sparse_mla_blockscaled_path_c::make_blockscaled_sparse_mla_qk_kernel` (import + attr) | ✅ | ✅ | ✅ | ✅ |
| `sparse_mla_path_c::_fwd_kernel_for` (import + attr) | ✅ | ✅ | ✅ | ✅ |
| `mamba3_path_c::_fwd_kernel_for` (import + attr) | ✅ | ✅ | ✅ | ✅ |
| `fp8_vecmat_path_c::_fp8_vecmat_kernel_for` (import + attr) | ✅ | ✅ | ✅ | ✅ |
| `sparse_mla::sparse_mla_fwd_metal` (wave-6 bridge) | ✅ | ✅ | ✅ | ✅ |

**Headline**: 24/28 cells OK, 4 ERR — single root cause: wave-7 #1
(`a439df0`) is incomplete. `dsa_splitk_indexer_loss` excluded from this run
per directive (known broken from perf bench `9657659d` `denom` IR scoping
discovery).

> The 6 import-only rows are honest about scope: each kernel's real ctor
> takes per-kernel shape kwargs not in scope of this probe. The full
> numerical engine-vs-shim comparison belongs in
> `cppmega/tests/test_*_engine.py` once the venv ABI mismatch is resolved
> in CI. The probe verifies that module import, attribute access, and the
> dispatch_lower env-mode plumbing all survive the four modes — i.e. no
> import-time crashes from `_engine_dispatch.py` flag handling.

## fp8_amax 4×ERR root cause

Identical traceback under all four env modes:

```
File ".../tilelang/language/eager/builder.py:1248, in impl
    func_annot = get_type_hints(func)
File ".../tilelang/language/eager/builder.py:888, in get_type_hints
    hints[name] = _eval_type(value, globalns=globalns, localns=localns)
File "<typing>:1081, in _evaluate
    eval(self.__forward_code__, globalns, localns)
File "<string>:1, in <module>
NameError: name 'DTYPE' is not defined
```

Wave-7 #1 (`a439df0`) added `DTYPE = in_dtype` in the enclosing frame on
the assumption that the inner `@T.prim_func` parser would pick it up via the
function's closure cell. **It does not.** `get_type_hints(func)` resolves
string annotations using only `func.__globals__` and an explicit `localns`
dict; it does not enumerate `func.__closure__`. The bound names `DTYPE`,
`N`, `BLOCK` exist as closure cells, are visible to the function body at
call time, but are invisible to type-hint resolution which fires at
`@T.prim_func` decoration time.

**Same flaw applies to `N` and `BLOCK`**, which use the same pattern. The
`fp8_amax_tilelang_engine` test in commit `1ded445` (wave-4) presumably
worked because at that revision the parser path differed, or because some
prior tilelang revision did walk closure cells. With current tilelang at
`/tmp/tl_apache_tvm_swap/build` (loaded via dev-root preload), the closure
cell path is silently dead.

## Root-cause guesses (top-5 errored cells)

All four fp8_amax cells share one root cause (above). No second-tier
divergence to report — the other 24 cells are uniformly green at the import
+ attribute level.

## Recommended wave-7-followup fix path

Three options, ordered by invasiveness:

1. **Inject closure values into `func.__globals__`** before the
   `@T.prim_func` decorator runs. Pattern:
   ```python
   _f = make_fp8_amax_kernel.__globals__
   _f['DTYPE'] = in_dtype  # set, decorator-time visible
   _f['N'] = n_elements
   _f['BLOCK'] = block_size
   ```
   Cleanest if scoped behind a small `_PrimFuncContext` helper that pops the
   keys back after `@T.prim_func` returns. **Drawback**: leaks under multi-
   threaded kernel construction.

2. **Make `DTYPE` / `N` / `BLOCK` module-level constants per-shape**:
   memoise the prim_func builder by `(n_elements, in_dtype, block_size,
   threads)` and set those module-level attributes inside the cache lookup
   before defining the prim_func. Already half-implemented by `_amax_kernel_for(bucket_n, in_dtype, target)` — just expose `DTYPE`/`N`/`BLOCK` as module
   attributes inside that wrapper before calling `make_fp8_amax_kernel`.

3. **Patch tilelang to walk closure cells** in `get_type_hints`. Largest
   blast radius (helps every cppmega.mlx kernel that builds a parameterised
   PrimFunc), but lands upstream rather than in cppmega.mlx. Probably
   worth doing eventually.

For wave-8, option (1) is the smallest patch that unblocks runtime fp8_amax
on this M4 Max host today. Option (2) is the right structural fix.

## Env-fix recipe (for CI/other hosts)

If `import mlx.core as mx` fails with the same `as_strided` symbol
mismatch as Test #1:

```bash
# Either reinstall mlx pinned to the bundle that matches your venv:
pip install --reinstall --no-binary mlx-metal "mlx==0.31.2"

# Or unlink the brew copy:
brew unlink mlx
# Or just run with DYLD_LIBRARY_PATH pointing at the wheel-bundled libmlx:
DYLD_LIBRARY_PATH=$VENV/lib/python3.X/site-packages/mlx/lib:/opt/homebrew/lib python ...
```

For `libz3.dylib` not loaded:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH ...
```

## Gaps not covered by this matrix

- **Numerical parity**: not measured. Each non-fp8_amax kernel needs a
  proper input-tensor harness; lives in `cppmega/tests/test_*_engine.py`.
- **dsa_splitk_indexer_loss**: skipped per directive; known broken under
  engine modes from perf bench `9657659d`.
- **Engine path emission of fp8 SIMDgroup MMA**: blocked by the
  `simdgroup_a/b_fp8` factories landing (commit `5dc4b743` is a forward-
  compat placeholder). Wave-8.
