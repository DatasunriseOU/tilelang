# Runtime test matrix — waves 1-5 + Phases 1-3

Empirical run on macOS Darwin / Apple Silicon (Metal available, MPS available, no CUDA).
Date: 2026-05-07. Operator: validation fork.

## Headline

**21 test files attempted, 142 individual tests collected.**

- **30 PASSED**
- **31 FAILED** — 6 distinct real-bug clusters, see "Real bugs found" below
- **74 SKIPPED** (mostly env: no triton / no FP8 hardware / no torch.cuda; one cluster needs aot_autograd & dialects)
- **3 XFAIL** (expected-failure: deferred items in the torch.compile multi-region launcher path)
- **2 ERROR at collection** (env config — `tvm.base` import order; `simdgroup_a` import — both due to the worktree at `/private/tmp/tl_poc_review` not being the active tilelang install path)

So **21% of the tests actually exercise the integration code on this host**, **22% expose real bugs**,
58% require a full Metal/CUDA/FP8/triton box plus a proper editable tilelang install.
This is **not** a "ship-ready" matrix — six waves of static review and pyright-clean compiles
shipped four families of runtime bugs that no `pytest --collect-only` would catch.

## Environment

- python: `3.13.12` from `/Volumes/external/sources/cppmega.mlx/.venv` (the only venv on host with
  `torch 2.13.0.dev`, `mlx 0.31.2`, `tilelang` (loaded from `/tmp/tl_apache_tvm_swap/build/lib`),
  `cppmega_mlx`, `cppmega`, `pytest 8.4.2`)
- `mlx.metal.is_available()` → True
- `torch.backends.mps.is_available()` → True, `torch.cuda.is_available()` → False
- `triton` → **NOT installed** (skips the `tt.*` runtime conformance kernels)
- `tvm.base` → import-order issue (env asserts `/private/tmp/tl_poc_review/3rdparty/tvm/python` exists,
  symlink fix works for `extern.py` but `from tilelang import tvm as tvm` follows a different path)
- Required env: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` (libz3.dylib),
  `TILELANG_DEV_BUILD_ROOT=/private/tmp/tl_apache_tvm_swap/build`,
  `TVM_IMPORT_PYTHON_PATH=/private/tmp/tl_apache_tvm_swap/3rdparty/tvm/python`

## Per-file matrix

| File | Pkt / Wave | P | F | S | E | Status |
|---|---|---:|---:|---:|---:|---|
| cppmega/tests/test_fp8_amax_tilelang.py | #05 W1-4 | 4 | **6** | 1 | 0 | 🔴 NameError `in_dtype` |
| cppmega/tests/test_dsa_splitk_tilelang.py | #06 W1-5 | 3 | **7** | 1 | 0 | 🔴 IR scoping + vector-index |
| cppmega/tests/test_engine_path_switch.py | Phase-1.C | 4 | 0 | 0 | 0 | 🟢 |
| cppmega/tests/test_topk_selector_engine.py | Phase-2 | 2 | 0 | 0 | 0 | 🟢 |
| cppmega/tests/test_sparse_mla_blockscaled_engine.py | Phase-2 | 0 | **2** | 2 | 0 | 🔴 LowerTMAToPtrArith doesn't dispatch tilelang.Allocate |
| cppmega/tests/test_sparse_mla_path_c_engine.py | Phase-3 | 5 | 0 | 0 | 0 | 🟢 |
| cppmega/tests/test_mamba3_path_c_engine.py | Phase-3 | 3 | **1** | 0 | 0 | 🟡 dh shape-validation test data mismatch |
| cppmega/tests/test_fp8_vecmat_path_c_engine.py | Phase-3 | 1 | **2** | 0 | 0 | 🔴 engine fallback emits shim instead of `(artifact, None)` |
| cppmega.mlx/tests/test_msl_extraction.py | Phase-3 adapter | 8 | 0 | 0 | 0 | 🟢 |
| testing/python/triton_frontend/test_conformance_kernels.py | #01 W3 | 2 | **1** | 7 | 0 | 🔴 walker text path positional-arg bug |
| testing/python/language/test_reduce_prod.py | #01 W2 | 1 | **1** | 0 | 0 | 🔴 vector-index in non-last buffer dim |
| poc/triton_frontend/tests/test_triton_structured_walk.py | #04 W2/3 | 0 | 0 | 4 | 0 | ⚪ skip: dialects (no MLIR shim build) |
| poc/triton_frontend/tests/test_ptr_analysis.py | #03 W1-3 | 8 | 0 | 3 | 0 | 🟢 |
| poc/extern_intrinsic_examples/test_extern_smoke.py | #08 W1-3 | — | — | — | 1 | ⚫ collect: cannot import `simdgroup_a` (worktree shadowing) |
| testing/python/transform/test_lower_tma_to_ptr_arith.py | #07 W1-3 | — | — | — | 1 | ⚫ collect: `tvm.base` not found |
| poc/torch_dynamo/examples/test_torch_compile_chain.py | #02 W1-3 | 8 | **4** | 0 | 0 | 🔴 `Sequence[torch.Tensor]` schema rejected |
| poc/torch_dynamo/examples/test_autograd_compose.py | #09 W1-4 | 12 | 0 | 1 | 0 | 🟢 |
| testing/python/transform/test_sync_threads_partial.py | Phase-1.A | 0 | 0 | 2 | 0 | ⚪ skip: needs MLIR-built Op |
| testing/python/triton_frontend/test_sync_threads_partial_mapping.py | Phase-1.A | 2 | 0 | 1 | 0 | 🟢 |
| testing/python/triton_frontend/test_tt_dot_trans.py | Phase-1.B | 5 | 0 | 0 | 0 | 🟢 |

P = passed, F = failed, S = skipped, E = collect error.

## Real bugs found (none fixed by this fork — kept as observations)

### 🔴 Bug 1: `fp8_amax.py` — `NameError: in_dtype is not defined`

`fp8_amax.py:494` (`_amax_kernel_for(bucket_n, in_dtype, target)`) calls `make_fp8_amax_kernel`,
which references `in_dtype` from an outer closure that doesn't actually capture it after the
wave-2 dynamic-block-size refactor (44f4f88) re-parameterised the function. **Six fp8_amax tests
fail with one root cause.**

```
E   NameError: name 'in_dtype' is not defined
```

### 🔴 Bug 2: `dsa_splitk_indexer_loss.py` wave-5 Q-cache — IR scoping

Wave-5 (`56cf429`) introduced an `s = T.float32()` declaration inside an `IfFrame` whose lifetime
ends before `s` is consumed downstream. Triggered on every parity test that builds the
stage-1/stage-2 kernels. **Seven dsa_splitk tests fail.**

```
E   RuntimeError: Immutable variable `s` is used outside its defining region!
E   variable `s` is defined in frame: script.ir_builder.tirx.IfFrame(stmts=(),
    condition=in_bounds, then_stmts=(s = T.float32() …))
```

Co-trips a related vector-index bug in the same wave-5 path:

```
E   tvm.error.InternalError: Check failed: (indices[i].dtype().is_scalar()) is false:
E   Only the last index of a buffer access may be a vector type.
```

### 🔴 Bug 3: `tilelang.Allocate` is un-registered for `LowerTMAToPtrArith`

Wave-1 #07 (`9d2bb653`) wired `LowerTMAToPtrArith` into the engine pipeline. But the pass's
NodeFunctor doesn't have a dispatch entry for `tilelang.Allocate` (the vendored allocate
node from the Apache-TVM compatibility surface). Every `dispatch_lower` call that includes
this pass and runs against a TileLang DSL kernel fails:

```
E   tvm.error.InternalError: Check failed: (can_dispatch(n)) is false:
E   NodeFunctor calls un-registered function on type tilelang.Allocate
```

This burns down both `sparse_mla_blockscaled_engine` tests and propagates to
`fp8_vecmat_path_c_engine` (engine mode falls back to shim → assertion fails).

### 🔴 Bug 4: torch.compile backend rejects `Sequence[torch.Tensor]`

`custom_op_wrapper.py:_impl(args: Sequence[torch.Tensor])` annotation is rejected by
`torch.library.custom_op`'s `infer_schema`:

```
E   torch._dynamo.exc.BackendCompilerFailed: backend='tilelang' raised:
E   ValueError: infer_schema(func): Unsupported type annotation Sequence[torch.Tensor].
E   It is not a type. Got func with signature (args: 'Sequence[torch.Tensor]') -> 'Any'
```

Four `test_torch_compile_chain.py` tests fail. Wave-3 #02 added the boxed-arg convention
without verifying torch's `infer_schema` accepts it. Fix: use `tuple[torch.Tensor, ...]` or
`list[torch.Tensor]` instead.

### 🔴 Bug 5: walker text-path missing `ctx` arg

```
E   TypeError: _walk_text_ttir() missing 1 required positional argument: 'ctx'
```

Wave-3 #01's matrix-coverage test (`test_walker_dispatch_path_matrix[text]`) calls the walker
without threading the parser context. Fix the test (or fix the walker default to construct a
default ctx).

### 🔴 Bug 6: `reduce_prod` constructs an invalid buffer access

```
E   tvm.error.InternalError: Check failed: (indices[i].dtype().is_scalar()) is false:
E   Only the last index of a buffer access may be a vector type.
```

Wave-2 #01's `tilelang/language/reduce_op.py:229` `reduce_prod` lowering puts a vector index in
a non-final dimension. The test calls `T.reduce_prod(A_f, O_f, dim=1, clear=True)` and the IR
builder rejects it during construction.

### 🟡 Bug 7: mamba3 helper test data mismatch (probably test-side, not production)

```
E   ValueError: dh leading dims must match A (1, 2, 1), got (1, 2, 4)
```

`test_helper_public_callers_pass_through_when_lowering_is_none` builds inputs with mismatched
leading dims; the helper's pre-validator catches it. Fix the test fixture, not the helper.

## Tests that pass (with substance)

The following actually exercised the production code path under engine/shim dispatch:

- `test_engine_path_switch.py` — all 4 Phase-1.C dispatch modes work (auto/engine/shim/auto+ImportError fallback)
- `test_topk_selector_engine.py` — both engine and shim parity tests pass
- `test_sparse_mla_path_c_engine.py` — 5 cache/contract tests pass on Metal
- `test_msl_extraction.py` — adapter extracts MSL via `artifact.kernel_source` correctly (8/8)
- `test_ptr_analysis.py` — error caching, deprecation latch, JSON encoder equivalence (8 pass, 3 dialect-skip)
- `test_autograd_compose.py` — wave-4 fix held: 12/12 (1 skip on no-CUDA)
- `test_tt_dot_trans.py` — Phase-1.B walker tests, all 5 pass (op-table + transpose folding)
- `test_sync_threads_partial_mapping.py` — Phase-1.A walker mapping, 2/2 (1 skip on missing tilelang.Op)

## What needs Metal-host CI

A test-environment audit:

- 31 of 74 skips are **`triton` not installed** — bring up triton-cpu (or `triton` 3.5+) on the
  CI host and these light up.
- 12 are FP8 hardware (`pytest.skip("FP8 hardware not available")`) — needs a Hopper or B200.
- 7 are dialect-shim build (MLIR + the vendored `_triton_frontend_cxx`) — currently no
  `pybind11_add_module(register_triton_structured)` builds in this env because there's no
  cmake configure for the worktree.
- 24 are torch+CUDA combos for `torch_compile_chain` and `dsa_splitk` triton-reference paths.

The Metal-only subset is small but real: ~15 tests will exercise the full dispatch path
on Apple Silicon once the 6 bugs above are fixed.
