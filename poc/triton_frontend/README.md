# Triton -> TileLang TIR frontend (POC)

**Status: post-Wave-3 working frontend.** The package now lowers a
captured Triton TTIR module to a TileLang-shaped `tvm.tir.PrimFunc`,
runs the resulting kernel through the TileLang -> Metal -> MLX adapter,
and round-trips numeric outputs against the original `@triton.jit`
reference for the four end-to-end targets (`vector_add`, `softmax`,
`matmul`, `layer_norm`). The "every public function raises
`NotImplementedError`" disclaimer that opened earlier revisions of this
file no longer applies; see `## Current limitations` below for what is
still degraded.

This package implements the design in
[`../../RFC_unified_fused_kernel.md`](../../RFC_unified_fused_kernel.md),
specifically sections 5 (Triton -> TileLang mapper plan) and 6 (cross-source
extern intrinsic mechanism).

## Pipeline at a glance

```
+------------------+        +------------------+        +-------------------+
|  @triton.jit fn  |  --->  |   Triton TTIR    |  --->  | TileLang PrimFunc |
|  (Python)        |        |  (mlir.Module)   |        |   (tvm.tir)       |
+------------------+        +------------------+        +-------------------+
        |                            |                            |
        | from_triton_kernel()       | from_ttir()                | pipeline.run()
        |                            |                            |
        v                            v                            v
   triton.compile             walker over tt.* ops          TileLang TIR passes
   -> ttir module             dispatch via op_mapping       (LayoutInference,
   (Triton 3.6: capture       (84 entries in OP_TABLE)       LowerTileOp, ...)
    via make_ir(target,                |
    options, codegen,                  v
    module_map, ctx))         +----------------------+
                              | jaxlib alias bootstrap|
                              | (mlir.ir bindings    |
                              |  reused from jaxlib) |
                              +----------------------+
                                       |
                                       v
                         +-----------------------------+
                         | Custom -> generic round-trip |
                         | via _triton_frontend_cxx    |
                         | Module.to_generic() so the  |
                         | walker sees stable text     |
                         +-----------------------------+
                                       |
                                       v
                       +---------------+---------------+
                       |               |               |
                     CUDA            HIP/ROCm      Metal/SIMDgroup
                       |               |               |
                       +---------------+---------------+
                                       |
                                       v
                           TileLang -> Metal -> MLX
                          runtime adapter (numeric
                          parity vs Triton reference)
```

## File layout

| File                                  | Purpose                                                                   | RFC ref       |
|---------------------------------------|---------------------------------------------------------------------------|---------------|
| `__init__.py`                         | Public API: `from_triton_kernel`, `from_ttir`.                            | section 5     |
| `op_mapping.py`                       | Dispatch table: TTIR op name -> emitter. **84 entries** post-Wave-3.      | section 5.1   |
| `op_emitters/arith.py`                | Float/int/math arithmetic + comparison emitters.                          | section 5.1   |
| `op_emitters/memory.py`               | `tt.load`/`tt.store`/`tt.addptr`/range/splat/broadcast emitters.          | section 5.1   |
| `op_emitters/reduction.py`            | `tt.dot`, `tt.reduce`, `tt.atomic_*` emitters.                            | section 5.1   |
| `op_emitters/control.py`              | `scf.for`/`scf.if`/`scf.yield` emitters.                                  | section 5.1   |
| `mlir_walker.py`                      | jaxlib-shape MLIR walker driving `OP_TABLE`.                              | section 5     |
| `ptr_analysis.py`                     | Wrapper over vendored `triton-shared` `PtrAnalysis` (C++ shim).           | sections 3, 7 |
| `lowering_passes.py`                  | TTIR-side preparation passes (custom -> generic, structured walk).        | section 5     |
| `layout.py`                           | TTGIR encoding translators (placeholder; not used in MVP path).           | section 5.2   |
| `pipeline.py`                         | Ordered TileLang TIR transform passes (reuse / extend / skip).            | section 3     |
| `_test_harness/numeric_smoke.py`      | E2E numeric harness; compares TileLang output vs Triton reference.        | section 5.5   |
| `_test_harness/numeric_kernels/*.py`  | `vector_add`, `softmax`, `matmul`, `layer_norm` numeric targets.          | section 5.5   |
| `_test_harness/run_corpus.py`         | 17-kernel reducer-corpus driver (bulk regression sweep).                  | section 5.5   |
| `_cxx/`                               | pybind11 extension (`_triton_frontend_cxx`): `to_generic()`, ptr-analysis.| section 3     |
| `conformance/__init__.py`             | Reference kernels for end-to-end tests.                                   | section 5.5   |
| `vendored/triton_shared/`             | `microsoft/triton-shared` (Apache-2.0); ptr-analysis source.              | section 3     |

## Capabilities (post-Wave-3)

* **Op coverage**: 84 entries in `OP_TABLE` covering memory, arith,
  reduction, control, async/barrier, TMA, and grid/launch ops.
* **Triton 3.6 capture**: TTIR is captured via
  `make_ir(target, options, codegen, module_map, ctx)` so the captured
  module survives Triton's recent `mlir::Module` API churn.
* **jaxlib alias bootstrap**: the walker reuses jaxlib's `mlir.ir`
  bindings (the dialect tables from upstream MLIR are not directly
  available in the runtime venv); `_mlir_path_setup.py` wires this up
  on import.
* **Custom -> generic round-trip**: ops printed in their custom
  assembly form are routed through
  `_triton_frontend_cxx.Module.to_generic()` so the walker only ever
  parses the stable generic form. This avoids the parse-skew that
  custom-form printers introduce across Triton patch releases.
* **TileLang -> Metal -> MLX runtime adapter**: the pipeline's PrimFunc
  output runs end-to-end through TileLang's Metal codegen and is
  dispatched on MLX buffers; outputs are compared against the Triton
  reference inside `_test_harness/numeric_smoke.py`.
* **E2E numeric harness**: `vector_add`, `softmax`, `matmul`, and
  `layer_norm` targets pass numeric parity vs the `@triton.jit`
  reference under the harness.
* **Reducer corpus**: the 17-kernel reducer corpus (`run_corpus.py`)
  reports **17/17 LOWERED_DEGRADED** -- every kernel reaches the
  TileLang PrimFunc stage. "Degraded" here means the per-element
  fallback path was used somewhere in the kernel (tracked per kernel
  in the Wave F audit); none of the 17 fail to lower.
* **Coherent error hierarchy** (Wave G4): `EmitError` and
  `PipelineError` both subclass `TritonFrontendError`, so a single
  `except TritonFrontendError` catches every deliberate frontend
  failure without swallowing unrelated `RuntimeError`s from
  TVM/TileLang internals.

## Current limitations

These are the remaining blockers per the Wave F per-kernel audit. None
of them prevent lowering, but each one routes through the per-element
`# DEGRADED:` path and therefore costs runtime perf relative to a
hand-written TileLang kernel:

* **C++ `PtrAnalysis` shim is stub-mode by default on macOS.** Without
  `TRITON_INSTALL_DIR` set, the vendored `PtrAnalysis.cpp` is skipped
  (it transitively needs upstream Triton MLIR headers). The Python
  facade falls back to MVP scalar; multi-element tile loads do not
  fuse into `T.copy`. Build with `TRITON_INSTALL_DIR` pointing at a
  Triton install to light up the fast path.
* **`tt.atomic_*` ops** prefer the TileLang atomic intrinsic when
  importable, otherwise emit a raw `tir.atomic_*` `call_intrin`. The
  MLX adapter does not yet round-trip the raw `call_intrin` form.
* **`tt.dot`** uses `T.gemm` when `tilelang.language.gemm` is
  importable; otherwise emits a 3-loop `tir.For` nest with explicit
  `BufferStore`. The latter path is correct but unfused.
* **`tt.reduce` non-add combiners** (max, min, mul) work, but `mul`
  uses the log/exp synthesis when `_USE_LOGEXP_PROD` is `True` -- the
  native `reduce_prod` path is gated by per-backend validation.
* **TTGIR encoding ingestion** is intentionally not wired up
  (RFC 5.2). `layout.py` exists for the rare future case only;
  TileLang re-derives layouts per target.
* **Autograd through fused kernels** is left to manual integration
  (RFC section 8 question 5).

## How to add a new op mapping

1. Pick the right module under `op_emitters/`
   (`arith.py`/`memory.py`/`reduction.py`/`control.py`) by op
   category, or add a new `map_tt_<name>` in `op_mapping.py` for ops
   that don't fit those categories.
2. Register the emitter in the per-module `*_EMITTERS` dict so
   `OP_TABLE.update(...)` at the bottom of `op_mapping.py` picks it up.
3. Cite the RFC subsection (sections 5.1, 5.2, 5.4 are the typical
   ones) in the docstring.
4. Add a unit test under `tests/test_op_emitters_<category>.py` using
   the shared `FakeSSA` / `FakeMlirOp` fixtures from
   `tests/_fixtures.py` -- do **not** roll a private SSA stand-in.
5. If the new op is exercised by an existing numeric kernel, no
   harness change is needed; otherwise add a new
   `_test_harness/numeric_kernels/<name>.py` that exercises it
   end-to-end against a Triton reference.

## Setup

Install Triton (with the Apple backend) into `.venv313` so this POC can
capture TTIR from `@triton.jit` kernels:

```bash
bash scripts/setup_local_triton.sh
```

Idempotent. Editable-installs Triton from
`/Volumes/external/sources/triton-pr9701` (or falls back to the PyPI wheel
where available), pins the resolved version into
`.venv313/etc/pip/constraints.txt`, and verifies that `tvm_ffi`, `tvm`,
`tilelang`, and `triton` all import side-by-side.

## Building the C++ PtrAnalysis shim

The Python facade in `ptr_analysis.py` drives the vendored
`mlir::tts::PtrAnalysis` via a small pybind11 extension (`_triton_frontend_cxx`)
under `_cxx/`. It is **not** built as part of the regular Python install; the
walker silently falls back to the MVP scalar path (per-element BufferLoad /
BufferStore) when the shim is missing, with a one-shot
`RuntimeWarning("C++ PtrAnalysis shim unavailable; ...")`.

### macOS (Apple Silicon, brew LLVM/MLIR) -- stub mode

```bash
brew install llvm ninja pybind11
python -m poc.triton_frontend.build_cxx --build
```

`build_cxx.py` auto-detects `MLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir`
and `LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm`, runs cmake + ninja in
`poc/triton_frontend/_cxx/build/`, and prepends that directory to
`sys.path` so the next `import _triton_frontend_cxx` succeeds.

Without `TRITON_INSTALL_DIR` the build defaults to
`TRITON_FRONTEND_STUB_BUILD=ON`: the vendored `PtrAnalysis.cpp` is skipped
(it transitively needs upstream Triton MLIR headers), the C ABI compiles,
the extension loads, but `tl_pa_run_rewrite` returns `TL_PA_ERR_INTERNAL`
so the Python facade still falls back to MVP scalar.

### GB10 (Linux, sm_121, system LLVM/MLIR) -- full build

```bash
export TRITON_INSTALL_DIR=$HOME/triton/python/build/cmake.linux-aarch64-cpython-3.10/install
python -m poc.triton_frontend.build_cxx --build
```

`build_cxx.py` scans `/usr/lib/llvm-{20..14}` for system MLIR (GB10 dev
images ship 18) and passes `-DTRITON_INSTALL_DIR=...` through to cmake.
With both Triton dialect headers and `libTritonIR` available the full
PtrAnalysis is compiled and the multi-element tile load fast path lights
up automatically (no Python-side switch -- `dialects_available` flips to
`True`).

### Verification

```bash
python -m poc.triton_frontend.build_cxx --check    # exit 0 if importable
python -c "from poc.triton_frontend.ptr_analysis import shim_available, dialects_available; \
           print('shim:', shim_available(), 'dialects:', dialects_available())"
```

See `vendored/triton_shared/VENDORING_NOTES.md` for the full set of CMake
options (`TRITON_FRONTEND_USE_NLOHMANN_JSON`, `TRITON_FRONTEND_STUB_BUILD`)
and the file-level rationale for stub mode.

## How to run the test suite

Unit tests (op emitters, walker, pipeline, ptr analysis):

```bash
pytest poc/triton_frontend/tests/
```

End-to-end numeric harness (requires Triton + MLX):

```bash
python -m poc.triton_frontend._test_harness.numeric_smoke
```

Local Apple Silicon run with `cppmega_mlx` enabled uses the sibling
checkout at `/Users/dave/sources/cppmega.mlx` (same real path as
`/Volumes/external/sources/cppmega.mlx`) plus this repo's `.venv313`
Python:

```bash
export CPPMEGA_MLX=/Users/dave/sources/cppmega.mlx
export TL_ROOT=/private/tmp/tl_apache_tvm_swap
export TL_PY="$TL_ROOT/.venv313/bin/python"
export DYLD_FALLBACK_LIBRARY_PATH="$TL_ROOT/.venv313/lib/python3.13/site-packages/mlx/lib:/opt/homebrew/lib"
export PYTHONPATH="$TL_ROOT:$TL_ROOT/3rdparty/tvm/python:$CPPMEGA_MLX"

"$TL_PY" -m pytest poc/triton_frontend/tests/test_e2e_numeric_smoke.py -q
"$TL_PY" -m poc.triton_frontend._test_harness.numeric_smoke
```

This keeps `tvm_ffi` and TileLang on the current checkout while loading the
`cppmega_mlx` package from the sibling tree. A healthy dependency probe
reports `None` for `triton`, `tvm`, `tilelang`, `mlx`, and `cppmega_mlx`.

Reducer corpus (17 kernels):

```bash
python -m poc.triton_frontend._test_harness.run_corpus
```

Order is set by RFC section 5.5: `vector_add` -> `softmax` -> `matmul` ->
`layer_norm` -> `fa_v2` -> `fa_v3` -> `paged_attn`. The first three must
pass on Metal before CUDA work begins (RFC section 7 phase 1.5).

## Non-goals (explicit)

- **TTGIR encoding ingestion.** RFC section 5.2 -- we hook before layout
  assignment; TileLang re-derives layouts per target. Stubs in
  `layout.py` exist for the rare future case only.
- **Autograd through fused kernels.** RFC section 8 question 5 -- left
  to phase 2.3, manual.
- **CUTile ingestion.** RFC section 7 phase 5 -- deferred.

## Cross-references

- TileLang transform list (the source for `pipeline.PASS_ORDER`):
  `tilelang/transform/__init__.py`.
- TileLang phase pipeline (where this frontend slots in):
  `tilelang/engine/phase.py`.
- TileLang DSL primitives (`T.copy`, `T.gemm`, `T.reduce_*`, ...):
  `tilelang/language/__init__.py`.
- Sibling vendoring effort:
  `poc/triton_frontend/vendored/triton_shared/`.
