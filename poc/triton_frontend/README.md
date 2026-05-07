# Triton -> TileLang TIR frontend (POC scaffold)

**Status: SCAFFOLD ONLY.** Every public function raises
`NotImplementedError`. This directory is **kept separate from the
production `tilelang/` tree** so the POC can churn independently. Do not
import from this package in production code yet.

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
                              + ptr_analysis (vendored)      LowerTileOp, ...)
                                       |
                                       v
                         +-----------------------------+
                         | layout.py (NOT used in MVP) |  <- TTGIR encodings
                         | RFC 5.2: re-derive layouts  |     deliberately not
                         | per target instead          |     ingested
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
                          torch.library.custom_op
```

## File layout

| File                          | Purpose                                                                   | RFC ref       |
|-------------------------------|---------------------------------------------------------------------------|---------------|
| `__init__.py`                 | Public API: `from_triton_kernel`, `from_ttir` (stubs).                    | section 5     |
| `ptr_analysis.py`             | Wrapper over vendored `microsoft/triton-shared` `PtrAnalysis`.            | sections 3, 7 |
| `op_mapping.py`               | Dispatch table: TTIR op name -> TileLang emitter (stubs).                 | section 5.1   |
| `layout.py`                   | TTGIR encoding translators (placeholder; **not used** in MVP path).       | section 5.2   |
| `pipeline.py`                 | Ordered TileLang TIR transform passes (reuse / extend / skip).            | section 3     |
| `conformance/__init__.py`     | Reference kernels for end-to-end tests.                                   | section 5.5   |
| `vendored/triton_shared/`     | Populated by sibling agent -- `microsoft/triton-shared` (Apache-2.0).     | section 3     |

## Stubbed-vs-done status

| Component                                   | Status   |
|---------------------------------------------|----------|
| Public API surface (`__init__.py`)          | stubbed  |
| PtrAnalysis Python wrapper                  | stubbed  |
| TTIR op dispatch table (16 ops)             | stubbed  |
| TTGIR layout translators (future-use)       | stubbed  |
| Pipeline driver / pass ordering             | listed, stubbed |
| Conformance kernels (7 from RFC 5.5)        | placeholders |
| Vendored `triton-shared` sources            | populated by sibling agent |
| End-to-end test running                     | not yet  |

## How to add a new op mapping

1. Add the emitter in `op_mapping.py`:
   ```python
   def map_tt_<name>(op, ctx) -> Any:
       """RFC section 5.1: tt.<name>."""
       raise NotImplementedError("RFC section 5.1: tt.<name>")
   ```
2. Register it in the `OP_TABLE` dict at the bottom of the same file.
3. Add a one-line conformance kernel in `conformance/__init__.py` that
   exercises the new op. Use the smallest kernel that fails without it.
4. Cite the RFC subsection (sections 5.1, 5.2, 5.4 are the typical ones)
   in the docstring.

## Building the C++ PtrAnalysis shim

The Python facade in `ptr_analysis.py` drives the vendored
`mlir::tts::PtrAnalysis` via a small pybind11 extension (`_triton_frontend_cxx`)
under `_cxx/`. It is **not** built as part of the regular Python install; the
walker silently falls back to the MVP scalar path (per-element BufferLoad /
BufferStore) when the shim is missing, with a one-shot
`RuntimeWarning("C++ PtrAnalysis shim unavailable; ...")`.

### macOS (Apple Silicon, brew LLVM/MLIR) — stub mode

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

### GB10 (Linux, sm_121, system LLVM/MLIR) — full build

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

## How to run conformance

Once stubs are filled in, the suite runs under pytest::

    pytest poc/triton_frontend/conformance/

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
