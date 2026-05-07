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
