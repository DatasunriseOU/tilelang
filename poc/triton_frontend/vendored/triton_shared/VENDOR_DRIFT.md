# Vendored triton-shared: intentional drift register

This file documents every vendored source under
`poc/triton_frontend/vendored/triton_shared/` whose on-disk content
intentionally diverges from the upstream snapshot referenced by
`VENDORING_NOTES.md`.

## Pinned upstream baseline

- Repo: https://github.com/microsoft/triton-shared
- Commit: `08684f92ad30696362dce1760a83be889639a3e4` (2025-12-05)
- Local clone for diff: `/Users/dave/sources/triton-shared`

## Drift policy

The committed `.vendor-manifest.sha256` always reflects the **forward-ported
on-disk state**, not the unmodified upstream state. Two reasons:

1. The N2 review (Wave 3 forward-port) intentionally introduces local
   edits to keep the snapshot compiling against current MLIR / Triton
   APIs. Reverting them on every refresh would break the build.
2. The drift detector's job is to catch *unintended* edits made after
   the manifest was last refreshed (e.g. a developer hand-tweaking a
   vendored file without updating this register). For that to work the
   manifest must encode the agreed-upon committed state, with each
   intentional drift documented below.

When refreshing the manifest after a deliberate forward-port edit:

```bash
python -m poc.triton_frontend.vendored.triton_shared.check_vendor_drift --refresh
```

Then update the table below — adding (or removing) the file, the
short reason, and the intended-upstream-rev for the eventual upstream
PR.

## Drift register (Wave 3)

| File | Drift category | Reason | Intended upstream rev |
|------|----------------|--------|------------------------|
| `lib/Analysis/PtrAnalysis.cpp` | Forward-port | Adapt legacy block-pointer rewrite to current MLIR `OpFoldResult` / `ConversionPatternRewriter` API surface. Gated behind `TRITON_SHARED_ALLOW_LEGACY_PTRANALYSIS`. | Upstream eventual deprecation of legacy variant; track in microsoft/triton-shared#TBD. |
| `lib/AnalysisStructured/PtrAnalysis.cpp` | Forward-port | Match current MLIR API drift (builder signatures, `IRMapping`, `OpFoldResult` helpers) and align with our pinned `tt.*` op set. This is the supported variant. | Upstream once microsoft/triton-shared rebases onto the same MLIR tip. |
| `lib/Analysis/UseAnalysis.cpp` | Override | Implements `visitNonControlFlowArguments` so the dataflow framework on current LLVM accepts `UseAnalysis` as a concrete `SparseForwardDataFlowAnalysis`. Upstream snapshot lacks this override. | Will be sent upstream as a defensive override patch. |
| `lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp` | TableGen drift fix | Adjust verifier/builder signatures to match the regenerated `*.cpp.inc` produced from our locally-pinned LLVM/MLIR TableGen. Pure boilerplate plumbing — no semantic change to op contracts. | None — pure regeneration artifact. |
| `include/triton-shared/Analysis/PtrAnalysis.h` | Forward-port (header) | Header signature changes paired with `lib/Analysis/PtrAnalysis.cpp`. | Pairs with the cpp entry above. |
| `include/triton-shared/Analysis/UseAnalysis.h` | Override (header) | Declares the `visitNonControlFlowArguments` override added in the cpp. | Pairs with the cpp entry above. |
| `include/triton-shared/AnalysisStructured/PtrAnalysis.h` | Forward-port (header) | Header signature changes paired with `lib/AnalysisStructured/PtrAnalysis.cpp`. | Pairs with the cpp entry above. |
| `include/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.td` | TableGen drift fix | Tweak `assemblyFormat` / op trait references to compile cleanly against our pinned LLVM/MLIR TableGen. Pure surface; op names + arg types unchanged. | None — paired with regen. |
| `lib/Analysis/MaskAnalysis.cpp` | Forward-port | Same MLIR API drift fixes as PtrAnalysis.cpp (e.g. `OpFoldResult` arithmetic helper relocation). | Pairs with `PtrAnalysis.cpp`. |
| `lib/Analysis/OpFoldResultUtils.cpp` | Forward-port | API drift in `mlir::affine::makeComposedFoldedAffineApply` plus tightened `OpFoldResult` type checks. | Pairs with `PtrAnalysis.cpp`. |

## Verification

After any refresh, the following must pass:

```bash
python -m poc.triton_frontend.vendored.triton_shared.check_vendor_drift
pytest poc/triton_frontend/tests/test_vendor_drift.py -v
```

Both invocations are CI-gated; a failure means either (a) someone
edited a vendored file without refreshing the manifest, or (b) the
register above lost sync with `.vendor-manifest.sha256`.
