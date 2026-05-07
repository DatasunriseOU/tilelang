# Vendored: triton-shared PtrAnalysis

## Source

- Upstream repo: https://github.com/microsoft/triton-shared
- Upstream commit: `08684f92ad30696362dce1760a83be889639a3e4`
- Upstream commit date: `2025-12-05 22:58:49 +0000`
- Upstream commit subject: `Update README with maintenance notice (#367)`
- Local clone: `~/sources/triton-shared` (`/Users/dave/sources/triton-shared`)
- Copy date (UTC): 2026-05-07
- License: **MIT** (Microsoft Corporation). See `./LICENSE`.

> Note: the original task brief said "Apache-2.0", but upstream `LICENSE`
> is MIT. Compatible with Apache-2.0 hosting projects but credit and the
> license text must be retained in any redistribution. PtrAnalysis.h
> headers also carry per-file `// Licensed under the MIT license.` notices.

## Files copied (verbatim, untouched)

Headers (under `include/triton-shared/`):
- `Analysis/PtrAnalysis.h`              (legacy: drives memref reinterpret_cast lowering)
- `Analysis/MaskAnalysis.h`             (mask decomposition; PtrAnalysis.cpp transitively #includes it)
- `Analysis/OpFoldResultUtils.h`        (`addOFRs`, `mulOFRs`, `minOFRs` helpers used everywhere)
- `AnalysisStructured/PtrAnalysis.h`    (current/preferred: drives lowering to `tts.make_tptr`)

Sources (under `lib/`):
- `Analysis/PtrAnalysis.cpp`            (1375 lines)
- `Analysis/MaskAnalysis.cpp`           (862 lines)
- `Analysis/OpFoldResultUtils.cpp`      (405 lines)
- `Analysis/CMakeLists.txt`             (declares `TritonSharedAnalysis` target)
- `AnalysisStructured/PtrAnalysis.cpp`  (1959 lines)
- `AnalysisStructured/CMakeLists.txt`   (declares `TritonSharedAnalysisStructured` target)

Top-level:
- `LICENSE`                             (MIT, Microsoft Corp.)

Total: 11 files copied.

## Files NOT copied but transitively referenced

These are needed to actually compile the vendored sources. We must either
vendor them later, or stub/replace them when we wire PtrAnalysis into the
TileLang frontend.

### From upstream triton-shared (not yet vendored)

- `include/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h`
  and the entire TritonStructured dialect (TableGen `.td` + generated cpp).
  Used heavily in `AnalysisStructured/PtrAnalysis.cpp` for
  `tts::MakeTensorPtrOp`, `tts::MakeGatherScatterTensorPtrOp`,
  `tts::GetStructuredStateOp`, attribute `tts::ptrAnalysisAttr`.
  Path upstream: `include/triton-shared/Dialect/TritonStructured/...`,
  `lib/Dialect/TritonStructured/...`.

- `lib/Analysis/UseAnalysis.cpp` and `include/triton-shared/Analysis/UseAnalysis.h`
  Sibling pass; not used by PtrAnalysis directly but is in the same CMake target,
  so the unmodified `lib/Analysis/CMakeLists.txt` will fail to build until we
  either vendor it too or edit the CMakeLists. (We chose NOT to edit the
  CMakeLists, as per "do not modify the copied files".)

- The driver pass `lib/Conversion/TritonToStructured/TritonToStructuredPass.cpp`
  shows how PtrAnalysis is invoked. Useful as a recipe; not required to
  use the analysis as a library.

### From `triton/` (OpenAI Triton MLIR) — external to triton-shared

- `triton/Dialect/Triton/IR/Dialect.h` (and `Types.h`) — defines the
  `triton` dialect with ops `tt.make_range`, `tt.addptr`, `tt.splat`,
  `tt.broadcast`, `tt.expand_dims`, `tt.make_tensor_ptr`, `tt.advance`,
  `tt.load`, `tt.store`, `tt.bitcast`, `tt.int_to_ptr`, `tt.reshape`, etc.
  These come from the upstream Triton repo, not from triton-shared.

### From upstream MLIR / LLVM (assume already available via mlir-config)

- `mlir/Dialect/Arith/IR/Arith.h`
- `mlir/Dialect/MemRef/IR/MemRef.h`
- `mlir/Dialect/SCF/IR/SCF.h`
- `mlir/Dialect/Tensor/IR/Tensor.h`
- `mlir/IR/{Builders,BuiltinAttributes,BuiltinTypes,IRMapping,Location,OpDefinition,Value,Visitors}.h`
- `mlir/Support/{LLVM,LogicalResult}.h`
- `mlir/Transforms/DialectConversion.h`
- `llvm/ADT/{ArrayRef,SmallVector,STLExtras,TypeSwitch}.h`
- `llvm/Support/{Casting,Debug,LogicalResult}.h`

## Public API summary

There are TWO `PtrAnalysis` classes; they live in different namespaces.

### `mlir::triton::PtrAnalysis` (legacy — `Analysis/PtrAnalysis.h`)

- Static-method-driven, operates on a `ConversionPatternRewriter`.
- Maintains `llvm::SmallDenseMap<Value, PtrState>` of known pointers.
- Public entry points:
  - `rewriteAddptrOp(triton::AddPtrOp, rewriter, knownPtrs)`
  - `rewriteAdvanceOp(triton::AdvanceOp, rewriter, knownPtrs)`
  - `rewriteForOp(scf::ForOp, rewriter, levelToBlockArgIndex, level, knownPtrs)`
  - `rewriteYieldOp(scf::YieldOp, rewriter, levelToBlockArgIndex, level, knownPtrs)`
- `PtrState` produces `memref::ReinterpretCastOp`s.
- Supports modulo (wraparound), side-by-side and stacked layouts.

### `mlir::tts::PtrAnalysis` (preferred — `AnalysisStructured/PtrAnalysis.h`)

- Instance-based (`PtrAnalysis(bool enableMakeGatherScatterTensorPtr)`).
- Public state:
  - `llvm::SmallDenseMap<Value, PtrState> knownPtrs`
  - `IRMapping ptrMap`
- Public entry points:
  - `initializeMaybeStructuredArgs(Operation *)`
  - `rewriteAddptrOp(triton::AddPtrOp)`
  - `rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp)`
  - `rewriteAdvanceOp(triton::AdvanceOp)`
  - `rewriteYieldOp(scf::YieldOp, knownPtrsFor)`
  - `rewriteForOp(scf::ForOp)`
  - `rewriteLoadOp(triton::LoadOp, useUnsafeMask)`
  - `rewriteStoreOp(triton::StoreOp, useUnsafeMask)`
  - `rewriteOp(Operation *, useUnsafeMask)` (top-level convenience)
  - `rewriteGetStructuredStateOp(tts::GetStructuredStateOp)`
  - `getLoopInitArgPtrState`, `getLoopIterArgPtrState`, `getLoopResultPtrState`
- Output is `tts.make_tptr` / `tts.make_gather_scatter_tptr`.

`PtrState` (both versions) carries:
- `SmallVector<OpFoldResult> offsets, sizes, strides` (and `shape`, `order`,
  modulos in the structured variant)
- `Value source, scalar`
- Helpers `addState`, `mulState`, `getRank`, `isEmpty`, `hasModulo`, `isBlockPtr`.

## Test coverage upstream

PtrAnalysis is exercised end-to-end through three lit-test directories — these
are NOT vendored (60+ files each):
- `test/Conversion/TritonToStructured/` (68 .mlir files) — primary corpus
- `test/Conversion/StructuredToMemref/` — structured -> memref lowering
- `test/Conversion/TritonToLinalg/` — full pipeline
- Two files explicitly named `tensor_indices_loop_iterargs_not_used_ptranalysis_*`.

When we wire PtrAnalysis into the TileLang frontend we should consider
porting a starter subset of these `.mlir` files as golden-output regression
tests (probably the `addptr_*` family).

## Next steps (suggested)

1. Vendor `TritonStructured` dialect (TableGen + generated dialect) so
   `AnalysisStructured/PtrAnalysis.cpp` compiles.
2. Decide whether we link against an installed `triton/` MLIR build for
   the `tt.*` ops, or vendor a minimal subset of those op declarations.
3. Either:
   (a) build a small C++ shim that runs `tts::PtrAnalysis::rewriteOp`
       on a Triton MLIR module, then write a Python wrapper at
       `poc/triton_frontend/ptr_analysis.py` that drives it via
       `mlir-python-bindings` (preferred), or
   (b) port `PtrState` + the visitor functions to pure Python operating
       on the Triton MLIR textual form (slower to write, easier to
       integrate, no MLIR build dep).
4. Add `poc/triton_frontend/tests/` with a handful of upstream `.mlir`
   inputs translated to TileLang TIR expectations.

## Phase 2 vendoring (2026-05-07): TritonStructured dialect + UseAnalysis

Upstream commit: `08684f92ad30696362dce1760a83be889639a3e4` (microsoft/triton-shared).
Source clone: `/Users/dave/sources/triton-shared` (read-only).
License: MIT (Microsoft Corporation + Meta Platforms, Inc.). All copied
files retain their original per-file headers verbatim.

### Files copied

Headers (under `include/triton-shared/`):
- `Dialect/TritonStructured/IR/TritonStructuredDialect.td`   (365 lines, TableGen ops + dialect)
- `Dialect/TritonStructured/IR/TritonStructuredDialect.h`    (29 lines, declares `mlir::tts::TritonStructuredDialect`)
- `Dialect/TritonStructured/IR/CMakeLists.txt`               (upstream tablegen rules)
- `Analysis/UseAnalysis.h`                                   (119 lines, `mlir::triton::UseAnalysis` pass)

Sources (under `lib/`):
- `Dialect/TritonStructured/IR/TritonStructuredDialect.cpp`  (22 lines, dialect register)
- `Dialect/TritonStructured/IR/TritonStructuredOps.cpp`      (418 lines, op verifiers/builders)
- `Dialect/TritonStructured/IR/CMakeLists.txt`               (upstream `add_triton_library` for `TritonStructuredIR`)
- `Analysis/UseAnalysis.cpp`                                 (220 lines, pass impl)

Total: 8 files, ~1173 lines.

### Public symbols this brings in

From `mlir::tts` namespace (declared by TableGen output of `TritonStructuredDialect.td`):

- Dialect:
  - `mlir::tts::TritonStructuredDialect` (registered name `tts`)
- Ops:
  - `tts::MakeTensorPtrOp`               (`tts.make_tptr`)
  - `tts::MakeGatherScatterTensorPtrOp`  (`tts.make_gather_scatter_tptr`)
  - `tts::CreateTensorPtrOp`             (helper used during conversion)
  - `tts::GetStructuredStateOp`          (`tts.get_structured_state`)
  - `tts::LoadOp` / `tts::StoreOp`       (structured load/store with mask)
  - `tts::ScatterOp` / `tts::GatherOp`
- Attrs / utility:
  - `tts::ptrAnalysisAttr` (string attribute name used to mark argument
    PtrStates that survive across `func.call` boundaries)

From `mlir::triton` namespace (extra pass, not strictly required by
`PtrAnalysis` but listed in upstream `lib/Analysis/CMakeLists.txt`):

- `mlir::triton::UseAnalysis`  (dataflow analysis classifying `tt.*` value
  uses as `MetaUse` vs `DataUse`; PtrAnalysis consults the result via the
  query helper `getUseTag`).

### Why this set, exactly

`AnalysisStructured/PtrAnalysis.cpp` (already vendored in phase 1)
`#include`s `triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h`
and references the symbols listed above. Without the dialect headers and
the TableGen-generated `*.h.inc` / `*.cpp.inc` files, the analysis cannot
compile. Adding `UseAnalysis.{h,cpp}` lets us build the legacy
`TritonSharedAnalysis` target unmodified (its upstream `CMakeLists.txt`
lists `UseAnalysis.cpp`).

### Files still NOT vendored

- The `triton/` MLIR ops (`tt.addptr`, `tt.make_range`, `tt.load`, ...).
  These come from the upstream OpenAI Triton MLIR build and are reached
  via `TRITON_INSTALL_DIR` (see top-level `CMakeLists.txt` in this
  vendored tree).
- Conversion pass drivers (`lib/Conversion/TritonToStructured/...`). We
  invoke `tts::PtrAnalysis::rewriteOp` directly from our own shim, so the
  driver pass is not needed.
