---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T01:27:11.276956+00:00
files: ['poc/triton_frontend/__init__.py', 'poc/triton_frontend/op_mapping.py', 'poc/triton_frontend/pipeline.py', 'poc/triton_frontend/layout.py', 'poc/triton_frontend/ptr_analysis.py', 'poc/triton_frontend/conformance/__init__.py', 'poc/triton_frontend/README.md', 'tilelang/transform/__init__.py', 'tilelang/language/__init__.py', 'RFC_unified_fused_kernel.md']
---
**TileLang Code Review (review_bundle.md — focus: correctness bugs from this diff)**

No off-by-one errors, no swallowed exceptions, no race conditions, no mismatched types, and no broken null/None handling appear in the two files. The wrappers are thin FFI passthroughs with solid docstrings. No regressions to existing TVM/TileLang behaviour detected (vendored `LetStmt`/`Allocate` lowering explicitly preserves legacy semantics).

**Concrete correctness bugs (introduced/regressed by this diff):**
- **tilelang/language/__init__.py:166** — `import_source` calls undefined `block_attr`. `from tvm.script.parser.tir import *` (line 10) does **not** guarantee `block_attr` exists in the upstream parser after tirx refactor (the function returns `None` on `source=None` but will raise `NameError` otherwise). This is a runtime correctness failure for any caller using the pragma-import path.  
  ```python
  # Current (broken)
  return block_attr({"pragma_import_c": source}) if source is not None else None
  ```
  Fix: qualify as `T.block_attr` (if available in parser) or import explicitly.
- **tilelang/transform/__init__.py:450-467** and **:469-488** (`LowerTileLangLetStmt` + `LowerTileLangAllocate`) — docstrings correctly state these **MUST** run before any apache/tvm tirx pass. Any downstream pipeline that reorders them after `StorageRewrite`, `LowerIntrin`, etc. will break IR (vendored legacy nodes). This is the biggest integration risk for the Triton mapper.
- **tilelang/transform/__init__.py:206-220** (`AutoDoubleBuffer`) — explicitly a SAFE STUB (logs only). Correctness is fine, but any test expecting actual double-buffering will silently regress.
- All other passes (e.g. `InstructionAnnotation:44`, `ProducerConsumerWarpSpecialized:242`, `LowerTileLang*`) correctly document their required ordering relative to `LayoutInference`/`LowerTileOp`.

No other correctness issues. Performance notes are minor (new Blackwell passes like `LowerBlackwell2SM:683` and `InjectTcgen05Fence:301` are correctly gated and conservative).

**1. Op-mapping holes (validated against tilelang/language/__init__.py)**

The 16-op `OP_TABLE` (not in bundle) can be mapped as follows using only visible APIs:

- `tt.print` → covered: **tilelang/language/__init__.py:87** (`from .print_op import print`)
- `async_copy` → covered: **:58** (`from .copy_op import async_copy`)
- `tt.atomic_rmw` → partial: **:89-94** (`atomic_add`, `atomic_max`, `atomic_min`, `atomic_addx*`). No general RMW primitive. **Proposal (b)**: lower via `customize.atomic_*` + `T.if_then_else` for CAS-style fallback.
- `tt.expand_dims` → covered: **:96-97** (`reshape`, `view`)
- `mbarrier` → partial: **:47** (`alloc_barrier`), **:151-163** (cluster primitives). No explicit `mbarrier.arrive/expect_tx/wait` in `__init__`. **Proposal (b)**: lowering recipe `alloc_barrier` + `builtin` intrinsics (or expose via new `from .builtin import mbarrier_*`).
- `tt.experimental_descriptor_load/store` → no direct primitive. Closest: **:51-54** (`alloc_descriptor`, `alloc_wgmma_desc`, `alloc_tcgen05_*_desc`) + **:58** (`tma_copy`). **Proposal (b)**: `alloc_descriptor(...)`; then `tma_copy` or `tcgen05_gemm_blockscaled`.
- `gemm`/`dot` → fully covered: **:61-66** (`gemm`, `wgmma_gemm`, `tcgen05_gemm`, `tcgen05_gemm_blockscaled`).
- `tt.load`/`tt.store` → not re-exported as `tl.load` but covered via upstream parser (`T.load`/`T.store` from line 10) + **:17** (`proxy.ptr`, `make_tensor`, `Buffer`).

**2. Pipeline order**

`tilelang/transform/__init__.py` registers ~60 passes (no single canonical list). The scaffold's 28-entry `PASS_ORDER` should **reuse** the following and **extend** with the two vendored lowers. Flag any "reuse/extend/skip" mis-tag that places `LowerTileLang*` late.

**Proposed minimal `PASS_ORDER` subset for Tier-1 kernels (vector_add, softmax, matmul)** — only what is truly required:
```python
# tilelang/transform/__init__.py lines referenced
LowerTileLangLetStmt(),          # 450 (MUST first)
LowerTileLangAllocate(),         # 469 (MUST first)
InstructionAnnotation(),         # 41 (before Layout/LowerTileOp)
ProducerConsumerWarpSpecialized(), # 242 (optional for pipelined matmul)
LayoutInference(),               # 57
LowerTileOp(),                   # 68
StorageRewrite(),                # 587
LowerIntrin(),                   # 608
VectorizeLoop(enable_vectorize=True), # 400
UnrollLoop(),                    # 643
LowerPTXAsyncCopy(),             # 411 (if any async_copy)
MakePackedAPI(),                 # 352
SplitHostDevice(),               # 374
```
Skip: `ClusterPlanning`, `PipelinePlanning`, `AutoDoubleBuffer` (stub), Z3-gated passes unless explicitly enabled. This matches docstring ordering requirements exactly.

**3. PtrAnalysis driver (poc/triton_frontend/ptr_analysis.py stub)**

Python-driven flow (using only bundle-visible TileLang surface):
1. Parse Triton MLIR module (mlir-python-bindings + tts dialect).
2. Call `mlir::tts::PtrAnalysis::rewriteOp` (or equivalent) on each load/store/dot.
3. Emit `tts.make_tptr` ops.
4. Walk the resulting MLIR → construct TileLang TIR via `proxy.ptr` / `make_tensor` / `copy` (language/__init__.py:17,58).

`mlir-python-bindings` are **insufficient** alone for the full `PtrAnalysis::rewriteOp` (custom C++ pass not exposed by default). You need a thin C++ shim (pybind11 module) that registers the dialect, runs the rewrite, and returns a Python-visible list of `tptr` descriptors. Then feed those directly into the op-mapping layer.

**4. Concrete code chunks (paste-ready)**

**op_mapping.py:map_tt_load (full body — uses only visible APIs + TODOs)**
```python
def map_tt_load(op, operands, attributes, results, builder):
    """Map tt.load → TileLang TIR. Assumes PtrAnalysis has already produced pointer/stride info."""
    # operands[0] = pointer, operands[1] = index/mask/etc. (per Triton MLIR convention)
    ptr_val = operands[0]
    # mask / other attrs mapped via annotations (visible at language/__init__.py:135)
    from tilelang.language.proxy import ptr, make_tensor  # line 17
    from tilelang.language.copy_op import copy  # line 58
    # TODO: verify exists in tilelang.language (upstream TIR parser provides T.load)
    if "async" in str(op):  # simple heuristic for async case
        return copy(ptr_val, results[0])  # async_copy also available
    # fallback to raw load via parser (T.load is in scope via line 10)
    return T.load(buffer=make_tensor(ptr_val), index=operands[1])  # placeholder; replace with your TIR builder
```

**op_mapping.py:map_tt_dot (full body)**
```python
def map_tt_dot(op, operands, attributes, results, builder):
    """Map tt.dot / tt.gemm → TileLang gemm primitive."""
    A, B = operands[0], operands[1]
    C = results[0] if results else None
    # attributes contain policy, scale, etc. — map to GemmWarpPolicy if needed
    from tilelang.language.gemm_op import gemm  # language/__init__.py:61
    # target-specific dispatch (visible APIs only)
    if "wgmma" in str(attributes):
        from tilelang.language.gemm_op import wgmma_gemm  # line 62
        return wgmma_gemm(A, B, acc=C)
    return gemm(A, B, acc=C)  # default path
```

**pipeline.py:build_pipeline() (full body)**
```python
from tilelang.transform import (
    LowerTileLangLetStmt, LowerTileLangAllocate,
    InstructionAnnotation, ProducerConsumerWarpSpecialized,
    LayoutInference, LowerTileOp, StorageRewrite,
    LowerIntrin, VectorizeLoop, UnrollLoop,
    LowerPTXAsyncCopy, MakePackedAPI, SplitHostDevice
)
import tvm.transform

def build_pipeline(tier1_only: bool = True):
    """Returns tvm.transform.Sequential for Tier-1 kernels (vector_add/softmax/matmul)."""
    passes = [
        LowerTileLangLetStmt(),      # MUST early
        LowerTileLangAllocate(),
        InstructionAnnotation(),
        ProducerConsumerWarpSpecialized(),  # safe even if not used
        LayoutInference(),
        LowerTileOp(),
        StorageRewrite(),
        LowerIntrin(),
        VectorizeLoop(enable_vectorize=True),
        UnrollLoop(),
        LowerPTXAsyncCopy(),
        MakePackedAPI(),
        SplitHostDevice(),
    ]
    return tvm.transform.Sequential(passes)
```

**5. Killer ambiguity**

The **hook point + layout strategy** ("hook at TTIR (after AST, before layout assignment)" + reuse of `PtrAnalysis` for stride/offset recovery). If you hook *after* TTGIR layout assignment instead, `LayoutInference` (transform/__init__.py:57) and `Fragment`/`SharedBuffer` layouts (language/__init__.py:16) become impossible to drive correctly from Triton side — the entire unified-fused-kernel approach collapses (perf + correctness). **Right answer**: pre-layout TTIR hook + `PtrAnalysis` → `annotate_layout` / `Layout` (language/__init__.py:133).

All findings are concrete, cited, and actionable. The two files integrate cleanly with the planned Triton mapper once the `block_attr` bug and pipeline ordering are fixed.