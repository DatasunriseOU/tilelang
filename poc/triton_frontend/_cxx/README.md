# `_triton_frontend_cxx` --- C++ shim for triton-shared `PtrAnalysis`

This is a thin pybind11 module that drives
`mlir::tts::PtrAnalysis::rewriteOp` from Python by parsing MLIR text into a
`ModuleOp`, running the analysis, and printing the rewritten module back
out.

## Why a C++ shim and not pure mlir-python-bindings?

`mlir::tts::PtrAnalysis` is a stateful C++ class: it carries a
`DenseMap<Value, PtrState>`, an `IRMapping`, mutates the IR with a builder,
and walks `scf.for` regions across iterations. `mlir-python-bindings`
exposes only the upstream MLIR core (Operation/Block/Value/PassManager) plus
in-tree dialect ops; it has no surface for triton-shared's custom analyses.
We could re-implement `PtrAnalysis` in Python, but that defeats the point of
vendoring (RFC section 3 explicitly calls for re-using the C++ analysis
verbatim). Hence: a tiny C++ shim that exposes 5 entry points.

## Build

The shim links against an MLIR install. The full Triton/TritonStructured
dependency lands with sibling integration #5; until then the shim builds
in *parse-and-print* mode and `run_rewrite` returns `TL_PA_ERR_INTERNAL`.

```bash
# Replace paths with your local install prefixes.
export MLIR_DIR=/opt/llvm/lib/cmake/mlir
export LLVM_DIR=/opt/llvm/lib/cmake/llvm
export TRITON_INSTALL_DIR=/opt/triton          # optional, see CMakeLists.txt

cmake -S poc/triton_frontend/_cxx -B build/triton_frontend_cxx \
  -DMLIR_DIR=${MLIR_DIR} \
  -DLLVM_DIR=${LLVM_DIR} \
  -DTRITON_INSTALL_DIR=${TRITON_INSTALL_DIR}

cmake --build build/triton_frontend_cxx -j

# Result:
#   build/triton_frontend_cxx/_triton_frontend_cxx*.so
```

Add the build directory to `PYTHONPATH` (or copy/symlink the `.so` into
`poc/triton_frontend/`) so that `ptr_analysis.py`'s lazy `importlib` lookup
succeeds.

## API surface

```python
from poc.triton_frontend.ptr_analysis import PtrAnalysis, PtrState
pa = PtrAnalysis(mlir_text)
rewritten = pa.rewrite()              # returns the rewritten module text
states = pa.extract_states()          # list[PtrState]
```

## License

The shim derives from microsoft/triton-shared (MIT, copyright Microsoft +
Meta Platforms). All files preserve the original copyright header.
