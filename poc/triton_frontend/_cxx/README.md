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
from poc.triton_frontend.ptr_analysis import (
    PtrAnalysis,
    PtrState,
    shim_available,
    dialects_available,
)

assert shim_available()        # the .so loaded
assert dialects_available()    # built with -DTRITON_INSTALL_DIR

pa = PtrAnalysis(mlir_text)
rewritten = pa.rewrite()       # returns the rewritten module text (cached)
states = pa.extract_states()   # list[PtrState] (shares the same parse)
```

The first call to either ``rewrite()`` or ``extract_states()`` runs the C++
analysis once; subsequent calls return cached results. The pybind layer also
exposes ``run_ptr_analysis_with_states`` for callers that want both outputs
in a single round-trip.

## Optional: nlohmann::json encoder

The `tl_pa_extract_states_json` entry point ships with a hand-rolled
RFC-8259 escaper as the default. For builds that want to compose its
output with downstream nlohmann::json consumers (or just prefer a single
canonical encoder across the codebase), pass `-DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON`
to CMake. The two encoders MUST emit byte-identical output for the current
schema; the regression test in
`poc/triton_frontend/tests/test_ptr_analysis.py::test_manual_escaper_matches_json_dumps`
guards that contract.

Vendor the single-include header first:

```bash
curl -L -o poc/triton_frontend/_cxx/third_party/nlohmann/json.hpp \
     https://github.com/nlohmann/json/releases/latest/download/json.hpp
```

The CMake option fails fast with the same `curl` line if you forget. Pin
the upstream tag you fetched in `third_party/nlohmann/VERSION` so re-fetch
is reproducible.

You can introspect which encoder the compiled `.so` uses from Python via
`tl_pa_uses_nlohmann_json` (exposed as a module-level integer if/when the
pybind layer surfaces it; today it's only on the C ABI).

## License

The shim derives from microsoft/triton-shared (MIT, copyright Microsoft +
Meta Platforms). All files preserve the original copyright header.
nlohmann/json is MIT-licensed; if vendored, it goes under
`third_party/nlohmann/` with its own `LICENSE` copy.
