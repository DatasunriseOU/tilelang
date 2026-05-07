// Vendored helper for triton-shared integration with the unified fused-kernel compiler.
// Copyright (c) 2026 Project Contributors.
// Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
// Licensed under the MIT License.
//
// register_triton_structured_pybind.cc --- standalone pybind11 module that
// exposes a single entry point, ``register_dialects(mlir.ir.Context)``,
// which forwards into ``mlir::triton_shared_vendored::registerTritonStructured``
// (defined under ``vendored/triton_shared/RegisterTritonStructured.{h,cc}``).
//
// This shim is intentionally kept separate from ``pybind_module.cc`` (which
// produces ``_triton_frontend_cxx`` and wraps the C-shim around PtrAnalysis)
// so that downstream consumers can link the much smaller
// ``TritonSharedRegister`` static library on its own when they only need
// dialect registration -- e.g. from the lit-style test in
// ``vendored/triton_shared/verify_dialect_loads.py``.
//
// Build target (added by the vendored CMakeLists.txt or by the parent
// _cxx/CMakeLists.txt):
//     pybind11_add_module(_register_triton_structured
//                         register_triton_structured_pybind.cc)
//     target_link_libraries(_register_triton_structured PRIVATE
//                           TritonSharedRegister)
// The resulting extension is installed as
//     poc/triton_frontend/_cxx/register_triton_structured.<ext-suffix>.so
//

#include <pybind11/pybind11.h>

#include "mlir/IR/MLIRContext.h"

// Note: RegisterTritonStructured.h lives at the root of vendored/triton_shared/
// (alongside RegisterTritonStructured.cc), not under include/triton-shared/.
// The CMakeLists.txt for the vendored target adds vendored/triton_shared as a
// direct include directory, so we include it without a path prefix.
#include "RegisterTritonStructured.h"

namespace py = pybind11;

PYBIND11_MODULE(register_triton_structured, m) {
  m.doc() =
      "Standalone shim that registers the vendored `tts` (TritonStructured) "
      "dialect plus its required upstream MLIR/Triton dependencies into a "
      "user-supplied mlir.ir.Context.";

  m.def(
      "register_dialects",
      [](py::object ctx_obj) {
        // The MLIR Python bindings expose ``mlir.ir.Context._CAPIPtr`` as a
        // PyCapsule wrapping the ``MlirContext`` (which is just an
        // ``MLIRContext *``). pybind11's auto-cast cannot round-trip this
        // capsule, so we accept ``py::object`` and pull the pointer out
        // explicitly. Callers that already have an
        // ``mlir::MLIRContext &`` (e.g. the C++ test harness) can use the
        // overload below.
        py::object capi = ctx_obj.attr("_CAPIPtr");
        auto* mlir_ctx = static_cast<mlir::MLIRContext*>(
            PyCapsule_GetPointer(capi.ptr(), "mlir.ir.Context._CAPIPtr"));
        if (!mlir_ctx) {
          throw py::value_error(
              "register_dialects: ctx._CAPIPtr did not yield a valid "
              "MLIRContext pointer");
        }
        mlir::triton_shared_vendored::registerTritonStructured(*mlir_ctx);
      },
      py::arg("ctx"),
      "Register tts (TritonStructured) + tt (Triton) + the supporting "
      "upstream MLIR dialects (arith, func, memref, scf, tensor) into the "
      "given mlir.ir.Context.");

  // Direct C++ overload for in-process callers that already hold an
  // ``mlir::MLIRContext`` (no pybind PyCapsule round-trip).
  m.def(
      "register_dialects_cxx",
      [](mlir::MLIRContext& ctx) {
        mlir::triton_shared_vendored::registerTritonStructured(ctx);
      },
      py::arg("context"),
      "C++-typed overload of register_dialects(); takes an "
      "mlir::MLIRContext& directly.");
}
