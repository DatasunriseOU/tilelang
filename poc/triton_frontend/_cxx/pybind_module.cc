//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
// This file is part of the TileLang Triton-frontend integration. It re-uses
// the vendored microsoft/triton-shared PtrAnalysis (commit 08684f9, 2025-12-05)
// under the original MIT license.
//
//===----------------------------------------------------------------------===//
//
// pybind_module.cc --- pybind11 wrapper around the C shim. The Python module
// name is `_triton_frontend_cxx`; it is loaded lazily by
// poc/triton_frontend/ptr_analysis.py via importlib.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "ptr_analysis_shim.h"
#include "RegisterTritonStructured.h"
#include "mlir/IR/MLIRContext.h"

namespace py = pybind11;

namespace {

// RAII wrapper so Python exceptions don't leak the C handles.
class Context {
 public:
  Context() : raw_(tl_pa_context_create()) {}
  ~Context() { tl_pa_context_destroy(raw_); }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  TLPtrAnalysisContext* get() const { return raw_; }

 private:
  TLPtrAnalysisContext* raw_;
};

class Module {
 public:
  Module(Context& ctx, const std::string& text) : ctx_(&ctx) {
    TLPtrAnalysisStatus st = TL_PA_OK;
    raw_ = tl_pa_module_parse(ctx.get(), text.data(), text.size(), &st);
    if (!raw_ || st != TL_PA_OK) {
      throw std::runtime_error(std::string("tl_pa_module_parse failed: ") +
                               tl_pa_take_last_error(ctx.get()));
    }
  }
  ~Module() { tl_pa_module_destroy(raw_); }
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  TLPtrAnalysisModule* get() const { return raw_; }
  Context* ctx() const { return ctx_; }

  // Centralised wrapper around tl_pa_run_rewrite so both the pybind class
  // method and the top-level convenience helpers share one implementation
  // (and the same error-translation path).
  void run_rewrite(bool enable_gs, bool unsafe_mask) {
    auto st = tl_pa_run_rewrite(get(), enable_gs ? 1 : 0, unsafe_mask ? 1 : 0);
    if (st != TL_PA_OK) {
      throw std::runtime_error(std::string("tl_pa_run_rewrite failed: ") +
                               tl_pa_take_last_error(ctx_->get()));
    }
  }

 private:
  Context* ctx_;
  TLPtrAnalysisModule* raw_ = nullptr;
};

std::string moduleToString(Module& m) {
  char* s = tl_pa_module_to_string(m.get());
  if (!s) return {};
  std::string out(s);
  tl_pa_string_free(s);
  return out;
}

std::string moduleToGeneric(Module& m) {
  char* s = tl_pa_module_to_generic(m.get());
  if (!s) return {};
  std::string out(s);
  tl_pa_string_free(s);
  return out;
}

}  // namespace

// Probe whether the Triton + TritonStructured dialect headers were available
// at compile time. When false, ``run_rewrite`` will return TL_PA_ERR_INTERNAL
// (see ptr_analysis_shim.cc); tests use this to differentiate "shim built but
// stub mode" from "shim built with full dialect support".
#define TL_PA_DIALECTS_AVAILABLE 1

PYBIND11_MODULE(_triton_frontend_cxx, m) {
  m.doc() = "TileLang Triton-frontend C++ shim around triton-shared PtrAnalysis";

  m.attr("dialects_available") = bool(TL_PA_DIALECTS_AVAILABLE);

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("last_error",
           [](Context& self) -> std::string {
             return tl_pa_take_last_error(self.get());
           });

  py::class_<Module>(m, "Module")
      .def(py::init<Context&, const std::string&>(),
           py::keep_alive<1, 2>(),
           py::arg("ctx"), py::arg("mlir_text"))
      .def("to_string", &moduleToString)
      .def("to_generic", &moduleToGeneric,
           "Print the module in generic op form (\"dialect.op\"(...) : ...) "
           "so external parsers without Triton dialect registered can "
           "round-trip the text — e.g. jaxlib's stripped mlir.ir bindings.")
      .def("run_rewrite", &Module::run_rewrite,
           py::arg("enable_make_gather_scatter_tensor_ptr") = false,
           py::arg("use_unsafe_mask") = false)
      .def("extract_states_json",
           [](Module& self) -> std::string {
             const char* s = tl_pa_extract_states_json(self.get());
             return s ? std::string(s) : std::string("[]");
           });

  // Run the StructuredToMemref conversion pass on a parsed TTIR text.
  // Parses, applies StructuredToMemref, prints, returns the rewritten module.
  // Raises RuntimeError on parse/verify/conversion failure.
  m.def("run_structured_to_memref",
        [](const std::string& mlir_text) -> std::string {
          Context ctx;
          Module mod(ctx, mlir_text);
          auto st = tl_pa_run_structured_to_memref(mod.get());
          if (st != TL_PA_OK) {
            throw std::runtime_error(
                std::string("tl_pa_run_structured_to_memref failed: ") +
                tl_pa_take_last_error(ctx.get()));
          }
          return moduleToString(mod);
        },
        py::arg("mlir_text"));

  m.def("walk_op_names_json",
        [](const std::string& mlir_text) -> std::string {
          Context ctx;
          Module mod(ctx, mlir_text);
          char* names = tl_pa_module_op_names_json(mod.get());
          std::string out = names ? std::string(names) : std::string("[]");
          tl_pa_string_free(names);
          return out;
        },
        py::arg("mlir_text"),
        "Parse MLIR text with the shim context and return walked operation "
        "names as a JSON array.");

  // Convenience top-level: parse, rewrite, return printed text in one step.
  m.def("run_ptr_analysis",
        [](const std::string& mlir_text,
           bool enable_gs,
           bool unsafe_mask) -> std::string {
          Context ctx;
          Module mod(ctx, mlir_text);
          mod.run_rewrite(enable_gs, unsafe_mask);
          // Reach back into mod via the helper.
          return moduleToString(mod);
        },
        py::arg("mlir_text"),
        py::arg("enable_make_gather_scatter_tensor_ptr") = false,
        py::arg("use_unsafe_mask") = false);

  // Honour the same flags as ``run_ptr_analysis`` so callers can switch
  // between the two without surprises. The ``PtrAnalysis`` class default
  // (False/False) matches upstream TritonToStructuredPass.
  m.def("extract_ptr_states",
        [](const std::string& mlir_text,
           bool enable_gs,
           bool unsafe_mask) -> std::string {
          Context ctx;
          Module mod(ctx, mlir_text);
          mod.run_rewrite(enable_gs, unsafe_mask);
          return std::string(tl_pa_extract_states_json(mod.get()));
        },
        py::arg("mlir_text"),
        py::arg("enable_make_gather_scatter_tensor_ptr") = false,
        py::arg("use_unsafe_mask") = false);

  // Single-pass helper: parse the module once, run rewrite, return both the
  // rewritten text and the states-JSON. Lets the Python facade populate both
  // caches with a single MLIRContext + parse.
  m.def("run_ptr_analysis_with_states",
        [](const std::string& mlir_text,
           bool enable_gs,
           bool unsafe_mask) -> py::tuple {
          Context ctx;
          Module mod(ctx, mlir_text);
          mod.run_rewrite(enable_gs, unsafe_mask);
          std::string rewritten = moduleToString(mod);
          std::string states(tl_pa_extract_states_json(mod.get()));
          return py::make_tuple(std::move(rewritten), std::move(states));
        },
        py::arg("mlir_text"),
        py::arg("enable_make_gather_scatter_tensor_ptr") = false,
        py::arg("use_unsafe_mask") = false);

  // Same as ``run_ptr_analysis_with_states`` but returns the rewritten module
  // in generic op form. This is the safe external-parser seam: the walker
  // consumes this exact text, and the PtrState JSON is extracted from the
  // same rewritten Module lifetime instead of a second custom->generic
  // re-print that can assign different SSA names.
  m.def("run_ptr_analysis_with_states_generic",
        [](const std::string& mlir_text,
           bool enable_gs,
           bool unsafe_mask) -> py::tuple {
          Context ctx;
          Module mod(ctx, mlir_text);
          mod.run_rewrite(enable_gs, unsafe_mask);
          std::string rewritten = moduleToGeneric(mod);
          std::string states(tl_pa_extract_states_json(mod.get()));
          return py::make_tuple(std::move(rewritten), std::move(states));
        },
        py::arg("mlir_text"),
        py::arg("enable_make_gather_scatter_tensor_ptr") = false,
        py::arg("use_unsafe_mask") = false);

  // Dialect registration via C++ shim
  m.def("register_dialects",
        [](py::object ctx_obj) {
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
        py::arg("ctx"));
}
