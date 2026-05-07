//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
// This file is part of the TileLang Triton-frontend integration. It re-uses
// the vendored microsoft/triton-shared PtrAnalysis (commit 08684f9, 2025-12-05)
// under the original MIT license. See poc/triton_frontend/vendored/triton_shared/
// for the upstream sources and LICENSE.
//
//===----------------------------------------------------------------------===//
//
// ptr_analysis_shim.cc --- C ABI implementation that drives PtrAnalysis.
//
// The runtime workflow (mirroring TritonToStructuredPass.cpp::runOnOperation):
//
//     ctx = tl_pa_context_create();
//     mod = tl_pa_module_parse(ctx, text, n, &st);
//     tl_pa_run_rewrite(mod, gs, unsafe_mask);
//     out = tl_pa_module_to_string(mod);
//
// The actual PtrAnalysis call sequence is identical to upstream:
//   PtrAnalysis pa(enable_gs);
//   pa.initializeMaybeStructuredArgs(moduleOp);
//   pa.rewriteOp(moduleOp, useUnsafeMask);
//   moduleOp.walk([&](GetStructuredStateOp op){ pa.rewriteGetStructuredStateOp(op); });
//
// We intentionally skip the `runTritonToStructuredPrepass` step: the prepass
// is required only when a downstream pipeline still needs the
// `tts.get_structured_state` book-keeping. Callers that want the full
// triton-to-structured pass should use the `tl_pa_run_rewrite` wrapper, which
// runs both phases. Today the prepass entry point is a TODO (see below) until
// the TritonStructured dialect is vendored under integration #5.
//
//===----------------------------------------------------------------------===//

#include "ptr_analysis_shim.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// Vendored triton-shared headers. The TritonStructured dialect (sibling
// integration #5) is not yet vendored; the include is gated so this TU can
// compile against both layouts. When integration #5 lands, drop the gate.
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#if __has_include("triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h")
#  include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#  define TL_PA_HAVE_TRITON_STRUCTURED 1
#else
#  define TL_PA_HAVE_TRITON_STRUCTURED 0
#endif

// Upstream Triton dialect. TODO(integration#5): once Triton sources are
// vendored, drop the gate and require the include.
#if __has_include("triton/Dialect/Triton/IR/Dialect.h")
#  include "triton/Dialect/Triton/IR/Dialect.h"
#  define TL_PA_HAVE_TRITON 1
#else
#  define TL_PA_HAVE_TRITON 0
#endif

namespace {

struct ContextImpl {
  mlir::MLIRContext ctx;
  std::string lastError;
  // Persists the most recent value handed back via tl_pa_take_last_error so
  // the returned `const char*` stays valid until the next take_last_error call.
  std::string lastErrorReturned;
};

struct ModuleImpl {
  ContextImpl* parent = nullptr;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string statesJson;
};

void setError(ContextImpl* ctx, std::string msg) {
  if (ctx) ctx->lastError = std::move(msg);
}

}  // namespace

extern "C" {

TLPtrAnalysisContext* tl_pa_context_create(void) {
  auto* impl = new ContextImpl();
  // getDialectRegistry() returns the context's internal registry by reference;
  // mutating it directly is sufficient. Do NOT appendDialectRegistry(reg) on
  // the same registry -- that's a self-append and can cause subtle MLIR state
  // issues in future versions.
  auto& reg = impl->ctx.getDialectRegistry();
  reg.insert<mlir::arith::ArithDialect,
             mlir::math::MathDialect,
             mlir::affine::AffineDialect,
             mlir::scf::SCFDialect,
             mlir::tensor::TensorDialect,
             mlir::memref::MemRefDialect>();
#if TL_PA_HAVE_TRITON
  reg.insert<mlir::triton::TritonDialect>();
#endif
#if TL_PA_HAVE_TRITON_STRUCTURED
  reg.insert<mlir::tts::TritonStructuredDialect>();
#endif
  impl->ctx.loadAllAvailableDialects();
  return reinterpret_cast<TLPtrAnalysisContext*>(impl);
}

void tl_pa_context_destroy(TLPtrAnalysisContext* ctx) {
  delete reinterpret_cast<ContextImpl*>(ctx);
}

TLPtrAnalysisModule* tl_pa_module_parse(TLPtrAnalysisContext* ctx,
                                        const char* mlir_text,
                                        size_t mlir_text_len,
                                        TLPtrAnalysisStatus* status) {
  if (!ctx || !mlir_text) {
    if (status) *status = TL_PA_ERR_NULL_HANDLE;
    return nullptr;
  }
  auto* cimpl = reinterpret_cast<ContextImpl*>(ctx);
  llvm::StringRef sr(mlir_text, mlir_text_len);
  auto owning = mlir::parseSourceString<mlir::ModuleOp>(sr, &cimpl->ctx);
  if (!owning) {
    setError(cimpl, "parseSourceString<ModuleOp> failed");
    if (status) *status = TL_PA_ERR_PARSE;
    return nullptr;
  }
  if (mlir::failed(mlir::verify(*owning))) {
    setError(cimpl, "verify(ModuleOp) failed");
    if (status) *status = TL_PA_ERR_VERIFY;
    return nullptr;
  }
  auto* m = new ModuleImpl();
  m->parent = cimpl;
  m->module = std::move(owning);
  if (status) *status = TL_PA_OK;
  return reinterpret_cast<TLPtrAnalysisModule*>(m);
}

void tl_pa_module_destroy(TLPtrAnalysisModule* mod) {
  delete reinterpret_cast<ModuleImpl*>(mod);
}

char* tl_pa_module_to_string(TLPtrAnalysisModule* mod) {
  if (!mod) return nullptr;
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
  std::string out;
  llvm::raw_string_ostream os(out);
  m->module.get().print(os);
  os.flush();
  char* buf = static_cast<char*>(std::malloc(out.size() + 1));
  if (!buf) return nullptr;
  std::memcpy(buf, out.data(), out.size());
  buf[out.size()] = '\0';
  return buf;
}

void tl_pa_string_free(char* s) { std::free(s); }

TLPtrAnalysisStatus tl_pa_run_rewrite(TLPtrAnalysisModule* mod,
                                      int enable_make_gather_scatter_tensor_ptr,
                                      int use_unsafe_mask) {
  if (!mod) return TL_PA_ERR_NULL_HANDLE;
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
#if !TL_PA_HAVE_TRITON_STRUCTURED || !TL_PA_HAVE_TRITON
  setError(m->parent,
           "TritonStructured/Triton dialect not yet vendored: rebuild after "
           "integration #5 with -DTRITON_INSTALL_DIR set.");
  return TL_PA_ERR_INTERNAL;
#else
  mlir::ModuleOp moduleOp = m->module.get();
  mlir::tts::PtrAnalysis pa(
      static_cast<bool>(enable_make_gather_scatter_tensor_ptr));
  pa.initializeMaybeStructuredArgs(moduleOp);
  if (mlir::failed(pa.rewriteOp(moduleOp,
                                static_cast<bool>(use_unsafe_mask)))) {
    setError(m->parent, "PtrAnalysis::rewriteOp returned failure");
    return TL_PA_ERR_REWRITE;
  }
  moduleOp.walk([&](mlir::tts::GetStructuredStateOp op) {
    (void)pa.rewriteGetStructuredStateOp(op);
  });
  return TL_PA_OK;
#endif
}

const char* tl_pa_extract_states_json(TLPtrAnalysisModule* mod) {
  if (!mod) return "";
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
  // Best-effort: walk tts.make_tptr ops and serialize their attributes. We
  // intentionally avoid pulling in nlohmann::json; the format is a tiny
  // hand-rolled JSON array. Each entry mirrors the public fields of
  // mlir::tts::PtrState (offsets/sizes/strides/source).
  std::ostringstream os;
  os << "[";
#if TL_PA_HAVE_TRITON_STRUCTURED
  bool first = true;
  m->module.get().walk([&](mlir::tts::MakeTensorPtrOp op) {
    if (!first) os << ",";
    first = false;
    std::string opStr;
    {
      llvm::raw_string_ostream s(opStr);
      op->print(s);
    }
    // Full RFC-8259 escaping for strings: backslash, quote, the named control
    // chars (\b \f \n \r \t), and \uXXXX for everything else in U+0000..U+001F.
    // Bytes >= 0x20 (including UTF-8 continuation bytes) pass through verbatim.
    std::string esc;
    esc.reserve(opStr.size());
    for (unsigned char uc : opStr) {
      switch (uc) {
        case '\\': esc += "\\\\"; break;
        case '"':  esc += "\\\""; break;
        case '\b': esc += "\\b";  break;
        case '\f': esc += "\\f";  break;
        case '\n': esc += "\\n";  break;
        case '\r': esc += "\\r";  break;
        case '\t': esc += "\\t";  break;
        default:
          if (uc < 0x20) {
            static const char kHex[] = "0123456789abcdef";
            char buf[7] = {'\\', 'u', '0', '0',
                           kHex[(uc >> 4) & 0xF], kHex[uc & 0xF], '\0'};
            esc += buf;
          } else {
            esc += static_cast<char>(uc);
          }
          break;
      }
    }
    os << "{\"op\":\"" << esc << "\"}";
  });
#endif
  os << "]";
  m->statesJson = os.str();
  return m->statesJson.c_str();
}

const char* tl_pa_take_last_error(TLPtrAnalysisContext* ctx) {
  if (!ctx) return "";
  auto* c = reinterpret_cast<ContextImpl*>(ctx);
  // The header documents this as "retrieve and clear". Stash the message in a
  // per-context buffer that survives until the *next* call (so the returned
  // pointer remains valid for the duration of the caller's expression), then
  // empty the live `lastError` so subsequent calls don't see stale state.
  c->lastErrorReturned = std::move(c->lastError);
  c->lastError.clear();
  return c->lastErrorReturned.c_str();
}

}  // extern "C"
