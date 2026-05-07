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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

// Vendored triton-shared headers. The TritonStructured dialect (sibling
// integration #5) is not yet vendored; the include is gated so this TU can
// compile against both layouts. When integration #5 lands, drop the gate.
//
// Note on stub builds: `triton-shared/AnalysisStructured/PtrAnalysis.h`
// transitively references `mlir::triton::AddPtrOp`, `MakeRangeOp`, etc., so
// it cannot be parsed unless the upstream Triton dialect headers are on the
// include path. We therefore gate it on the same condition we use for the
// dialect symbols; when the gate is false the C ABI still compiles but
// `tl_pa_run_rewrite` returns TL_PA_ERR_INTERNAL.
#if __has_include("triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h") \
    && __has_include("triton/Dialect/Triton/IR/Dialect.h")
#  include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#  include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
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

// Optional nlohmann::json encoder. Compiled in via
// -DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON in CMake (see CMakeLists.txt).
#ifdef TL_PA_USE_NLOHMANN_JSON
#  include "nlohmann/json.hpp"
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
  // Build a DialectRegistry, append it to the context, and load. We construct
  // a separate registry rather than mutating `getDialectRegistry()` because
  // upstream MLIR (>= 18) returns it by *const* reference; the supported
  // append point is `appendDialectRegistry`.
  mlir::DialectRegistry reg;
  reg.insert<mlir::arith::ArithDialect,
             mlir::math::MathDialect,
             mlir::affine::AffineDialect,
             mlir::bufferization::BufferizationDialect,
             mlir::cf::ControlFlowDialect,
             mlir::func::FuncDialect,
             mlir::linalg::LinalgDialect,
             mlir::scf::SCFDialect,
             mlir::tensor::TensorDialect,
             mlir::memref::MemRefDialect>();
#if TL_PA_HAVE_TRITON
  reg.insert<mlir::triton::TritonDialect>();
#endif
#if TL_PA_HAVE_TRITON_STRUCTURED
  reg.insert<mlir::tts::TritonStructuredDialect>();
#endif
  impl->ctx.appendDialectRegistry(reg);
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

// Print the module in generic op form so external parsers (jaxlib's stripped
// mlir.ir, brew LLVM's mlir-opt, etc.) can re-parse it without needing the
// Triton dialect's custom assembly registered. The output is functionally
// equivalent to the custom-form output above; it just uses the
// `"dialect.op"(operands) {attrs} : (operand_types) -> result_types` shape
// instead of dialect-specific shorthand.
char* tl_pa_module_to_generic(TLPtrAnalysisModule* mod) {
  if (!mod) return nullptr;
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
  std::string out;
  llvm::raw_string_ostream os(out);
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm();
  m->module.get().print(os, flags);
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

TLPtrAnalysisStatus tl_pa_run_structured_to_memref(TLPtrAnalysisModule* mod) {
  if (!mod) return TL_PA_ERR_NULL_HANDLE;
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
#if !TL_PA_HAVE_TRITON_STRUCTURED || !TL_PA_HAVE_TRITON
  setError(m->parent,
           "TritonStructured/Triton dialect not yet vendored: rebuild after "
           "integration #5 with -DTRITON_INSTALL_DIR set.");
  return TL_PA_ERR_INTERNAL;
#else
  // Mirror facebookincubator/triton-shared
  // lib/Conversion/StructuredToMemref/StructuredToMemrefPass.cpp::runOnOperation,
  // minus the dialect-registry entries for `tptr::TPtrDialect` /
  // `ttx::TritonTilingExtDialect` (those dialects are not vendored). The
  // TypeConverter and target legalization rules are otherwise identical to
  // upstream.
  mlir::ModuleOp moduleOp = m->module.get();

  // ---- TypeConverter: tt.ptr -> unranked memref ---------------------------
  mlir::TypeConverter typeConverter;
  typeConverter.addConversion([](mlir::Type type) { return type; });
  typeConverter.addConversion([](mlir::triton::PointerType ptrType) {
    return mlir::UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
  });
  auto materialize = [](mlir::OpBuilder& builder,
                        mlir::Type resultType,
                        mlir::ValueRange inputs,
                        mlir::Location loc) -> mlir::Value {
    return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                    inputs)
        .getResult(0);
  };
  typeConverter.addTargetMaterialization(materialize);
  typeConverter.addSourceMaterialization(materialize);

  // ---- ConversionTarget ---------------------------------------------------
  mlir::ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<
      mlir::func::FuncDialect, mlir::arith::ArithDialect,
      mlir::math::MathDialect, mlir::linalg::LinalgDialect,
      mlir::affine::AffineDialect, mlir::scf::SCFDialect,
      mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect,
      mlir::bufferization::BufferizationDialect,
      mlir::memref::MemRefDialect>();
  target.addIllegalOp<mlir::tts::LoadOp, mlir::tts::StoreOp,
                      mlir::tts::MakeTensorPtrOp>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  // ---- Patterns -----------------------------------------------------------
  mlir::RewritePatternSet patterns(moduleOp.getContext());
  mlir::triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                             typeConverter);

  if (mlir::failed(mlir::applyPartialConversion(moduleOp, target,
                                                std::move(patterns)))) {
    setError(m->parent,
             "applyPartialConversion(StructuredToMemref) returned failure");
    return TL_PA_ERR_REWRITE;
  }
  return TL_PA_OK;
#endif
}

const char* tl_pa_extract_states_json(TLPtrAnalysisModule* mod) {
  if (!mod) return "";
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
  // Walk tts.make_tptr ops and serialize their printed form. Two encoders are
  // supported and are REQUIRED to emit byte-identical output for the current
  // minimal schema -- the regression test in tests/test_ptr_analysis.py
  // guards the contract:
  //   `[]`                        -> empty
  //   `[{"op":"<escaped>"}]`      -> compact, no spaces, no trailing newline
  //
  // The default is a hand-rolled RFC-8259 escaper (no third-party deps);
  // -DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON swaps it for nlohmann::json.
#if TL_PA_HAVE_TRITON_STRUCTURED
#  ifdef TL_PA_USE_NLOHMANN_JSON
  // ---- nlohmann::json encoder ---------------------------------------------
  nlohmann::json arr = nlohmann::json::array();
  m->module.get().walk([&](mlir::tts::MakeTensorPtrOp op) {
    std::string opStr;
    {
      llvm::raw_string_ostream s(opStr);
      op->print(s);
    }
    arr.push_back({{"op", opStr}});
  });
  // dump() with no indent matches the hand-rolled compact form. Note:
  // nlohmann::json by default escapes the same RFC-8259 set the manual path
  // emits (control chars via \uXXXX, ", and \). UTF-8 continuation bytes pass
  // through unescaped because we leave ensure_ascii at default.
  m->statesJson = arr.dump();
#  else
  // ---- hand-rolled RFC-8259 encoder ---------------------------------------
  std::ostringstream os;
  os << "[";
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
  os << "]";
  m->statesJson = os.str();
#  endif  // TL_PA_USE_NLOHMANN_JSON
#else
  m->statesJson = "[]";
#endif
  return m->statesJson.c_str();
}

int tl_pa_uses_nlohmann_json(void) {
#ifdef TL_PA_USE_NLOHMANN_JSON
  return 1;
#else
  return 0;
#endif
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
