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
#include <unordered_set>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
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

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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
  reg.insert<mlir::triton::TritonDialect>();
  reg.insert<mlir::tts::TritonStructuredDialect>();
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
  mlir::ModuleOp moduleOp = m->module.get();

  std::unordered_set<std::string> seenDiagnostics;
  mlir::ScopedDiagnosticHandler diagHandler(moduleOp.getContext(),
      [&](mlir::Diagnostic& diag) {
        std::string diagStr;
        llvm::raw_string_ostream os(diagStr);
        diag.print(os);
        os.flush();
        if (seenDiagnostics.insert(diagStr).second) {
          // New diagnostic. Allow next handler to process it (e.g. print it).
          return mlir::failure();
        }
        // Duplicate diagnostic. Return success to consume/suppress it.
        return mlir::success();
      });

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
}

TLPtrAnalysisStatus tl_pa_run_structured_to_memref(TLPtrAnalysisModule* mod) {
  if (!mod) return TL_PA_ERR_NULL_HANDLE;
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
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
}

}  // extern "C"

namespace {

// Pretty-print a single OpFoldResult to a compact string suitable for the
// JSON value inside an `offsets`/`sizes`/`strides`/`shape` array.
//
//   - Static IntegerAttr  -> the bare integer literal ("0", "16", "-1")
//   - Dynamic Value       -> the SSA name without type ("%c5", "%arg2")
//   - Anything else       -> fall back to `operator<<`
//
// We deliberately strip the leading whitespace `operator<<` adds for Values
// (Value::print emits "%name : type" with surrounding spaces) so the JSON
// stays consistent with the existing regex-based fallback in
// `ptr_analysis.py::_TPTR_RE`, which expects bare SSA names like `%2`.
std::string fmtOpFoldResult(mlir::OpFoldResult ofr, mlir::AsmState& asmState) {
  std::string out;
  llvm::raw_string_ostream os(out);
  if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(ofr)) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      // getInt() handles signed; use APInt to be safe across widths.
      intAttr.getValue().print(os, /*isSigned=*/true);
    } else {
      attr.print(os);
    }
  } else if (auto val = llvm::dyn_cast_if_present<mlir::Value>(ofr)) {
    val.printAsOperand(os, asmState);
  } else {
    os << "<null>";
  }
  os.flush();
  return out;
}

// Pretty-print an MLIR Value as its SSA operand name only (no type suffix).
std::string fmtValueAsOperand(mlir::Value v, mlir::AsmState& asmState) {
  if (!v) return {};
  std::string out;
  llvm::raw_string_ostream os(out);
  v.printAsOperand(os, asmState);
  os.flush();
  return out;
}

// RFC-8259 escape a UTF-8 string for embedding inside JSON quotes. This must
// stay byte-identical to nlohmann::json::dump()'s default escaping so the two
// encoder branches agree (regression-tested in test_ptr_analysis.py).
std::string jsonEscape(llvm::StringRef in) {
  std::string esc;
  esc.reserve(in.size() + 2);
  for (unsigned char uc : in) {
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
  return esc;
}

}  // namespace

extern "C" {

const char* tl_pa_extract_states_json(TLPtrAnalysisModule* mod) {
  if (!mod) return "";
  auto* m = reinterpret_cast<ModuleImpl*>(mod);
  // Walk tts.make_tptr ops and serialize the recovered PtrState fields. Two
  // encoders are supported and are REQUIRED to emit byte-identical output
  // for the schema below -- the regression test in
  // tests/test_ptr_analysis.py guards the contract:
  //
  //   `[]` when there are no make_tptr ops; otherwise an array of objects
  //   whose keys appear in this fixed order so both encoder branches agree
  //   byte-for-byte:
  //
  //     {"op": "<escaped printed op>",
  //      "result_ssa": "%2",
  //      "source": "%arg0",
  //      "offsets": ["0", "%c5"],
  //      "sizes":   ["4", "256"],
  //      "strides": ["1", "256"],
  //      "shape":   ["0", "0"],
  //      "order":   [0, 1]}
  //
  // Empty arrays / null source are still emitted explicitly so Python's
  // Path-A parser (see _parse_states_json) picks them up without needing
  // to fall back to printed-op regex parsing.
  //
  // The default is a hand-rolled RFC-8259 escaper (no third-party deps);
  // -DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON swaps it for nlohmann::json.
  mlir::ModuleOp moduleOp = m->module.get();
  // Build one AsmState per module so SSA numbering matches what the module
  // printer would use -- this keeps the source/offsets/sizes/strides strings
  // consistent with the `op` field's printed form.
  mlir::AsmState asmState(moduleOp);

#  ifdef TL_PA_USE_NLOHMANN_JSON
  // ---- nlohmann::json encoder ---------------------------------------------
  // nlohmann::json's default object is an unordered map; to match the
  // hand-rolled encoder byte-for-byte we build an ordered_json instead, which
  // preserves insertion order on dump().
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  moduleOp.walk([&](mlir::tts::MakeTensorPtrOp op) {
    std::string opStr;
    {
      llvm::raw_string_ostream s(opStr);
      op->print(s);
    }
    nlohmann::ordered_json entry;
    entry["op"] = opStr;
    entry["result_ssa"] = fmtValueAsOperand(op.getResult(), asmState);
    entry["source"] = fmtValueAsOperand(op.getBase(), asmState);

    auto pushList = [&](const char* key,
                        mlir::SmallVector<mlir::OpFoldResult> vals) {
      nlohmann::ordered_json arr2 = nlohmann::ordered_json::array();
      for (auto v : vals) arr2.push_back(fmtOpFoldResult(v, asmState));
      entry[key] = std::move(arr2);
    };
    pushList("offsets", op.getMixedOffsets());
    pushList("sizes",   op.getMixedSizes());
    pushList("strides", op.getMixedStrides());
    pushList("shape",   op.getMixedShape());

    nlohmann::ordered_json orderArr = nlohmann::ordered_json::array();
    for (int32_t o : op.getOrder()) orderArr.push_back(o);
    entry["order"] = std::move(orderArr);

    arr.push_back(std::move(entry));
  });
  m->statesJson = arr.dump();
#  else
  // ---- hand-rolled RFC-8259 encoder ---------------------------------------
  std::ostringstream os;
  os << "[";
  bool first = true;
  moduleOp.walk([&](mlir::tts::MakeTensorPtrOp op) {
    if (!first) os << ",";
    first = false;

    std::string opStr;
    {
      llvm::raw_string_ostream s(opStr);
      op->print(s);
    }
    std::string resultStr = fmtValueAsOperand(op.getResult(), asmState);
    std::string sourceStr = fmtValueAsOperand(op.getBase(), asmState);

    auto emitList = [&](const char* key,
                        mlir::SmallVector<mlir::OpFoldResult> vals) {
      os << ",\"" << key << "\":[";
      bool firstV = true;
      for (auto v : vals) {
        if (!firstV) os << ",";
        firstV = false;
        os << "\"" << jsonEscape(fmtOpFoldResult(v, asmState)) << "\"";
      }
      os << "]";
    };

    // Keys must be emitted in the same order as the nlohmann path above
    // (op, result_ssa, source, offsets, sizes, strides, shape, order) so
    // the two encoders stay byte-identical.
    os << "{\"op\":\"" << jsonEscape(opStr) << "\"";
    os << ",\"result_ssa\":\"" << jsonEscape(resultStr) << "\"";
    os << ",\"source\":\"" << jsonEscape(sourceStr) << "\"";
    emitList("offsets", op.getMixedOffsets());
    emitList("sizes",   op.getMixedSizes());
    emitList("strides", op.getMixedStrides());
    emitList("shape",   op.getMixedShape());

    os << ",\"order\":[";
    bool firstO = true;
    for (int32_t o : op.getOrder()) {
      if (!firstO) os << ",";
      firstO = false;
      os << o;
    }
    os << "]";

    os << "}";
  });
  os << "]";
  m->statesJson = os.str();
#  endif  // TL_PA_USE_NLOHMANN_JSON
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
