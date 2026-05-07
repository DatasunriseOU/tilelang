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
// ptr_analysis_shim.h --- C-callable surface around mlir::tts::PtrAnalysis
//
// The Python bindings consume MLIR exclusively as text (parse / print) so we
// can keep the ABI stable across MLIR versions and avoid leaking
// mlir::MLIRContext lifetime into Python. Callers that need richer access
// should use the pybind11 layer in pybind_module.cc, which wraps the same C
// surface with std::string/std::vector for ergonomics.
//
//===----------------------------------------------------------------------===//

#ifndef TILELANG_TRITON_FRONTEND_PTR_ANALYSIS_SHIM_H
#define TILELANG_TRITON_FRONTEND_PTR_ANALYSIS_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Opaque handles (heap-allocated; freed via the *_destroy entry points).
typedef struct TLPtrAnalysisContext TLPtrAnalysisContext;
typedef struct TLPtrAnalysisModule  TLPtrAnalysisModule;

// Status codes returned by every entry point.
typedef enum {
  TL_PA_OK              = 0,
  TL_PA_ERR_PARSE       = 1,
  TL_PA_ERR_VERIFY      = 2,
  TL_PA_ERR_REWRITE     = 3,
  TL_PA_ERR_NULL_HANDLE = 4,
  TL_PA_ERR_INTERNAL    = 5,
} TLPtrAnalysisStatus;

// ---- Context lifecycle -----------------------------------------------------

// Construct an MLIRContext with TritonStructured + Triton + scf/arith/memref
// dialects registered. The returned handle owns the context.
TLPtrAnalysisContext* tl_pa_context_create(void);
void                  tl_pa_context_destroy(TLPtrAnalysisContext* ctx);

// ---- Module lifecycle ------------------------------------------------------

// Parse `mlir_text` into a ModuleOp owned by this handle. Returns NULL and
// sets *status on failure. The returned module borrows `ctx`; do not destroy
// `ctx` before this module.
TLPtrAnalysisModule* tl_pa_module_parse(TLPtrAnalysisContext* ctx,
                                        const char* mlir_text,
                                        size_t mlir_text_len,
                                        TLPtrAnalysisStatus* status);
void                 tl_pa_module_destroy(TLPtrAnalysisModule* mod);

// Print the (possibly rewritten) module to a freshly allocated, NUL-terminated
// C string. Caller must free with `tl_pa_string_free`.
char*                tl_pa_module_to_string(TLPtrAnalysisModule* mod);
void                 tl_pa_string_free(char* s);

// ---- Driver entry points ---------------------------------------------------

// Run `mlir::tts::PtrAnalysis::rewriteOp` over the top-level ModuleOp.
// `enable_make_gather_scatter_tensor_ptr` and `use_unsafe_mask` mirror the
// upstream pass options. Diagnostics are captured and returned via
// `tl_pa_take_last_error`.
TLPtrAnalysisStatus tl_pa_run_rewrite(TLPtrAnalysisModule* mod,
                                      int enable_make_gather_scatter_tensor_ptr,
                                      int use_unsafe_mask);

// Walk every value in `knownPtrs` after a rewrite and emit a JSON description
// of the recovered PtrStates. The buffer is owned by the shim until the next
// call to this function on the same module.
const char*          tl_pa_extract_states_json(TLPtrAnalysisModule* mod);

// Retrieve and clear the last error message produced by any of the entry
// points above. Returns an empty string if no error is pending.
const char*          tl_pa_take_last_error(TLPtrAnalysisContext* ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TILELANG_TRITON_FRONTEND_PTR_ANALYSIS_SHIM_H
