// Vendored helper for triton-shared integration with the unified fused-kernel compiler.
// Copyright (c) 2026 Project Contributors.
// Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
// Licensed under the MIT License.
//
// Public registration helper. Downstream pybind modules call this when they
// construct an `mlir::MLIRContext` to make the `tts` dialect (and the upstream
// `tt` Triton dialect that `tts` builds on) available for parsing/printing.

#ifndef TRITON_SHARED_VENDORED_REGISTER_TRITON_STRUCTURED_H
#define TRITON_SHARED_VENDORED_REGISTER_TRITON_STRUCTURED_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
}  // namespace mlir

namespace mlir {
namespace triton_shared_vendored {

/// Register the `tts` (TritonStructured) dialect into the given registry.
/// Also pulls in the upstream OpenAI Triton `tt` dialect since `tts` ops
/// reference `tt`-typed values, plus the MLIR built-in dialects that the
/// vendored AnalysisStructured pipeline emits (`arith`, `func`, `scf`,
/// `tensor`, `memref`).
void registerTritonStructured(::mlir::DialectRegistry &registry);

/// Convenience: load the dialect immediately into a context that is already
/// constructed. Equivalent to creating a registry, calling
/// `registerTritonStructured(registry)`, appending it, and forcing-load the
/// `tts` dialect.
void registerTritonStructured(::mlir::MLIRContext &context);

}  // namespace triton_shared_vendored
}  // namespace mlir

#endif  // TRITON_SHARED_VENDORED_REGISTER_TRITON_STRUCTURED_H
