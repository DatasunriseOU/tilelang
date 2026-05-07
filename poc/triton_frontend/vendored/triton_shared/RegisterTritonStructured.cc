// Vendored helper for triton-shared integration with the unified fused-kernel compiler.
// Copyright (c) 2026 Project Contributors.
// Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
// Licensed under the MIT License.

#include "RegisterTritonStructured.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

namespace mlir {
namespace triton_shared_vendored {

void registerTritonStructured(::mlir::DialectRegistry &registry) {
  // Built-in MLIR dialects that the vendored Analysis pipeline produces.
  registry.insert<::mlir::arith::ArithDialect>();
  registry.insert<::mlir::func::FuncDialect>();
  registry.insert<::mlir::memref::MemRefDialect>();
  registry.insert<::mlir::scf::SCFDialect>();
  registry.insert<::mlir::tensor::TensorDialect>();

  // Upstream OpenAI Triton dialect (`tt`) that the `tts` dialect builds on.
  registry.insert<::mlir::triton::TritonDialect>();

  // The vendored TritonStructured dialect (`tts`).
  registry.insert<::mlir::tts::TritonStructuredDialect>();
}

void registerTritonStructured(::mlir::MLIRContext &context) {
  ::mlir::DialectRegistry registry;
  registerTritonStructured(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<::mlir::tts::TritonStructuredDialect>();
  context.loadDialect<::mlir::triton::TritonDialect>();
}

}  // namespace triton_shared_vendored
}  // namespace mlir
