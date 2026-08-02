/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

/*!
 * \file src/transform/vendored/allocate.cc
 * \brief Implementation of the TileLang-private `tilelang::tl_tir::Allocate`
 *        IR node. See `allocate.h` for rationale.
 */

#include "allocate.h"
// CPPMEGA: include the pass-through helper header here purely to force the
// build system to syntax-check it. The header is consumed by individual
// TileLang transforms as they add `VisitStmt(const Stmt&)` overrides for the
// "tilelang.Allocate" NodeFunctor drift; including it here ensures the
// header stays compilable even if no transform yet uses it.
#include "allocate_visit_passthrough.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tilelang {
namespace tl_tir {

TVM_FFI_STATIC_INIT_BLOCK() { AllocateNode::RegisterReflection(); }

Allocate::Allocate(Var buffer_var, DataType dtype,
                   tvm::ffi::Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body,
                   tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations,
                   Span span) {
  TVM_FFI_ICHECK(body.defined());
  TVM_FFI_ICHECK(condition.defined());
  tvm::ffi::ObjectPtr<AllocateNode> node =
      tvm::ffi::make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tilelang.Allocate",
      [](Var buffer_var, DataType dtype, tvm::ffi::Array<PrimExpr> extents,
         PrimExpr condition, Stmt body,
         tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations,
         Span span) {
        return Allocate(std::move(buffer_var), dtype, std::move(extents),
                        std::move(condition), std::move(body),
                        std::move(annotations), std::move(span));
      });
}

} // namespace tl_tir
} // namespace tilelang
