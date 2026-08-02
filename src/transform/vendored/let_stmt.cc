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
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/transform/vendored/let_stmt.cc
 * \brief Vendored TileLang-internal LetStmt IR node implementation.
 *
 * Mirrors the pre-tirx `tirx::LetStmt` constructor and FFI registration but
 * under the type key `"tilelang.LetStmt"` and the FFI symbol
 * `tilelang.LetStmt`. apache/tvm's `tir.LetStmt` / `tirx.Bind` keys are left
 * untouched.
 */

#include "let_stmt.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>

namespace tilelang {
namespace tl_tir {

using tvm::PointerTypeNode;

TVM_FFI_STATIC_INIT_BLOCK() { LetStmtNode::RegisterReflection(); }

// LetStmt constructor — matches the legacy tirx::LetStmt signature exactly.
LetStmt::LetStmt(Var var, PrimExpr value, Stmt body, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(body.defined());
  auto vdtype = value.dtype();
  // It is still valid to bind a pointer-typed var to a value of type handle.
  if (var->type_annotation.as<PointerTypeNode>()) {
    TVM_FFI_ICHECK(vdtype.is_handle());
  } else {
    TVM_FFI_ICHECK_EQ(value.dtype(), var.dtype());
  }

  tvm::ffi::ObjectPtr<LetStmtNode> node = tvm::ffi::make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tilelang.LetStmt",
                        [](Var var, PrimExpr value, Stmt body, Span span) {
                          return LetStmt(std::move(var), std::move(value),
                                         std::move(body), std::move(span));
                        });
}

} // namespace tl_tir
} // namespace tilelang
