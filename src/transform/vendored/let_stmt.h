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
 * \file src/transform/vendored/let_stmt.h
 * \brief Vendored, TileLang-internal LetStmt IR node.
 *
 * apache/tvm renamed `tirx::LetStmt` to `tirx::Bind` and dropped the `body`
 * field (BindNode is `{Var var; PrimExpr value;}` only). TileLang has 50+
 * sites across 26 transform/op files that rely on the legacy 3-arg form
 * `LetStmt(var, value, body)` and `let->body` accessor. Rather than rewrite
 * every site we vendor the old class as a TileLang-private node.
 *
 * This node MUST be lowered into the tirx-equivalent
 * `SeqStmt({tirx::Bind(var, value), body})` (see lower_let_stmt.cc) before
 * the IR is handed to apache/tvm's tirx pipeline — apache codegen does NOT
 * know how to traverse `tilelang.LetStmt`.
 *
 * The node uses the type key `"tilelang.LetStmt"` to avoid collision with
 * any current or future apache `tir.LetStmt` / `tirx.*`.
 */
#ifndef TILELANG_TRANSFORM_VENDORED_LET_STMT_H_
#define TILELANG_TRANSFORM_VENDORED_LET_STMT_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/var.h>

#include <utility>

namespace tilelang {
namespace tl_tir {

using tvm::Span;
using tvm::PrimExpr;  // PrimExpr lives at tvm:: top-level (not tirx::)
using tvm::tirx::Stmt;
using tvm::tirx::StmtNode;
using tvm::tirx::Var;

/*!
 * \brief Let binding, bind var to value, then run body.
 *
 * Vendored from apache/tvm pre-tirx `tirx::LetStmtNode`. Field order, default
 * values, and Span semantics match the upstream definition exactly.
 */
class LetStmtNode : public StmtNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be bound. */
  PrimExpr value;
  /*! \brief The body block. */
  Stmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LetStmtNode>()
        .def_ro("var", &LetStmtNode::var, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("value", &LetStmtNode::value)
        .def_ro("body", &LetStmtNode::body);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tilelang.LetStmt", LetStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to LetStmtNode.
 * \sa LetStmtNode
 */
class LetStmt : public Stmt {
 public:
  TVM_DLL LetStmt(Var var, PrimExpr value, Stmt body, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LetStmt, Stmt, LetStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetStmtNode);
};

}  // namespace tl_tir
}  // namespace tilelang

#endif  // TILELANG_TRANSFORM_VENDORED_LET_STMT_H_
