/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file frontend_legalize.cc
 * \brief Legalize the program from frontend
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "vendored/let_stmt.h"

namespace tvm {
namespace tl {

using namespace tirx;
// Use TileLang vendored LetStmt (with `body` field). See vendored/let_stmt.h.
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

class LetInliner : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LetInliner substituter(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const ForNode *node) final {
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_++;
    }
    auto n = StmtExprMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_--;
    }
    return n;
  }

  PrimExpr VisitExpr_(const VarNode *node) final {
    if (let_bindings_.count(node)) {
      return arith::IRMutatorWithAnalyzer::VisitExpr(let_bindings_[node]);
    } else {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }
  }

  // CPPMEGA: vendored `tl_tir::LetStmt` is not part of apache/tvm's TIR class
  // hierarchy any more, so we cannot mark this `final` (the base mutator has
  // no matching virtual to override). Instead we intercept LetStmt in
  // `VisitStmt(const Stmt&)` below — that visitor is virtual on the base.
  Stmt VisitStmt_(const LetStmtNode *node) {
    let_bindings_[node->var.get()] = node->value;
    return arith::IRMutatorWithAnalyzer::VisitStmt(node->body);
  }

  Stmt VisitStmt(const Stmt &stmt) override {
    if (const auto *node = stmt.as<LetStmtNode>()) {
      return VisitStmt_(node);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt(stmt);
  }

  PrimExpr VisitExpr_(const LetNode *node) final {
    let_bindings_[node->var.get()] = node->value;
    return arith::IRMutatorWithAnalyzer::VisitExpr(node->body);
  }

  int parallel_for_scope_ = 0;
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;
};

using namespace tirx::transform;

Pass LetInline() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LetInliner::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LetInline", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LetInline", LetInline);
}

} // namespace tl
} // namespace tvm
