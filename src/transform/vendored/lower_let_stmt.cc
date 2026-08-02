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
 * \file src/transform/vendored/lower_let_stmt.cc
 * \brief Pass that rewrites every TileLang-private `tilelang::tl_tir::LetStmt`
 *        into the tirx-equivalent `SeqStmt({tirx::Bind(var, value), body})`.
 *
 * apache/tvm's tirx pipeline does not know about `tilelang.LetStmt` — its
 * `StmtFunctor` vtable only dispatches on built-in tirx node types and will
 * throw `InternalError: Do not have a default for tilelang.LetStmt` if an
 * un-lowered LetStmt reaches it. This pass MUST run before any apache/tvm
 * tirx pass that traverses the IR (see tilelang/engine/lower.py).
 *
 * Registered as `tl.transform.LowerTileLangLetStmt`.
 */

#include "let_stmt.h"

// CPPMEGA: Include allocate.h + passthrough helper so this converter can pass
// `tilelang::tl_tir::AllocateNode` through unchanged. Apache `StmtFunctor`
// has no vtable entry for vendored Allocate; without this passthrough, a
// LetStmt body containing an Allocate would crash apache's vtable dispatch
// the moment our `StmtMutator::VisitStmt` fallback recurses into apache's
// `VisitStmt_(const XNode*)` impls (which then call `this->VisitStmt(child)`,
// which dispatches by typekey and explodes on `tilelang.Allocate`).
#include "allocate.h"
#include "allocate_visit_passthrough.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tilelang {
namespace transform {

using tvm::IRModule;
using tvm::tirx::Bind;
using tvm::tirx::PrimFunc;
using tvm::tirx::PrimFuncNode;
using tvm::tirx::SeqStmt;
using tvm::tirx::Stmt;
using tvm::tirx::StmtMutator;
using tvm::transform::Pass;
using tvm::transform::PassContext;

namespace {

/*!
 * \brief Mutator that rewrites every `tilelang::tl_tir::LetStmtNode` it
 *        encounters into `SeqStmt({Bind(var, value), body})`.
 *
 * Implementation note: the apache/tvm tirx `StmtFunctor` dispatches via a
 * fixed vtable initialised at startup that only knows about built-in tirx
 * node types. An unknown node like `tilelang.LetStmt` would fall through to
 * `VisitStmtDefault_` and throw. We intercept it by overriding the top-level
 * `VisitStmt(const Stmt&)` method and short-circuiting before the vtable
 * dispatch. Recursion into `body` then re-enters the normal mutator path.
 */
class LowerTileLangLetStmtMutator : public StmtMutator {
public:
  bool found_any() const { return found_any_; }

protected:
  Stmt VisitStmt(const Stmt &stmt) override {
    if (const auto *let = stmt.as<tl_tir::LetStmtNode>()) {
      found_any_ = true;
      // Recurse into value (via VisitExpr) and body first, preserving
      // bottom-up rewriting semantics.
      // Recurse via our own VisitStmt so nested LetStmts (including a body
      // that is itself a LetStmt) are also rewritten.
      auto value = VisitExpr(let->value);
      auto body = this->VisitStmt(let->body);
      return SeqStmt({Bind(let->var, value, let->span), body}, let->span);
    }
    // CPPMEGA: Pre-emptive pass-through for vendored
    // `tilelang::tl_tir::AllocateNode`. Apache's `StmtFunctor` vtable does not
    // know this type — falling through to `StmtMutator::VisitStmt` on an
    // Allocate would route via the vtable and crash with
    // "NodeFunctor calls un-registered function on type tilelang.Allocate".
    // The passthrough helper recurses into `Allocate::body` via our own
    // `VisitStmt` (which handles nested LetStmts) and rebuilds the Allocate
    // unchanged. The downstream `LowerTileLangAllocate` pass converts it.
    if (auto out = ::tilelang::tl_tir::TryVisitAllocateMutator(this, stmt)) {
      return *out;
    }
    return StmtMutator::VisitStmt(stmt);
  }

private:
  bool found_any_{false};
};

} // namespace

Pass LowerTileLangLetStmt() {
  auto pass_func = [](PrimFunc f, IRModule, PassContext) -> PrimFunc {
    LowerTileLangLetStmtMutator mutator;
    Stmt new_body = mutator(std::move(f->body));
    if (!mutator.found_any()) {
      // No-op fast path — leave the function untouched.
      return f;
    }
    auto *node = f.CopyOnWrite();
    node->body = std::move(new_body);
    return f;
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, /*opt_level=*/0, "tl.LowerTileLangLetStmt", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerTileLangLetStmt",
                        LowerTileLangLetStmt);
}

} // namespace transform
} // namespace tilelang
