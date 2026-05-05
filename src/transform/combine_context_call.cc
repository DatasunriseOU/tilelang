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
 *  Combine calls into context related function into one.
 *
 * \file combine_context_call.cc
 *
 * Vendored into TileLang from apache/tvm fork (pre-tirx refactor).
 * Migrated to tirx namespace. Registered as `tl.transform.CombineContextCall`
 * so `tilelang/engine/lower.py` can keep invoking it without conditional logic.
 *
 * Uses TileLang-vendored `tilelang::tl_tir::LetStmt` (3-arg form). The pass
 * `tl.transform.LowerTileLangLetStmt` desugars these to
 * `SeqStmt({tirx::Bind(var, value), body})` before apache/tvm sees the IR.
 *
 * --------------------------------------------------------------------------
 * EFFECTIVELY NO-OP IN CPPMEGA / APACHE-LATEST WORLD
 * --------------------------------------------------------------------------
 * The original purpose of this pass was to deduplicate repeated calls to the
 * `tvm_thread_context(ctx)` builtin emitted by apache TVM's host-side
 * `make_packed_api` lowering, by Let-binding the result into a `ctx_cache_`
 * variable inside each `thread_extent` / `coproc_uop_scope` / parallel-For
 * scope.
 *
 * apache/tvm latest (PR #18776, commit 82b01c9486) **deleted both**:
 *   1. the `tvm_thread_context` builtin (`builtin::tvm_thread_context()`),
 *   2. all call sites that emitted it (`make_packed_api.cc`, `arg_binder.cc`,
 *      runtime stub generation).
 *
 * Verified for our tree:
 *   $ grep -rn 'tvm_thread_context' \
 *         /tmp/tl_apache_tvm_swap/3rdparty/tvm \
 *         /tmp/tl_apache_tvm_swap/src
 *   (only this file's own commentary appears)
 *
 * TileLang itself never emits `tvm_thread_context` from any of its codegen
 * or transform passes, so no IR reaching this point can contain such a call.
 * Vendoring the builtin and the original `VisitExpr_(CallNode)` merge branch
 * would only add unreachable code.
 *
 * What we keep:
 *   - The scope-walking visitor (`AttrStmt::thread_extent`,
 *     `tl_attr::coproc_uop_scope`, parallel `For`) and the `BuildContext`
 *     LetStmt-emission helper, so the pass shape matches the original.
 *     `ctx_map_` will simply remain empty and `BuildContext` will return
 *     `body` unchanged.
 *   - Registration as `tl.transform.CombineContextCall` so the python-side
 *     pipeline call in `tilelang/engine/lower.py:215` resolves.
 *
 * What we deliberately drop:
 *   - The `VisitExpr_(CallNode)` merge branch that compared against
 *     `builtin::tvm_thread_context()`. There is no equivalent builtin in
 *     apache-latest tirx and no producer in TileLang, so it would be dead.
 *
 * If apache reintroduces a thread-context-style builtin (or TileLang grows
 * a runtime context handle of its own that benefits from CSE in host code),
 * port the merge branch back here against that new symbol.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

#include "tirx/transform/ir_utils.h"
#include "vendored/let_stmt.h"
#include "vendored/tl_attr.h"

namespace tvm {
namespace tirx {

using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

// Calculate the statistics of packed function.
// These information are needed during codegen.
//
// See the file-level header comment for why the merge branch is intentionally
// absent: apache-latest deleted both `tvm_thread_context` and all its
// emitters, and TileLang never emitted the builtin itself.
class ContextCallCombiner final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == ::tilelang::tl_attr::coproc_uop_scope) {
      // Map of comparison expression to variable
      std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> temp;
      std::swap(temp, ctx_map_);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      std::swap(temp, ctx_map_);
      return BuildContext(temp, stmt);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      // Map of comparison expression to variable
      std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> temp;
      std::swap(temp, ctx_map_);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      std::swap(temp, ctx_map_);
      return BuildContext(temp, stmt);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt Combine(Stmt stmt) { return BuildContext(ctx_map_, this->VisitStmt(stmt)); }

 private:
  // Uses vendored `tilelang::tl_tir::LetStmt` (3-arg form).
  // `tl.transform.LowerTileLangLetStmt` desugars to
  // `SeqStmt({tirx::Bind(var, value), body})` before apache/tvm pipeline.
  static Stmt BuildContext(
      const std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual>& cmap, Stmt body) {
    for (const auto& kv : cmap) {
      body = LetStmt(kv.second, kv.first, body);
    }
    return body;
  }
  // Map of comparison expression to variable
  std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> ctx_map_;
};

namespace transform {

Pass CombineContextCall() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    if (IsHostFunc(f).value_or(false)) {
      f.CopyOnWrite()->body = ContextCallCombiner().Combine(f->body);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CombineContextCall", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CombineContextCall", CombineContextCall);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
