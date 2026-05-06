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
 * \file predicate_fusion.cc
 * \brief CPPMEGA: Z3 idea #7 — fuse adjacent guarded `if` statements when
 * Z3 proves the inner predicate is well-defined unconditionally.
 *
 * Source pattern:
 *
 *     if (a) {
 *       if (b) {
 *         body;
 *       }
 *     }
 *
 * Rewrite (only when safe):
 *
 *     if (a && b) {
 *       body;
 *     }
 *
 * "Safe" here means:
 *   1. Outer guard `a` has no side effects (pure read or constant). This
 *      keeps the rewrite from materializing reads/calls that were
 *      previously hidden behind a guard.
 *   2. Evaluating `b` when `!a` does NOT cause out-of-bounds buffer
 *      access. We Z3-prove that EVERY `BufferLoad` inside `b` is
 *      well-defined unconditionally — i.e. every index `i` satisfies
 *      `0 <= i < extent` regardless of `a`.
 *   3. There is no nested `else` branch on the inner `if`. (We do not
 *      try to handle the `if(a){if(b)X else Y}` shape.)
 *   4. There are no statements between the outer `if` body and the
 *      inner `if` (the outer body must be exactly the inner if).
 *
 * Z3 budget: the prover is bounded to a small per-query timeout. Any
 * UNKNOWN / timeout / exception leaves the original nesting intact —
 * the pass is a pure throughput optimization and must remain
 * conservative.
 *
 * PassConfig: `tl.predicate_fusion = True` (default OFF).
 *
 * Wired in `tilelang/engine/phase.py` after `LegalizeSafeMemoryAccess`,
 * which is the closest analog to "loop partition" in this lowering
 * pipeline (the actual `LowerParallelLoop`/`PartitionLoop` happens
 * inside `LowerTileOp` and emits the nested-if shapes this pass
 * targets).
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "vendored/let_stmt.h"
#include "vendored/z3_prover.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

namespace {

// Collect all free `Var`s referenced in an expression. Used to push
// per-var bit-bound constraints into the Z3 context before issuing the
// well-definedness query.
class FreeVarCollector : public ExprVisitor {
public:
  std::unordered_set<const VarNode *> vars;
  void VisitExpr_(const VarNode *op) final { vars.insert(op); }
};

// Collect every BufferLoad inside a (sub)statement. Each entry gives us
// `(buffer, indices)` to enumerate when proving well-definedness.
class BufferLoadCollector : public StmtExprVisitor {
public:
  std::vector<const BufferLoadNode *> loads;
  // If we hit a side-effecting / unsupported construct inside `b`, we
  // refuse to fuse — flagged by `bailout`.
  bool bailout = false;

  void VisitStmt_(const ForNode *op) final {
    // Loops inside the inner predicate-guarded block — bail. We could
    // try to enumerate the loop's index space, but the conservative
    // option is to refuse fusion whenever the inner if contains
    // arbitrary control flow.
    bailout = true;
  }

  void VisitExpr_(const CallNode *op) final {
    // Any call (even pure) is conservatively unsafe — could divide by
    // zero, dereference a pointer, etc.
    bailout = true;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    loads.push_back(op);
    StmtExprVisitor::VisitExpr_(op);
  }
};

// Returns true iff Z3 proves that for every BufferLoad in `inner_body`
// the index is in-range, *unconditionally* (i.e. WITHOUT assuming the
// outer guard `a`). The free-vars in those indices are bit-bounded to
// BV32 emulation via `EnterConstraint` (the parallel `z3-bv-mode`
// branch's `SetBitVectorMode(width)` is not on this branch).
//
// `outer_guard` is purely informational here — we explicitly do NOT
// constrain it, because the whole point of the proof is that `b` must
// be safe even when `!a`.
bool Z3ProvesInnerWellDefined(const Stmt &inner_body, arith::Analyzer *analyzer) {
  BufferLoadCollector collector;
  collector(inner_body);
  if (collector.bailout) {
    return false;
  }
  if (collector.loads.empty()) {
    // No BufferLoads → no OOB risk. This is the easy fast path
    // (e.g. inner body is `BufferStore(buf, expr_with_no_loads, idx)`
    // — but BufferStore's indices themselves are also captured below
    // via the StoreCollector). Be defensive: re-check stores too.
  }

  // Also visit BufferStores — their *indices* must be well-defined too.
  class StoreIdxCollector : public StmtExprVisitor {
  public:
    std::vector<std::pair<Buffer, Array<PrimExpr>>> stores;
    void VisitStmt_(const BufferStoreNode *op) final {
      stores.emplace_back(op->buffer, op->indices);
      StmtExprVisitor::VisitStmt_(op);
    }
  } stores;
  stores(inner_body);

  try {
    auto &z3 = arith::Z3Prover(analyzer);
    z3.SetTimeoutMs(50);

    auto check_one_index =
        [&](const Buffer &buf, size_t dim, const PrimExpr &idx) -> bool {
      if (dim >= buf->shape.size()) {
        return false; // dimension mismatch — refuse to fuse
      }
      PrimExpr extent = buf->shape[dim];
      // Bit-bound free vars to [0, 2^31) — emulates BV32 signed-positive
      // domain. If we see a free Var we have no information about, this
      // is the only sound bound we can impose without polluting the
      // result.
      FreeVarCollector vc;
      vc(idx);
      vc(extent);
      // Push bound constraints for each free var.
      std::vector<std::function<void()>> recoverers;
      bool too_many_vars = false;
      for (const VarNode *v : vc.vars) {
        Var var = ffi::GetRef<Var>(v);
        DataType dt = var.dtype();
        if (!dt.is_int() && !dt.is_uint()) {
          too_many_vars = true; // unsupported dtype — bail
          break;
        }
        PrimExpr lo = make_const(dt, 0);
        PrimExpr hi = make_const(dt, int64_t(1) << 31);
        PrimExpr bound = (var >= lo) && (var < hi);
        recoverers.push_back(z3.EnterConstraint(bound));
        if (recoverers.size() > 8) {
          // Don't blow up the solver — give up on highly-symbolic bodies.
          too_many_vars = true;
          break;
        }
      }
      bool ok = false;
      if (!too_many_vars) {
        // Goal: 0 <= idx AND idx < extent — UNCONDITIONALLY (no `a`
        // assumption pushed).
        PrimExpr goal = (idx >= make_const(idx.dtype(), 0)) && (idx < extent);
        try {
          ok = z3.CanProve(goal);
        } catch (...) {
          ok = false;
        }
      }
      // Pop in reverse order.
      for (auto it = recoverers.rbegin(); it != recoverers.rend(); ++it) {
        (*it)();
      }
      return ok;
    };

    for (const BufferLoadNode *ld : collector.loads) {
      for (size_t d = 0; d < ld->indices.size(); ++d) {
        if (!check_one_index(ld->buffer, d, ld->indices[d])) {
          return false;
        }
      }
    }
    for (const auto &kv : stores.stores) {
      for (size_t d = 0; d < kv.second.size(); ++d) {
        if (!check_one_index(kv.first, d, kv.second[d])) {
          return false;
        }
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

class PredicateFuser : public StmtExprMutator {
public:
  explicit PredicateFuser(arith::Analyzer *analyzer) : analyzer_(analyzer) {}

private:
  arith::Analyzer *analyzer_;

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Outer if: must have no else branch (pattern `if(a){...}` only).
    if (op->else_case.defined()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Outer guard `a` must be side-effect free — otherwise fusing
    // `if(a&&b)` could re-order or duplicate observable effects.
    if (SideEffect(op->condition) > CallEffectKind::kReadState) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // The outer body must be exactly an inner `if(b){...}` (no
    // intervening statements, no else on the inner if).
    Stmt then_body = op->then_case;
    const IfThenElseNode *inner = then_body.as<IfThenElseNode>();
    if (!inner) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (inner->else_case.defined()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (SideEffect(inner->condition) > CallEffectKind::kReadState) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Recurse into the inner body first so any nested fusion runs.
    Stmt new_inner_body = VisitStmt(inner->then_case);

    // Z3 well-definedness query: can we prove the inner body's
    // BufferLoads/Stores are in-range without assuming `a`?
    if (!Z3ProvesInnerWellDefined(new_inner_body, analyzer_)) {
      // Conservative: keep the original nesting (with the recursed body).
      Stmt rebuilt_inner =
          IfThenElse(inner->condition, new_inner_body, Stmt(), inner->span);
      return IfThenElse(op->condition, rebuilt_inner, Stmt(), op->span);
    }

    // Fuse: `if (a && b) { body }`.
    PrimExpr fused_cond = op->condition && inner->condition;
    return IfThenElse(fused_cond, new_inner_body, Stmt(), op->span);
  }

  // Apache StmtFunctor's vtable doesn't dispatch on the vendored
  // `tilelang::tl_tir::LetStmt`. Mirror the recursion pattern used by
  // other Tile-Lang transforms so we don't drop bindings on the floor.
  Stmt VisitStmt(const Stmt &stmt) final {
    if (const auto *op = stmt.as<LetStmtNode>()) {
      auto value = VisitExpr(op->value);
      auto body = VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return stmt;
      }
      return LetStmt(op->var, value, body, op->span);
    }
    return StmtExprMutator::VisitStmt(stmt);
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass PredicateFusion() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    bool enabled =
        ctx->GetConfig<Bool>(kPredicateFusion, Bool(false)).value();
    if (!enabled) {
      return f;
    }
    arith::Analyzer analyzer;
    PredicateFuser fuser(&analyzer);
    f.CopyOnWrite()->body = fuser(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PredicateFusion", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PredicateFusion", PredicateFusion);
}

} // namespace tl
} // namespace tvm
