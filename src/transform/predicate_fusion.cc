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
#include "vendored/z3_constraint_scope.h"
#include "vendored/z3_prover.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

namespace {

// Collect all `Var`s referenced in an expression. Used to push per-var
// bit-bound constraints into the Z3 context before issuing the
// well-definedness query.
//
// Note: this collects ALL vars, including loop-bound and let-bound vars
// that may already have tighter ranges from the analyzer. The BV bounds
// pushed by `Z3ProvesIndexInRange` are layered ON TOP of the analyzer's
// existing constraints, so Z3 intersects them — a redundant wider bound
// on a var that already has [0, N) from a For-loop is harmless. The
// alternative (filtering out bound vars) would require walking the
// enclosing scope chain, which adds complexity for no correctness gain.
// The cost is purely solver effort: BV-bounding a var the analyzer
// already constrained is a redundant assertion that Z3 handles trivially.
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

// Collect every BufferLoad reachable from a (sub)expression. Used by
// the b-condition probe to enumerate any direct loads inside the inner
// `if`'s condition — those loads are evaluated even when `!a` after
// the fusion to `if(a && b)` — though C++/MSL short-circuit at runtime,
// the lowered TIR may evaluate both sides depending on the codegen.
class ExprBufferLoadCollector : public ExprVisitor {
 public:
  std::vector<const BufferLoadNode *> loads;
  void VisitExpr_(const BufferLoadNode *op) final {
    loads.push_back(op);
    ExprVisitor::VisitExpr_(op);
  }
};

// Probe a single (buf, dim, idx) tuple: prove `0 <= idx < buf.shape[dim]`
// UNCONDITIONALLY (no outer-guard assumptions pushed). Bit-bounds free
// vars to a dtype-aware range (see `BVBoundsForDtype`). Returns false
// on bailout / timeout / >8 free vars.
//
// Hoisted from the previous lambda inside `Z3ProvesInnerWellDefined` so
// the b-condition probe (fix-B3) can reuse it.
static bool Z3ProvesIndexInRange(const Buffer &buf, size_t dim,
                                 const PrimExpr &idx,
                                 ::tilelang::tlz3::Z3Prover &z3) {
  if (dim >= buf->shape.size()) {
    return false; // dimension mismatch — refuse to fuse
  }
  PrimExpr extent = buf->shape[dim];
  FreeVarCollector vc;
  vc(idx);
  vc(extent);
  std::vector<::tilelang::tlz3::ConstraintScope> scopes;
  bool too_many_vars = false;
  for (const VarNode *v : vc.vars) {
    Var var = ffi::GetRef<Var>(v);
    DataType dt = var.dtype();
    if (!dt.is_int() && !dt.is_uint()) {
      too_many_vars = true; // unsupported dtype — bail
      break;
    }
    // CPPMEGA fix-B4 (idea712): dtype-aware BV bounds. The previous flat
    // [0, 2^31) was unsound for signed vars that could legitimately be
    // negative. If the dtype range cannot be represented exactly as an
    // int64 half-open interval, refuse to fuse.
    auto bounds = ::tilelang::tlz3::BVBoundsForDtype(dt);
    if (!bounds.has_value()) {
      too_many_vars = true;
      break;
    }
    auto [lo64, hi64] = *bounds;
    PrimExpr lo = make_const(dt, lo64);
    PrimExpr hi = make_const(dt, hi64);
    PrimExpr bound = (var >= lo) && (var < hi);
    scopes.emplace_back(z3, bound);
    if (scopes.size() > 8) {
      too_many_vars = true; // don't blow up the solver
      break;
    }
  }
  if (too_many_vars) return false;
  PrimExpr goal = (idx >= make_const(idx.dtype(), 0)) && (idx < extent);
  try {
    return z3.CanProve(goal);
  } catch (...) {
    return false;
  }
}

// CPPMEGA fix-B3 (idea712): prove every BufferLoad inside the inner
// `if` *condition* itself is in-range UNCONDITIONALLY. Without this
// check, a body like
//
//     if (a) { if (buf[i] > 0) { ... } }
//
// would fuse to `if (a && buf[i] > 0) { ... }`. While C++/MSL `&&`
// short-circuits at runtime, the lowered TIR fold may evaluate both
// sides — and if `buf[i]` is only safe under `!a`, that load OOBs.
//
// Returns true iff every BufferLoad in `cond` is provably in-range
// regardless of `a`'s value.
bool Z3ProvesConditionLoadsWellDefined(const PrimExpr &cond,
                                       arith::Analyzer *analyzer) {
  ExprBufferLoadCollector cl;
  cl(cond);
  if (cl.loads.empty()) {
    return true; // no buffer loads in `b` → trivially safe
  }
  // CPPMEGA z3-final per-pass gate: TILELANG_DISABLE_Z3_PREDICATE_FUSION
  // bypasses the b-condition load probe (idea #7). Conservative default —
  // refuse to fuse if Z3 is disabled, since fusion soundness depends on
  // proving every load in `b` is well-defined when `!a`.
  if (!::tilelang::tlz3::Z3PassGate::IsEnabled("PREDICATE_FUSION")) {
    return false;
  }
  try {
    auto &z3 = arith::Z3Prover(analyzer);
    z3.SetTimeoutMs(50);
    for (const BufferLoadNode *ld : cl.loads) {
      // CPPMEGA idea712 round-final fix-NG: degenerate-load null guard.
      // A BufferLoad with no indices or an undefined buffer is malformed
      // for our proof: there's no index to range-check, and the buffer
      // shape lookup in Z3ProvesIndexInRange would dereference a null.
      // Conservatively bail (return false → predicate-fusion is skipped).
      if (ld == nullptr || ld->indices.empty() || !ld->buffer.defined()) {
        return false;
      }
      for (size_t d = 0; d < ld->indices.size(); ++d) {
        if (!Z3ProvesIndexInRange(ld->buffer, d, ld->indices[d], z3)) {
          return false;
        }
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

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

  // CPPMEGA z3-final per-pass gate: TILELANG_DISABLE_Z3_PREDICATE_FUSION
  // bypasses the inner-body well-definedness proof (idea #7).
  if (!::tilelang::tlz3::Z3PassGate::IsEnabled("PREDICATE_FUSION")) {
    return false;
  }
  try {
    auto &z3 = arith::Z3Prover(analyzer);
    z3.SetTimeoutMs(50);

    for (const BufferLoadNode *ld : collector.loads) {
      // CPPMEGA idea712 round-final fix-NG: same null guard as the
      // condition-load probe. Treat malformed BufferLoads conservatively.
      if (ld == nullptr || ld->indices.empty() || !ld->buffer.defined()) {
        return false;
      }
      for (size_t d = 0; d < ld->indices.size(); ++d) {
        if (!Z3ProvesIndexInRange(ld->buffer, d, ld->indices[d], z3)) {
          return false;
        }
      }
    }
    for (const auto &kv : stores.stores) {
      if (!kv.first.defined() || kv.second.empty()) {
        return false;
      }
      for (size_t d = 0; d < kv.second.size(); ++d) {
        if (!Z3ProvesIndexInRange(kv.first, d, kv.second[d], z3)) {
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

    // CPPMEGA fix-B3 (idea712): the inner CONDITION `b` itself may
    // contain BufferLoads (e.g. `b == buf[i] > 0`). After fusion to
    // `if (a && b)`, the lowered TIR may evaluate both sides regardless
    // of short-circuit semantics in the surface language. If any load
    // in `b` is only safe under `!a`, the fusion would OOB. Require Z3
    // to prove every load index in `b` is in-range UNCONDITIONALLY too.
    if (!Z3ProvesConditionLoadsWellDefined(inner->condition, analyzer_)) {
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
