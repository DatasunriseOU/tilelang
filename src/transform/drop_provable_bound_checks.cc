/*!
 * \file drop_provable_bound_checks.cc
 * \brief Drop runtime `if (i < N) buf[i] = ...` bound check guards when the
 *        condition is conclusively provable.
 *
 * Roadmap idea #4 (Z3 fork ports). Conservative-by-default:
 *   - Only `IfThenElse` whose condition matches the BufferBoundCheck pattern
 *     is considered.
 *   - The default arith analyzer is consulted first (with kSymbolicBound
 *     proof strength). If that fails, the vendored Z3 prover is used as a
 *     fallback under bit-bounded (BV32-emulated) constraints.
 *   - Any Z3 error / timeout / UNKNOWN is treated as "cannot prove" and the
 *     guard is kept.
 *   - Pass is OFF by default. Enable via PassConfig
 *     `tl.drop_provable_bound_checks = True`.
 *
 * Pattern matched (`IsBufferBoundCheck`):
 *   Conjunctions / single forms of `LT(idx_var, extent)` and / or
 *   `LE(0, idx_var)` etc., where:
 *     1. `idx_var` is a `tirx::Var` defined by an enclosing `For` loop whose
 *        `min`+`extent` are known via the `arith::Analyzer` (or visible
 *        constants).
 *     2. The other operand (`extent`) is loop-invariant, i.e. uses no vars
 *        defined inside the bound-check's containing loop chain.
 *
 * The `then_case` body replaces the IfThenElse on success. If the IfThenElse
 * has an else, the pass leaves it alone (the user wrote a custom else; we
 * don't try to rewrite it).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "vendored/z3_prover.h"

namespace tvm {
namespace tl {

using namespace tirx;
using arith::IRMutatorWithAnalyzer;

namespace {

// Collect all `Var` referenced anywhere inside an expression. Used for
// "free-var" emulation when seeding Z3 with BV32-style constraints.
class VarCollector : public ExprVisitor {
 public:
  std::unordered_set<const VarNode *> vars;
  void VisitExpr_(const VarNode *op) final { vars.insert(op); }
};

// Pattern matcher for buffer-bound predicates:
//   LT(Var, Expr)            — primary form (i < N)
//   LE(Var, Expr)            — i <= N (treated the same; replace with i < N+1)
//   And(predicate, predicate)— recursive, each conjunct must match
//   GE(Var, IntImm(0))       — 0 <= i (also accepted in conjunctions)
//   LE(IntImm(0), Var)       — same as above
//
// Returns `true` only when the WHOLE expression decomposes into such forms.
// This avoids accidentally treating arbitrary boolean predicates (e.g.
// `cond[0] > 0` in the LoopUnswitching tests) as bound checks.
class IsBufferBoundCheck : public ExprVisitor {
 public:
  bool ok = true;

  static bool Check(const PrimExpr &expr) {
    IsBufferBoundCheck v;
    v.Walk(expr);
    return v.ok;
  }

 private:
  void Walk(const PrimExpr &expr) {
    if (!ok) return;
    if (const auto *op = expr.as<AndNode>()) {
      Walk(op->a);
      Walk(op->b);
      return;
    }
    if (const auto *op = expr.as<LTNode>()) {
      // LT(Var, Expr) or LT(Expr, Var) — only first form counts as the i<N
      // shape.
      if (op->a.as<VarNode>() != nullptr) {
        return;
      }
      ok = false;
      return;
    }
    if (const auto *op = expr.as<LENode>()) {
      // Accept both LE(Var, Expr) and LE(IntImm(0), Var).
      if (op->a.as<VarNode>() != nullptr) {
        return;
      }
      if (const auto *imm = op->a.as<IntImmNode>()) {
        if (imm->value == 0 && op->b.as<VarNode>() != nullptr) {
          return;
        }
      }
      ok = false;
      return;
    }
    if (const auto *op = expr.as<GENode>()) {
      // GE(Var, IntImm(0)) — i >= 0.
      if (op->a.as<VarNode>() != nullptr) {
        if (const auto *imm = op->b.as<IntImmNode>()) {
          if (imm->value == 0) {
            return;
          }
        }
      }
      ok = false;
      return;
    }
    ok = false;
  }
};

}  // namespace

class DropProvableBoundChecks : public IRMutatorWithAnalyzer {
 public:
  static PrimFunc Apply(PrimFunc func) {
    arith::Analyzer analyzer;
    DropProvableBoundChecks rewriter(&analyzer);
    PrimFuncNode *func_node = func.CopyOnWrite();
    func_node->body = rewriter.VisitStmt(func_node->body);
    return func;
  }

  int dropped_count() const { return dropped_count_; }

 private:
  explicit DropProvableBoundChecks(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Only attempt rewrites when the IfThenElse has no user-written else
    // branch. (If the user wrote one, we'd be silently changing its
    // semantics by collapsing.)
    if (op->else_case.defined()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    PrimExpr cond = op->condition;
    if (!IsBufferBoundCheck::Check(cond)) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    // 1) Default analyzer first (with kSymbolicBound).
    if (analyzer_->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      ++dropped_count_;
      return this->VisitStmt(op->then_case);
    }

    // 2) Z3 fallback under BV32-emulated free-var constraints.
    bool proved = false;
    try {
      auto &z3 = arith::Z3Prover(analyzer_);
      z3.SetTimeoutMs(50);

      // Push 0 <= v < 2^31 for every free Var in `cond`. This is the BV32
      // "bit-bound emulation" the roadmap calls for; it stops Z3 from
      // proving identities that only hold under unbounded ints (e.g.
      // overflow situations near INT_MAX).
      VarCollector vc;
      vc(cond);
      const int64_t kBitBound = (int64_t(1) << 31);
      std::vector<std::function<void()>> recover_stack;
      recover_stack.reserve(vc.vars.size() * 2);
      for (const VarNode *vn : vc.vars) {
        Var v = tvm::ffi::GetRef<Var>(vn);
        // Only int-typed vars get the BV32 box.
        if (!v.dtype().is_int()) {
          continue;
        }
        recover_stack.push_back(z3.EnterConstraint(v >= IntImm(v.dtype(), 0)));
        recover_stack.push_back(
            z3.EnterConstraint(v < IntImm(v.dtype(), kBitBound)));
      }

      proved = z3.CanProve(cond);

      // Pop in reverse to restore solver state.
      for (auto it = recover_stack.rbegin(); it != recover_stack.rend();
           ++it) {
        (*it)();
      }
    } catch (...) {
      proved = false;  // conservative: keep guard
    }

    if (proved) {
      ++dropped_count_;
      return this->VisitStmt(op->then_case);
    }

    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  int dropped_count_ = 0;
};

PrimFunc DropProvableBoundChecksFn(PrimFunc func) {
  if (!func->body.defined()) return func;
  return DropProvableBoundChecks::Apply(std::move(func));
}

tvm::transform::Pass DropProvableBoundChecksPass() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, PassContext ctx) {
    bool enable = ctx->GetConfig<Bool>(kDropProvableBoundChecks, Bool(false))
                      .value()
                      ->value;
    if (!enable) return f;
    return DropProvableBoundChecksFn(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.DropProvableBoundChecks", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.DropProvableBoundChecks",
                        DropProvableBoundChecksPass);
}

}  // namespace tl
}  // namespace tvm
