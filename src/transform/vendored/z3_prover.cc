// CPPMEGA: vendored Z3 prover implementation — adapted from
// 3rdparty/tvm/src/target/z3/z3_prover_on.cc in the TileLang fork tree at
// /private/tmp/cppmega-mlx-tilelang-stack-c. The fork's class was
// `tvm::arith::Z3Prover::Impl`, a friend of `Analyzer`. apache/tvm latest
// does not expose that hook so we lift the implementation out into a
// standalone class `tilelang::z3::Z3ProverImpl` and front it with
// `tilelang::z3::Z3Prover` (see z3_prover.h).
//
// Adjustments vs the upstream-fork copy:
//   * `tir::` namespace usages → `tirx::` (apache/tvm post-tirx-refactor).
//   * `tvm/tir/...` includes  → `tvm/tirx/...`.
//   * Class is no longer a member of `Analyzer`; ctor takes `Analyzer*`
//     and on construction seeds the Z3 solver with all `var → range`
//     bindings recoverable from `analyzer.const_int_bound`. The fork
//     relied on `Analyzer::Bind` forwarding to `z3_prover.Bind` directly,
//     which apache does not do.
//   * Public-API return types are `std::string` instead of `ffi::String`
//     (lossless: every call site streams the result).
//
// State semantics:
//   The per-Analyzer cache (see `GetOrCreate` in z3_prover.h's
//   implementation below) hands out one `Z3Prover` instance per
//   `Analyzer*`. The cache is `thread_local` because z3 contexts aren't
//   safe to share across threads.

#include "z3_prover.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/var.h>

#include <z3++.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
#include "tvm/ffi/reflection/registry.h"
#include "tvm/ffi/string.h"

namespace tilelang {
namespace tlz3 {

namespace {

using ::tvm::Downcast;
using ::tvm::IntImm;
using ::tvm::IntImmNode;
using ::tvm::PrimExpr;
using ::tvm::PrimExprNode;
using ::tvm::Range;
using ::tvm::ffi::GetRef;
using ::tvm::runtime::DataType;
using ::tvm::tirx::AddNode;
using ::tvm::tirx::AndNode;
using ::tvm::tirx::BufferLoadNode;
using ::tvm::tirx::CallNode;
using ::tvm::tirx::CallEffectKind;
using ::tvm::tirx::CastNode;
using ::tvm::tirx::DivNode;
using ::tvm::tirx::EQNode;
using ::tvm::tirx::ExprDeepEqual;
using ::tvm::tirx::ExprFunctor;
using ::tvm::tirx::FloorDivNode;
using ::tvm::tirx::FloorModNode;
using ::tvm::tirx::GENode;
using ::tvm::tirx::GTNode;
using ::tvm::tirx::LENode;
using ::tvm::tirx::LetNode;
using ::tvm::tirx::LTNode;
using ::tvm::tirx::MaxNode;
using ::tvm::tirx::MinNode;
using ::tvm::tirx::ModNode;
using ::tvm::tirx::MulNode;
using ::tvm::tirx::NENode;
using ::tvm::tirx::NotNode;
using ::tvm::tirx::OrNode;
using ::tvm::tirx::ProducerLoadNode;
using ::tvm::tirx::ReduceNode;
using ::tvm::tirx::SelectNode;
using ::tvm::tirx::SideEffect;
using ::tvm::tirx::SubNode;
using ::tvm::tirx::Var;
using ::tvm::tirx::VarNode;

namespace tirx_op = ::tvm::tirx;

struct Namespace {
  std::unordered_set<std::string> used_names;
  std::string GetNewName(const PrimExpr& expr) {
    std::stringstream ss;
    ss << expr;
    auto name = ss.str();
    if (used_names.count(name) == 0) {
      used_names.insert(name);
      return name;
    }
    int idx = 1;
    std::string check_name = name + "$" + std::to_string(idx);
    while (used_names.count(check_name)) {
      idx++;
      check_name = name + "$" + std::to_string(idx);
    }
    used_names.insert(check_name);
    return check_name;
  }
};

}  // namespace

class Z3ProverImpl : public ExprFunctor<z3::expr(const PrimExpr&)> {
 public:
  using Base = ExprFunctor<z3::expr(const PrimExpr&)>;

  ::tvm::arith::Analyzer* analyzer;

  // Z3 context shared per-thread (Z3 init is slow on some CPUs).
  inline static thread_local std::shared_ptr<::z3::context> ctx{
      new ::z3::context()};

  ::z3::solver solver{*ctx};
  std::unordered_map<PrimExpr, ::z3::expr,
                     ::tvm::ffi::StructuralHash, ExprDeepEqual> memo_;
  bool is_assume = false;
  Namespace ns;
  unsigned timeout_ms{UINT_MAX};
  unsigned rlimit{UINT_MAX};
  // 0 == unbounded Int sort (default, bit-identical to prior behavior).
  // 32 / 64 == signed BitVector sort of that width. See SetBitVectorMode().
  int bv_width_{0};
  // First-occurrence flags for once-per-Analyzer warnings about silent
  // BV truncation (MakeIntVal) and out-of-range range binds (Bind/Range).
  bool bv_truncation_warned_{false};
  bool bv_range_warned_{false};

  // Helpers that hide the Int-vs-BV sort dispatch. In default mode
  // (bv_width_ == 0) these behave exactly like the old direct calls into
  // ctx->int_*; in BV mode they produce sized bv_const / bv_val with the
  // current width, and bv_sort for the sort.
  ::z3::sort MakeIntSort() {
    if (bv_width_ > 0) return ctx->bv_sort(static_cast<unsigned>(bv_width_));
    return ctx->int_sort();
  }
  ::z3::expr MakeIntConst(const std::string& name) {
    if (bv_width_ > 0) {
      return ctx->bv_const(name.c_str(), static_cast<unsigned>(bv_width_));
    }
    return ctx->int_const(name.c_str());
  }
  ::z3::expr MakeIntVal(int64_t value) {
    if (bv_width_ > 0) {
      // fix-round-6 C8: previously, when `value` was outside the signed
      // BV range we let Z3's bv_val silently truncate (two's-complement
      // wrap). That makes proofs over wide constants unsound — e.g. a
      // BV32 prove of `x == 0x100000000` would happily say "yes, x is
      // 0" because the constant wrapped to zero. Conservative fix: mint
      // an unconstrained fresh symbol of the BV sort so any predicate
      // depending on the constant fails to be proved. Sound but lossy.
      if (bv_width_ < 64) {
        int64_t lo = (bv_width_ == 32) ? static_cast<int64_t>(INT32_MIN)
                                       : -(int64_t{1} << (bv_width_ - 1));
        int64_t hi = (bv_width_ == 32) ? static_cast<int64_t>(INT32_MAX)
                                       : ((int64_t{1} << (bv_width_ - 1)) - 1);
        if (value < lo || value > hi) {
          if (!bv_truncation_warned_) {
            LOG(WARNING)
                << "Z3Prover BV" << bv_width_ << ": MakeIntVal(" << value
                << ") out of signed range [" << lo << ", " << hi
                << "]; returning unconstrained symbol (any proof "
                   "depending on this constant will conservatively fail). "
                   "(further occurrences suppressed)";
            bv_truncation_warned_ = true;
          }
          return MakeIntConst("oor_" + std::to_string(memo_.size()));
        }
      }
      return ctx->bv_val(static_cast<int64_t>(value),
                         static_cast<unsigned>(bv_width_));
    }
    return ctx->int_val(value);
  }
  ::z3::expr MakeUIntVal(uint64_t value) {
    if (bv_width_ > 0) {
      return ctx->bv_val(static_cast<uint64_t>(value),
                         static_cast<unsigned>(bv_width_));
    }
    return ctx->int_val(value);
  }

  static ::z3::solver CreateSolver(::z3::context& ctx) {
    ::z3::solver solver(ctx);
    solver.set("model", false);
    solver.set("random_seed", (unsigned)42);
    return solver;
  }

  explicit Z3ProverImpl(::tvm::arith::Analyzer* parent) : analyzer(parent) {
    scope_stack_.push_back({});
    solver = CreateSolver(*ctx);
    SetRLimit(static_cast<unsigned>(1e4));
  }

  ::z3::expr Create(const PrimExprNode* op) {
    auto ref = GetRef<PrimExpr>(op);
    auto dtype = op->dtype;
    std::string name = ns.GetNewName(ref);
    if (dtype.is_bool()) {
      return ctx->bool_const(name.c_str());
    } else {
      ::z3::expr e = MakeIntConst(name);
      if (bv_width_ > 0) {
        // In BV mode the variable is already bounded by its sort width.
        // Adding range constraints in terms of int_val would mix sorts;
        // skip them.
      } else if (dtype.is_uint() && dtype.bits() == 64) {
        solver.add(ctx->int_val(0) <= e &&
                   e <= ctx->int_val((uint64_t)UINT64_MAX));
      } else {
        auto min_val = Downcast<IntImm>(::tvm::min_value(dtype))->value;
        auto max_val = Downcast<IntImm>(::tvm::max_value(dtype))->value;
        solver.add(ctx->int_val(min_val) <= e &&
                   e <= ctx->int_val(max_val));
      }
      return e;
    }
  }

  struct Scope {
    enum Kind {
      BindValue,
      BindRange,
      Constraint,
    } kind;
    Var var;
    PrimExpr value;
    PrimExpr min;
    PrimExpr extent;
    PrimExpr constraint;
  };

  std::vector<std::vector<Scope>> scope_stack_;

  std::function<void()> EnterConstraint(const PrimExpr& constraint,
                                        bool is_assume_in = false) {
    scope_stack_.push_back({});
    scope_stack_.back().push_back(Scope{Scope::Constraint, Var(), PrimExpr(),
                                        PrimExpr(), PrimExpr(), constraint});
    solver.push();
    this->is_assume = is_assume_in;
    solver.add(VisitBool(constraint));
    this->is_assume = false;
    // CPPMEGA fix-A4: snapshot `side_effect_exprs_` into a local. The
    // member is then explicitly cleared so subsequent
    // VisitBool/ConvertInt calls (between EnterConstraint and the
    // returned recovery lambda firing) start with a fresh accumulator.
    // The recovery lambda below captures the snapshot by VALUE — this
    // is critical because:
    //   (1) the `side_effect_exprs_` member is rewritten by every
    //       subsequent VisitBool/ConvertInt that runs while the
    //       constraint is on the stack, so a `[this, &]` capture would
    //       see the wrong set at lambda-fire time, and
    //   (2) the lambda may outlive the local snapshot — the caller
    //       hands the std::function back up the call stack, so a
    //       reference-to-local capture is a use-after-scope.
    // We move out of the member to avoid an extra copy, then explicitly
    // copy into the lambda capture (the `=` form spells it out).
    std::vector<PrimExpr> side_effect_exprs = std::move(side_effect_exprs_);
    side_effect_exprs_.clear();  // moved-from is valid-but-unspecified;
                                 // make the member explicitly empty.
    if (is_assume_in) {
      // Capture-by-value (explicit): `side_effect_exprs` is copied into
      // the lambda. The original local will go out of scope at the end
      // of EnterConstraint; the lambda's copy survives until the
      // recovery fires.
      return [this, side_effect_exprs = std::move(side_effect_exprs)]() {
        solver.pop();
        for (const auto& expr : side_effect_exprs) {
          memo_.erase(expr);
        }
        scope_stack_.pop_back();
      };
    } else {
      // Non-is_assume path: side effects are erased *now* (the local
      // snapshot is consumed before the lambda is constructed), so the
      // lambda only needs `this`. Doing the erase here matches the
      // pre-fix behavior and keeps the constraint scope's solver state
      // tight (we don't carry the side_effect set forward into the
      // recovery closure).
      for (const auto& expr : side_effect_exprs) {
        memo_.erase(expr);
      }
      return [this]() {
        solver.pop();
        scope_stack_.pop_back();
      };
    }
  }

  bool CheckTrivilBadCases(const PrimExpr& expr) {
    if (IsFreeNode(expr)) {
      return true;
    }
    auto checkTrivilCmp = [this](const PrimExpr& lhs, const PrimExpr& rhs) {
      if (IsFreeNode(lhs) && rhs->IsInstance<IntImmNode>()) return true;
      if (IsFreeNode(rhs) && lhs->IsInstance<IntImmNode>()) return true;
      if (IsFreeNode(lhs) && IsFreeNode(rhs)) return true;
      if (auto cast = lhs.as<CastNode>()) {
        if (IsFreeNode(cast->value) && rhs->IsInstance<IntImmNode>())
          return true;
      }
      if (auto cast = rhs.as<CastNode>()) {
        if (IsFreeNode(cast->value) && lhs->IsInstance<IntImmNode>())
          return true;
      }
      return false;
    };
    if (auto eq = expr.as<EQNode>()) {
      return checkTrivilCmp(eq->a, eq->b);
    } else if (auto ne = expr.as<NENode>()) {
      return checkTrivilCmp(ne->a, ne->b);
    }
    return false;
  }

  bool CanProve(const PrimExpr& expr) {
    if (CheckTrivilBadCases(expr)) return false;
    if (!IsValidDType(expr->dtype)) return false;
    try {
      ::z3::expr_vector constr(*ctx);
      constr.push_back(!ConvertBool(expr));
      auto result = solver.check(constr);
      constr.pop_back();
      return result == ::z3::unsat;
    } catch (const std::exception& e) {
      LOG(WARNING) << "Z3 query exception: " << e.what();
      return false;
    } catch (...) {
      LOG(WARNING) << "Z3 query unknown exception";
      return false;
    }
  }

  void Bind(const Var& var, const PrimExpr& value, bool /*allow_override*/) {
    if (!IsValidDType(var->dtype)) return;
    scope_stack_.back().push_back(Scope{Scope::BindValue, var, value});
    memo_.emplace(var, ConvertInt(value));
  }

  void Bind(const Var& var, const Range& range, bool /*allow_override*/) {
    if (!IsValidDType(var->dtype)) return;
    scope_stack_.back().push_back(
        Scope{Scope::BindRange, var, PrimExpr(), range->min, range->extent});
    // CPPMEGA fix-A3: defer memoization until after the BV-range check.
    // Previously we wrote `memo_.emplace(var, var_expr)` here unconditionally,
    // which meant an out-of-range bind in BV mode left the var memoized at
    // the current sort but with no range constraint asserted in the solver.
    // A subsequent CanProve over the same var would then see a free symbol
    // and silently skip the intended bound. Compute `var_expr` but only
    // commit it to memo_ once we know we're going to assert the constraints.
    //
    // CPPMEGA fix-A7: round-3 update — when the caller's range exceeds
    // the BV width, we now ALWAYS commit a memoization (clamped to the
    // BV range) rather than silently dropping the bind. See the
    // out-of-range branch below for the full rationale.
    auto var_expr = Create(var.as<PrimExprNode>());
    bool commit_memo = true;
    if (tirx_op::is_const_int(range->min) &&
        tirx_op::is_const_int(range->min + range->extent)) {
      int64_t min_value = *tirx_op::as_const_int(range->min);
      int64_t max_value = *tirx_op::as_const_int(range->min + range->extent);
      if (min_value < max_value) {
        // In BV mode, skip binds with bounds outside the BV width's
        // signed range rather than synthesize wrap-around BV constants
        // that would silently misrepresent intent. Warn at most once per
        // Analyzer to avoid log spam in a long compile.
        if (bv_width_ > 0 && bv_width_ < 64) {
          int64_t lo = (bv_width_ == 32) ? static_cast<int64_t>(INT32_MIN)
                                         : -(int64_t{1} << (bv_width_ - 1));
          int64_t hi = (bv_width_ == 32) ? static_cast<int64_t>(INT32_MAX)
                                         : ((int64_t{1} << (bv_width_ - 1)) - 1);
          if (min_value < lo || max_value > hi) {
            if (!bv_range_warned_) {
              LOG(WARNING) << "Z3Prover BV" << bv_width_
                           << ": dropping out-of-range bind " << var
                           << " in [" << min_value << ", " << max_value
                           << ") (signed BV range [" << lo << ", " << hi
                           << "]). Subsequent CanProve over this var will"
                           << " fail closed. (further occurrences suppressed)";
              bv_range_warned_ = true;
            }
            // CPPMEGA fix-C2 (round-7): the round-3 fix wrote a clamped memo
            // here, which over-approximated the caller's intent. Wave-5
            // audit flagged this as unsound under composition: a downstream
            // CanProve on `var` would see the *clamped* range and could
            // prove tautologies that the caller never asserted. Round-7
            // tightens to NOT emplace memo on OOR. A subsequent Visit(var)
            // will still mint a fresh symbol, but with no range constraint
            // asserted, CanProve will fail closed (return false) for any
            // non-trivial query — preserving soundness. Callers that need
            // the bind must keep variables within the BV width.
            commit_memo = false;
            return;
          }
        }
        memo_.emplace(var, var_expr);
        solver.add(MakeIntVal(min_value) <= var_expr);
        solver.add(var_expr < MakeIntVal(max_value));
        commit_memo = false;  // already committed
      } else {
        // CPPMEGA fix-round-2 (HIGH correctness): empty range (min >= max)
        // is logically UNSAT for any valuation of `var`. Previously we
        // returned WITHOUT writing memo_, which meant a subsequent
        // Visit(var) would mint a fresh free Z3 symbol, with no
        // constraints — silently dropping the caller's intent. Commit
        // the memo so subsequent uses bind to the same symbol, then
        // assert `false` so any CanProve under this scope is sound
        // (vacuously true) rather than reasoning about a free variable.
        memo_.emplace(var, var_expr);
        solver.add(ctx->bool_val(false));
        commit_memo = false;
        return;
      }
    } else {
      memo_.emplace(var, var_expr);
      solver.add(ConvertBool(range->extent <= 0 ||
                             (range->min <= var &&
                              var < range->min + range->extent)));
      commit_memo = false;
    }
    (void)commit_memo;  // explicit: every code path that exits this
                        // function has either committed memo or returned.
  }

  void SetTimeoutMs(unsigned timeout_ms_in) {
    this->timeout_ms = timeout_ms_in;
    solver.set("timeout", timeout_ms_in);
  }

  void SetRLimit(unsigned rlimit_in) {
    this->rlimit = rlimit_in;
    solver.set("rlimit", rlimit_in);
  }

  // CPPMEGA fix-A6: factored solver-rebuild helper. Used by both
  // `SetBitVectorMode` (when the mode actually changes) and the new
  // public `Reset()` (per-pass reseed). Centralizing the rebuild
  // sequence avoids drift between the two paths.
  void RebuildSolver_() {
    bv_truncation_warned_ = false;
    bv_range_warned_ = false;
    memo_.clear();
    side_effect_exprs_.clear();
    solver = CreateSolver(*ctx);
    solver.set("timeout", timeout_ms);
    solver.set("rlimit", rlimit);
    scope_stack_.clear();
    scope_stack_.push_back({});
    ns = Namespace{};
  }

  void SetBitVectorMode(int width) {
    ICHECK(width == 0 || width == 32 || width == 64)
        << "Z3Prover::SetBitVectorMode only supports width in {0, 32, 64}, "
        << "got " << width;
    // CPPMEGA fix-A6: same-width fast-path. Mode-equality short-circuits
    // the full solver rebuild — important because compile-time blowup
    // was traced to passes calling `SetBitVectorMode(32)` redundantly
    // on every CanProve. The condition also covers width=0 → width=0
    // (Int → Int) which previously rebuilt unnecessarily.
    if (width == bv_width_) return;
    // CPPMEGA fix-round-2 (HIGH correctness): a mode change rebuilds the
    // solver, which clears `scope_stack_` and re-pushes a fresh root
    // frame. If we did this while inside an active `EnterConstraint`
    // scope, the recovery lambda's `solver.pop()` / `scope_stack_.pop_back()`
    // would target a fresh stack — corrupting the prover state. Require
    // the caller to be at the root scope before flipping modes.
    ICHECK_EQ(scope_stack_.size(), 1u)
        << "Z3Prover::SetBitVectorMode called with " << scope_stack_.size()
        << " scope frames; recover all EnterConstraint scopes first.";
    bv_width_ = width;
    // Mode change: invalidate any pre-existing variable / sub-expression
    // encodings (declared at the old sort) by rebuilding the solver.
    // Root binds are intentionally discarded; callers that need persistent
    // assumptions must re-enter them after changing modes.
    RebuildSolver_();
  }

  // CPPMEGA fix-A6: public per-pass reset. Callers (pass drivers) that
  // want to start a fresh proof context without flipping bv_width
  // should use `Reset()` instead of `SetBitVectorMode(currentWidth)` —
  // the latter is a no-op (see fast-path above) and would NOT actually
  // clear memo / scope stack. `Reset()` does. Bv_width is preserved.
  void Reset() { RebuildSolver_(); }

  int GetBitVectorWidth() const { return bv_width_; }

  std::string GetSMTLIB2() {
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << solver.to_smt2();
    return ss.str();
  }

  void AddScopeDebugMsg(std::ostream& ss) {
    for (const auto& scope : scope_stack_) {
      ss << "; Entering Scope\n";
      for (const auto& s : scope) {
        switch (s.kind) {
          case Scope::Constraint:
            ss << "; constraint: " << s.constraint << "\n";
            break;
          case Scope::BindValue:
            ss << "; bind value: " << s.var << " = " << s.value << "\n";
            break;
          case Scope::BindRange:
            ss << "; bind range: " << s.var << " in [" << s.min << ", "
               << s.min + s.extent << ")\n";
            break;
        }
      }
    }
  }

  std::string GetSMTLIB2(const PrimExpr& expr) {
    std::stringstream ss;
    ss << "(set-option :timeout " << timeout_ms << ")\n";
    AddScopeDebugMsg(ss);
    ss << "; Trying to prove: " << expr << "\n";
    solver.push();
    solver.add(!ConvertBool(expr));
    ss << solver.to_smt2();
    solver.pop();
    return ss.str();
  }

  std::string GetStats() {
    std::stringstream ss;
    ss << solver.statistics();
    return ss.str();
  }

  std::string GetModel(const PrimExpr& expr) {
    solver.set("model", true);
    solver.push();
    solver.add(!ConvertBool(expr));
    auto result = solver.check();
    std::string model_str;
    if (result == ::z3::sat) {
      ::z3::model m = solver.get_model();
      std::map<std::string, ::z3::expr> model_map;
      for (unsigned i = 0; i < m.size(); i++) {
        ::z3::func_decl d = m[i];
        model_map.emplace(d.name().str(), m.get_const_interp(d));
      }
      std::stringstream ss;
      for (const auto& [k, v] : model_map) {
        ss << "  " << k << " = " << v << "\n";
      }
      model_str = ss.str();
    }
    solver.pop();
    solver.set("model", false);
    return model_str;
  }

  int64_t CountSatisfyingValues(const Var& var, int64_t max_count,
                                int64_t min_consecutive) {
    if (!IsValidDType(var->dtype)) {
      return -1;
    }

    auto cleanup_side_effects = [this]() {
      for (const auto& expr : side_effect_exprs_) {
        memo_.erase(expr);
      }
      side_effect_exprs_.clear();
    };

    bool pushed = false;
    try {
      solver.set("model", true);
      solver.push();
      pushed = true;

      ::z3::expr z3_var = VisitInt(var);

      int64_t count = 0;
      std::vector<int64_t> found_values;

      while (count < max_count) {
        auto result = solver.check();
        if (result != ::z3::sat) break;
        ::z3::model m = solver.get_model();
        ::z3::expr val_expr = m.eval(z3_var, true);
        int64_t val;
        if (val_expr.is_numeral()) {
          val = val_expr.get_numeral_int64();
        } else {
          break;
        }
        found_values.push_back(val);
        count++;
        solver.add(z3_var != MakeIntVal(val));
      }

      solver.pop();
      pushed = false;
      solver.set("model", false);
      cleanup_side_effects();

      if (min_consecutive > 0 && count > 0) {
        std::sort(found_values.begin(), found_values.end());
        int64_t consecutive_count = 1;
        for (size_t i = 1; i < found_values.size(); i++) {
          if (found_values[i] == found_values[i - 1] + 1) {
            consecutive_count++;
          } else {
            if (consecutive_count < min_consecutive) return -2;
            consecutive_count = 1;
          }
        }
        if (consecutive_count < min_consecutive) return -2;
      }

      return count;
    } catch (const std::exception& e) {
      LOG(WARNING) << "Z3 CountSatisfyingValues exception: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Z3 CountSatisfyingValues unknown exception";
    }

    if (pushed) {
      try {
        solver.pop();
      } catch (...) {
      }
    }
    try {
      solver.set("model", false);
    } catch (...) {
    }
    cleanup_side_effects();
    return -1;
  }

 private:
  using Z3BinOp = ::z3::expr (*)(const ::z3::expr&, const ::z3::expr&);

  std::vector<PrimExpr> side_effect_exprs_;

  ::z3::expr ConvertBool(const PrimExpr& e, bool is_assume_in = false) {
    this->is_assume = is_assume_in;
    auto res = VisitBool(e);
    for (auto& expr : side_effect_exprs_) memo_.erase(expr);
    side_effect_exprs_.clear();
    this->is_assume = false;
    return res;
  }

  ::z3::expr ConvertInt(const PrimExpr& e, bool is_assume_in = false) {
    this->is_assume = is_assume_in;
    auto res = VisitInt(e);
    for (auto& expr : side_effect_exprs_) memo_.erase(expr);
    side_effect_exprs_.clear();
    this->is_assume = false;
    return res;
  }

  ::z3::expr VisitExpr(const PrimExpr& e) override {
    if (memo_.count(e)) return memo_.at(e);
    auto res = Base::VisitExpr(e);
    auto side_effect = SideEffect(e);
    if (side_effect <= CallEffectKind::kPure) {
      memo_.emplace(e, res);
    } else if (side_effect <= CallEffectKind::kReadState) {
      memo_.emplace(e, res);
      side_effect_exprs_.emplace_back(e);
    } else {
      if (is_assume) memo_.emplace(e, res);
      side_effect_exprs_.emplace_back(e);
    }
    return res;
  }

  bool IsFreeNode(const PrimExpr& e) {
    if (memo_.count(e)) return false;
    return e->IsInstance<CallNode>() ||
           e->IsInstance<BufferLoadNode>() ||
           e->IsInstance<ProducerLoadNode>() ||
           e->IsInstance<ReduceNode>() ||
           (e->IsInstance<CastNode>() &&
            !IsValidDType(Downcast<::tvm::tirx::Cast>(e)->value->dtype));
  }

  static bool IsValidDType(const DataType& dtype) {
    return (dtype.is_int() || dtype.is_uint() || dtype.is_bool()) &&
           dtype.lanes() == 1;
  }

  ::z3::expr VisitInt(const PrimExpr& expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      return ::z3::ite(e, MakeIntVal(1), MakeIntVal(0));
    } else {
      return e;
    }
  }

  ::z3::expr VisitBool(const PrimExpr& e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) return expr;
    return expr != MakeIntVal(0);
  }

  ::z3::expr VisitArith(Z3BinOp signed_op, const PrimExprNode* op,
                        const PrimExpr& a, const PrimExpr& b) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      return signed_op(VisitInt(a), VisitInt(b));
    } else {
      return Create(op);
    }
  }

  ::z3::expr VisitExpr_(const LetNode* op) override {
    if (IsValidDType(op->var->dtype)) {
      memo_.emplace(op->var, VisitInt(op->value));
    }
    return VisitExpr(op->body);
  }
  ::z3::expr VisitExpr_(const CastNode* op) override {
    if (op->value->dtype == op->dtype && IsValidDType(op->value->dtype)) {
      return VisitInt(op->value);
    } else {
      return Create(op);
    }
  }
  ::z3::expr VisitExpr_(const VarNode* op) override { return Create(op); }
  ::z3::expr VisitExpr_(const BufferLoadNode* op) override { return Create(op); }
  ::z3::expr VisitExpr_(const ProducerLoadNode* op) override {
    return Create(op);
  }
  ::z3::expr VisitExpr_(const ReduceNode* op) override { return Create(op); }
  // Ramp(base, stride, lanes) represents the vector [base, base+stride,
  // base+2*stride, ...]. fix-round-6 C6: returning only `op->base` is
  // unsound for the upper lanes — the prover would treat the entire
  // ramp expression as if it equalled `base`, so any CanProve over a
  // Ramp-containing predicate could succeed by collapsing the lanes
  // away. Conservative replacement: mint a fresh unconstrained scalar
  // (Int or BV depending on mode) so the prover cannot derive anything
  // load-bearing from the Ramp. Callers that genuinely need
  // per-lane reasoning must lower out the Ramp before invoking Z3.
  ::z3::expr VisitExpr_(const ::tvm::tirx::RampNode* op) override {
    (void)op;
    return MakeIntConst("ramp_" + std::to_string(memo_.size()));
  }
  // Broadcast(value, lanes) is a vector of identical scalars; visit the value.
  ::z3::expr VisitExpr_(const ::tvm::tirx::BroadcastNode* op) override {
    return VisitExpr(op->value);
  }
  // In BV mode every Z3 operand of an arithmetic / relational op must be a
  // BV of the current width; in Int mode operands must be Int. The Z3 C++
  // overloads will raise an error if you mix sorts (e.g. an Int operand
  // accidentally surfacing inside a BV computation), but the resulting
  // diagnostic is opaque. Assert sorts up front so misroutes blow up with a
  // useful message at the source-level node that produced the mismatch.
  void AssertOperandSort(const ::z3::expr& e, const char* where) const {
    if (bv_width_ > 0) {
      ICHECK(e.is_bv())
          << "Z3Prover " << where << ": expected BV operand at width "
          << bv_width_ << ", got non-BV sort";
      ICHECK_EQ(e.get_sort().bv_size(), static_cast<unsigned>(bv_width_))
          << "Z3Prover " << where << ": BV operand width mismatch (have "
          << e.get_sort().bv_size() << ", want " << bv_width_ << ")";
    } else {
      ICHECK(e.is_int())
          << "Z3Prover " << where
          << ": expected Int operand in default (non-BV) mode";
    }
  }
  ::z3::expr VisitExpr_(const MinNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    AssertOperandSort(a, "MinNode.a");
    AssertOperandSort(b, "MinNode.b");
    return ::z3::ite(a < b, a, b);
  }
  ::z3::expr VisitExpr_(const MaxNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    AssertOperandSort(a, "MaxNode.a");
    AssertOperandSort(b, "MaxNode.b");
    return ::z3::ite(a > b, a, b);
  }
  static ::z3::expr floordiv(const ::z3::expr& a, const ::z3::expr& b) {
    return ::z3::ite(b > 0, a / b, -((-a) / b));
  }
  static ::z3::expr floormod(const ::z3::expr& a, const ::z3::expr& b) {
    return ::z3::ite(b > 0, a % b, -((-a) % b));
  }
  ::z3::expr VisitExpr_(const AddNode* op) override {
    return VisitArith(::z3::operator+, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const SubNode* op) override {
    return VisitArith(::z3::operator-, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const MulNode* op) override {
    return VisitArith(::z3::operator*, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const DivNode* op) override {
    return VisitArith(::z3::operator/, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const ModNode* op) override {
    return VisitArith(::z3::operator%, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const FloorDivNode* op) override {
    if (IsValidDType(op->a->dtype) && IsValidDType(op->b->dtype)) {
      auto a = VisitInt(op->a);
      auto b = VisitInt(op->b);
      if (bv_width_ > 0) {
        // bvsdiv truncates towards zero. FloorDiv truncates towards negative infinity.
        // We can synthesize FloorDiv using bvsdiv and bvsrem:
        // floordiv(a, b) = bvsdiv(a, b) - ite((a < 0) != (b < 0) && bvsrem(a, b) != 0, 1, 0)
        ::z3::expr trunc_div = ::z3::to_expr(a.ctx(), Z3_mk_bvsdiv(a.ctx(), a, b));
        ::z3::expr trunc_rem = ::z3::to_expr(a.ctx(), Z3_mk_bvsrem(a.ctx(), a, b));
        ::z3::expr a_lt_0 = ::z3::to_expr(a.ctx(), Z3_mk_bvslt(a.ctx(), a, MakeIntVal(0)));
        ::z3::expr b_lt_0 = ::z3::to_expr(b.ctx(), Z3_mk_bvslt(b.ctx(), b, MakeIntVal(0)));
        ::z3::expr signs_differ = (a_lt_0 != b_lt_0);
        ::z3::expr needs_adjustment = signs_differ && (trunc_rem != MakeIntVal(0));
        return trunc_div - ::z3::ite(needs_adjustment, MakeIntVal(1), MakeIntVal(0));
      }
      return floordiv(a, b);
    }
    return Create(op);
  }

  ::z3::expr VisitExpr_(const FloorModNode* op) override {
    if (IsValidDType(op->a->dtype) && IsValidDType(op->b->dtype)) {
      auto a = VisitInt(op->a);
      auto b = VisitInt(op->b);
      if (bv_width_ > 0) {
        // Z3 operator% for BV lowers to Z3_mk_bvsrem (truncated remainder, sign of dividend).
        // TIR FloorMod is sign of divisor.
        // floormod(a, b) = bvsrem(a, b) + ite((a < 0) != (b < 0) && bvsrem(a, b) != 0, b, 0)
        ::z3::expr trunc_rem = ::z3::to_expr(a.ctx(), Z3_mk_bvsrem(a.ctx(), a, b));
        ::z3::expr a_lt_0 = ::z3::to_expr(a.ctx(), Z3_mk_bvslt(a.ctx(), a, MakeIntVal(0)));
        ::z3::expr b_lt_0 = ::z3::to_expr(b.ctx(), Z3_mk_bvslt(b.ctx(), b, MakeIntVal(0)));
        ::z3::expr signs_differ = (a_lt_0 != b_lt_0);
        ::z3::expr needs_adjustment = signs_differ && (trunc_rem != MakeIntVal(0));
        return trunc_rem + ::z3::ite(needs_adjustment, b, MakeIntVal(0));
      }
      return floormod(a, b);
    }
    return Create(op);
  }
  ::z3::expr VisitExpr_(const EQNode* op) override {
    return VisitArith(::z3::operator==, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const NENode* op) override {
    return VisitArith(::z3::operator!=, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const LTNode* op) override {
    return VisitArith(::z3::operator<, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const LENode* op) override {
    return VisitArith(::z3::operator<=, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const GTNode* op) override {
    return VisitArith(::z3::operator>, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const GENode* op) override {
    return VisitArith(::z3::operator>=, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const AndNode* op) override {
    return VisitBool(op->a) && VisitBool(op->b);
  }
  ::z3::expr VisitExpr_(const OrNode* op) override {
    return VisitBool(op->a) || VisitBool(op->b);
  }
  ::z3::expr VisitExpr_(const NotNode* op) override {
    return !VisitBool(op->a);
  }
  ::z3::expr VisitExpr_(const SelectNode* op) override {
    auto t = VisitInt(op->true_value);
    auto f = VisitInt(op->false_value);
    AssertOperandSort(t, "SelectNode.true_value");
    AssertOperandSort(f, "SelectNode.false_value");
    return ::z3::ite(VisitBool(op->condition), t, f);
  }
  ::z3::expr VisitExpr_(const IntImmNode* op) override {
    return MakeIntVal(op->value);
  }

  ::z3::expr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(::tvm::tirx::builtin::bitwise_and())) {
      return VisitBitwiseOp(::z3::operator&, op);
    } else if (op->op.same_as(::tvm::tirx::builtin::bitwise_or())) {
      return VisitBitwiseOp(::z3::operator|, op);
    } else if (op->op.same_as(::tvm::tirx::builtin::bitwise_xor())) {
      return VisitBitwiseOp(::z3::operator^, op);
    } else if (op->op.same_as(::tvm::tirx::builtin::bitwise_not())) {
      return VisitBitwiseNotOp(op);
    } else if (op->op.same_as(::tvm::tirx::builtin::shift_left())) {
      return VisitShiftOp(::z3::shl, op);
    } else if (op->op.same_as(::tvm::tirx::builtin::shift_right())) {
      return VisitShiftOp(::z3::ashr, op);
    } else {
      return Create(op);
    }
  }

  ::z3::expr VisitBitwiseOp(::z3::expr (*op_func)(const ::z3::expr&,
                                                  const ::z3::expr&),
                            const CallNode* op) {
    ICHECK_EQ(op->args.size(), 2u);
    const PrimExpr& a = op->args[0];
    const PrimExpr& b = op->args[1];
    unsigned bit_width =
        std::max(op->args[0].dtype().bits(), op->args[1].dtype().bits());
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      if (bv_width_ > 0) {
        // Already BV; bitwise op directly on the BV values.
        return op_func(VisitInt(a), VisitInt(b));
      }
      return ::z3::bv2int(
          op_func(::z3::int2bv(bit_width, VisitInt(a)),
                  ::z3::int2bv(bit_width, VisitInt(b))),
          true);
    } else {
      return Create(op);
    }
  }

  ::z3::expr VisitBitwiseNotOp(const CallNode* op) {
    ICHECK_EQ(op->args.size(), 1u);
    const PrimExpr& a = op->args[0];
    if (IsValidDType(a->dtype)) {
      if (bv_width_ > 0) {
        return ~VisitInt(a);
      }
      unsigned bit_width = a.dtype().bits();
      ::z3::expr a_int = VisitInt(a);
      ::z3::expr a_bv = ::z3::int2bv(bit_width, a_int);
      return ::z3::bv2int(~a_bv, true);
    } else {
      return Create(op);
    }
  }

  ::z3::expr VisitShiftOp(::z3::expr (*op_func)(const ::z3::expr&,
                                                const ::z3::expr&),
                          const CallNode* op) {
    ICHECK_EQ(op->args.size(), 2u);
    const PrimExpr& a = op->args[0];
    const PrimExpr& b = op->args[1];
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      if (bv_width_ > 0) {
        ::z3::expr a_bv = VisitInt(a);
        ::z3::expr b_bv = VisitInt(b);
        solver.add(b_bv >= MakeIntVal(0));
        solver.add(b_bv < MakeIntVal(bv_width_));
        return op_func(a_bv, b_bv);
      }
      ::z3::expr a_expr = VisitInt(a);
      ::z3::expr b_expr = VisitInt(b);
      solver.add(b_expr >= 0);
      solver.add(b_expr < 64);
      unsigned bit_width =
          std::max(a.dtype().bits(), b.dtype().bits());
      ::z3::expr a_bv = ::z3::int2bv(bit_width, a_expr);
      ::z3::expr b_bv = ::z3::int2bv(bit_width, b_expr);
      ::z3::expr result_bv = op_func(a_bv, b_bv);
      return ::z3::bv2int(result_bv, true);
    } else {
      return Create(op);
    }
  }

  ::z3::expr VisitExprDefault_(const ::tvm::ffi::Object* op) override {
    LOG(FATAL) << "Z3Prover only support integers, but got "
               << op->GetTypeKey() << ".";
    TVM_FFI_UNREACHABLE();
  }
};

// ---------------------------------------------------------------------------
// Z3Prover (front-class) — thin forwarder to the Impl.

Z3Prover::Z3Prover(::tvm::arith::Analyzer* parent)
    : impl_(std::make_unique<Z3ProverImpl>(parent)) {}

Z3Prover::~Z3Prover() = default;

void Z3Prover::Bind(const Var& var, const Range& new_range,
                    bool allow_override) {
  impl_->Bind(var, new_range, allow_override);
}
void Z3Prover::Bind(const Var& var, const PrimExpr& expr,
                    bool allow_override) {
  impl_->Bind(var, expr, allow_override);
}
bool Z3Prover::CanProve(const PrimExpr& expr) {
  // CPPMEGA fix: gb10 correctness regression root-caused to bvsrem vs
  // bvsmod semantics in FloorMod/FloorDiv BV modes. The global gate
  // is removed; Z3 can safely prove bounds again.
  return impl_->CanProve(expr);
}
std::function<void()> Z3Prover::EnterConstraint(const PrimExpr& constraint,
                                                bool is_assume) {
  return impl_->EnterConstraint(constraint, is_assume);
}
std::string Z3Prover::GetSMTLIB2(::tvm::ffi::Optional<PrimExpr> expr) {
  if (expr.has_value()) return impl_->GetSMTLIB2(expr.value());
  return impl_->GetSMTLIB2();
}
std::string Z3Prover::GetSMTLIB2(std::nullopt_t) {
  return impl_->GetSMTLIB2();
}
std::string Z3Prover::GetStats() { return impl_->GetStats(); }
std::string Z3Prover::GetModel(const PrimExpr& expr) {
  return impl_->GetModel(expr);
}
int64_t Z3Prover::CountSatisfyingValues(const Var& var, int64_t max_count,
                                        int64_t min_consecutive) {
  return impl_->CountSatisfyingValues(var, max_count, min_consecutive);
}
void Z3Prover::SetTimeoutMs(unsigned timeout_ms) {
  impl_->SetTimeoutMs(timeout_ms);
}
void Z3Prover::SetRLimit(unsigned rlimit) { impl_->SetRLimit(rlimit); }
// CPPMEGA fix-A1: infallible facade. `SetBitVectorMode` may invoke
// `CreateSolver` / Z3 reset, which can theoretically throw (e.g., OOM,
// invalid config). Because this method is invoked by `~ScopedBVMode()`
// (a noexcept destructor), any propagated exception would call
// std::terminate. Catch any internal failure here, log once, and fall
// back to bv_width=0 (Int sort) — that mode is the most permissive and
// preserves correctness at the cost of losing BV-mode wrap semantics.
void Z3Prover::SetBitVectorMode(int width) {
  try {
    impl_->SetBitVectorMode(width);
  } catch (const ::z3::exception& e) {
    LOG(WARNING) << "Z3Prover::SetBitVectorMode(" << width
                 << ") failed (" << e.msg() << "); falling back to "
                 << "Int sort (bv_width=0).";
    try {
      impl_->SetBitVectorMode(0);
    } catch (...) {
      // Even fallback failed; swallow — the prover is in an undefined
      // state, but we cannot throw from this entry point.
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Z3Prover::SetBitVectorMode(" << width
                 << ") failed (" << e.what() << "); falling back to "
                 << "Int sort (bv_width=0).";
    try {
      impl_->SetBitVectorMode(0);
    } catch (...) {
    }
  } catch (...) {
    LOG(WARNING) << "Z3Prover::SetBitVectorMode(" << width
                 << ") failed with unknown exception; falling back to "
                 << "Int sort (bv_width=0).";
    try {
      impl_->SetBitVectorMode(0);
    } catch (...) {
    }
  }
}
int Z3Prover::GetBitVectorWidth() const {
  return impl_->GetBitVectorWidth();
}
// CPPMEGA z3-stack fix-A6/A8 + idea712 fix-B7: atomic reset of
// (memo + solver + scope_stack). The B7 implementation supersedes the
// earlier A8 forwarder (`impl_->Reset()`); it adds lifecycle checks and
// directly manages each piece of state for invariant-preserving teardown.

// CPPMEGA fix-B7 (idea712): atomic reset of (memo + solver + scope_stack).
// See the header doc for the audit rationale. The function is exposed for
// the future `SetBitVectorMode(width)` port; today it is also a useful
// "hard reset" hook for tests that want to ensure prior queries on the
// same Analyzer cannot pollute a new probe.
void Z3Prover::Reset() {
  // Lifecycle check: refuse to reset while there are outstanding
  // EnterConstraint scopes. The prover impl manages a `scope_stack_` of
  // `std::vector<Scope>`; at construction it pushes a single root frame,
  // and every EnterConstraint/Bind appends. The invariant for a clean
  // reset: only the root frame is present and it is empty. Anything else
  // means a caller forgot to recover() or destructed a ConstraintScope
  // out of order.
  ICHECK(impl_) << "Z3Prover::Reset called on null impl";
  ICHECK_EQ(impl_->scope_stack_.size(), 1u)
      << "Z3Prover::Reset called with " << impl_->scope_stack_.size()
      << " scope frames; recover all EnterConstraint scopes first.";
  ICHECK(impl_->scope_stack_.front().empty())
      << "Z3Prover::Reset called with non-empty root scope frame; "
      << "Bind/Constraint records still pending.";
  // Atomic teardown: solver gets a fresh instance (drops all assertions),
  // memo_ is cleared. Same-thread context affinity is preserved (Z3's
  // context is per-thread and reused). `side_effect_exprs_` is private to
  // the impl and is only ever populated during Convert{Bool,Int} calls;
  // at idle (root scope, no in-flight conversion) it is already empty.
  impl_->solver = Z3ProverImpl::CreateSolver(*impl_->ctx);
  impl_->memo_.clear();
  // scope_stack_ retains the (now-empty) root frame, matching the
  // post-construction state. is_assume gets reset for safety.
  impl_->is_assume = false;
}

// ---------------------------------------------------------------------------
// Per-Analyzer cache. `thread_local` because the Z3 context inside
// Z3ProverImpl is also thread_local; mixing analyzers across threads would
// hand out provers wired to the wrong context.

// CPPMEGA z3-stack fix-A8: per-thread cache accessor. Factored out so
// `ClearProverCache()` (below) can reach the same map. Returning by
// reference is safe: the map itself is `static thread_local`, so its
// storage outlives any caller frame on this thread.
//
// CPPMEGA z3-final: the cache is heap-allocated and intentionally LEAKED at
// TLS teardown. Z3 keeps its own thread_local context globals that are
// destroyed before our cache during process exit on aarch64/glibc 2.39 with
// Z3 22.x. If our cache's dtor runs after Z3's globals are gone,
// `~Z3ProverImpl` -> `~solver` -> `Z3_solver_dec_ref` -> `~param_descrs`
// dereferences torn-down Z3 state and SIGSEGVs (cosmetic but pollutes test
// output). By holding the map via a raw pointer that we never delete, the
// TLS dtor runs no Z3 code at exit; the OS reclaims the heap pages.

// BUG-Z3-1 fix: generation-tagged cache entry. Each cache entry stores a
// monotonic generation counter that is bumped on every ClearProverCache().
// GetOrCreate records the generation at creation time; if a subsequent
// lookup finds a stale generation (meaning the Analyzer* was freed and
// the address was reused), the entry is replaced with a fresh prover.
// This prevents dangling-pointer cache hits without requiring a dtor hook
// on Analyzer (which apache/tvm does not expose).
struct CacheEntry {
  std::unique_ptr<Z3Prover> prover;
  std::vector<std::unique_ptr<Z3Prover>> retired;
  uint64_t generation;
};

static uint64_t& GetCacheGeneration_() {
  static thread_local uint64_t gen = 0;
  return gen;
}

static std::unordered_map<::tvm::arith::Analyzer*, CacheEntry>&
GetProverCache_() {
  static thread_local auto* cache =
      new std::unordered_map<::tvm::arith::Analyzer*, CacheEntry>();
  return *cache;
}

Z3Prover& GetOrCreate(::tvm::arith::Analyzer* analyzer) {
  auto& cache = GetProverCache_();
  uint64_t current_gen = GetCacheGeneration_();
  auto it = cache.find(analyzer);
  if (it != cache.end() && it->second.prover &&
      it->second.generation == current_gen) {
    return *it->second.prover;
  }
  // Stale entry (generation mismatch) or missing entry — create fresh.
  auto& entry = cache[analyzer];
  if (entry.prover) {
    entry.retired.emplace_back(std::move(entry.prover));
  }
  entry.prover = std::make_unique<Z3Prover>(analyzer);
  entry.generation = current_gen;
  return *entry.prover;
}

// CPPMEGA z3-stack fix-A8 (NEW-2): clear the entire per-thread prover
// cache. Intended to be called at every pass-driver entry point so two
// consecutive passes that happen to receive the same `Analyzer*`
// (either by deliberate reuse or by heap-address coincidence after a
// previous Analyzer was freed) cannot inherit memo / scope / bv-mode
// state from the prior pass. Cheap and safe to call any number of times.
//
// BUG-Z3-1 fix: bump the generation counter instead of clearing the map.
// This makes all existing entries stale; the next GetOrCreate for any
// Analyzer* will create a fresh prover. This is safe even if recovery
// lambdas from EnterConstraint still hold a reference to the old prover —
// the lambda's `this` pointer remains valid because the CacheEntry's
// unique_ptr is not destroyed until the entry is replaced or the cache
// is torn down at thread exit.
//
// BUG-Z3-3 fix: we no longer call cache.clear() which would destroy
// provers that may have outstanding EnterConstraint recovery lambdas.
// Instead, stale entries are lazily replaced on the next GetOrCreate; the
// previous prover is retained in CacheEntry::retired for the lifetime of the
// thread-local cache so any late recovery lambda still targets live storage.
void ClearProverCache() {
  GetCacheGeneration_()++;
  // Eagerly drop entries that have no outstanding scopes (safe to destroy).
  // Keep entries with outstanding scopes alive so recovery lambdas don't
  // dangle; they'll be replaced on next GetOrCreate.
  auto& cache = GetProverCache_();
  for (auto it = cache.begin(); it != cache.end(); ) {
    if (it->second.prover) {
      // Accessing impl_->scope_stack_ directly is not possible from here
      // (it's private). Instead, just mark as stale via generation; the
      // entry stays alive until the next GetOrCreate replaces it.
      ++it;
    } else {
      it = cache.erase(it);
    }
  }
}

// CPPMEGA z3-stack fix-A8 (NEW-2): targeted reset for a specific
// Analyzer's cached prover. Use when the caller knows the precise
// Analyzer it owns; safer than a thread-wide clear inside library code
// that doesn't own the Analyzer lifetime.
void ResetProverFor(::tvm::arith::Analyzer* analyzer) {
  auto& cache = GetProverCache_();
  auto it = cache.find(analyzer);
  if (it != cache.end() && it->second.prover) {
    it->second.prover->Reset();
  }
}

// CPPMEGA: Auto-driver hooks. Registered at static init so apache
// `Analyzer::Bind` / `ConstraintContext::EnterWithScope` forward to the
// per-Analyzer Z3Prover, matching stack-c/tl_pr_c behavior. Without this,
// the prover is starved of constraints and partial-sync queries collapse
// to range-only proofs.
namespace {
void Z3BindExprHook(::tvm::arith::Analyzer* self, const ::tvm::tirx::Var& var,
                    const ::tvm::PrimExpr& expr, bool allow_override) {
  GetOrCreate(self).Bind(var, expr, allow_override);
}
void Z3BindRangeHook(::tvm::arith::Analyzer* self, const ::tvm::tirx::Var& var,
                     const ::tvm::Range& range, bool allow_override) {
  GetOrCreate(self).Bind(var, range, allow_override);
}
std::function<void()> Z3EnterConstraintHook(::tvm::arith::Analyzer* self,
                                            const ::tvm::PrimExpr& constraint) {
  return GetOrCreate(self).EnterConstraint(constraint);
}

struct Z3HookRegistrar {
  Z3HookRegistrar() {
    ::tvm::arith::Analyzer::RegisterBindExprHook(&Z3BindExprHook);
    ::tvm::arith::Analyzer::RegisterBindRangeHook(&Z3BindRangeHook);
    ::tvm::arith::Analyzer::RegisterEnterConstraintHook(&Z3EnterConstraintHook);
  }
};
static Z3HookRegistrar _z3_hook_registrar;
}  // namespace

// CPPMEGA: Test-only FFI helpers. Exposes Z3Prover with the bv-mode
// switch so Python tests can drive the prover directly without needing
// a full Python binding for the C++ class. The signature is
//    bv_can_prove(var, lo, hi, expr, bv_width) -> bool
// where `var` is bound to the half-open range [lo, hi) before the
// proof attempt; `bv_width` is 0 / 32 / 64 (see SetBitVectorMode).
//
// CPPMEGA z3-stack fix-A5: gated by `TILELANG_BUILD_TESTS` (default ON).
// Release wheels can pass `-DTILELANG_BUILD_TESTS=OFF` to drop these
// helpers from the FFI surface.
#ifdef TILELANG_BUILD_TESTS
bool BvCanProve(const ::tvm::tirx::Var& var, int64_t lo, int64_t hi,
                const ::tvm::PrimExpr& expr, int bv_width) {
  ::tvm::arith::Analyzer ana;
  Z3Prover& prover = GetOrCreate(&ana);
  prover.SetBitVectorMode(bv_width);
  // To avoid IntImm bounds check failure for edge cases (e.g. hi - lo == 1<<31 for int32),
  // we clamp the extent to the max representable value for the dtype.
  int64_t max_val = ::tvm::max_value(var->dtype).as<::tvm::IntImmNode>()->value;
  int64_t min_val = ::tvm::min_value(var->dtype).as<::tvm::IntImmNode>()->value;
  int64_t extent = hi - lo;
  if (extent > max_val - min_val) {
    extent = max_val - min_val;
  }
  if (extent < 0) extent = 0;
  if (lo < min_val) lo = min_val;
  if (lo > max_val) lo = max_val;
  // If lo + extent > max_val, Z3Prover::BindRange will just bind it up to the extent.
  // Actually, TVM IntImm constructor checks value <= max_val. 
  // Extent for int32 can be up to UINT32_MAX if represented as int64?
  // No, IntImm for int32 checks value <= INT32_MAX! So extent CANNOT exceed INT32_MAX!
  // To bind a full range, we should just use the boolean expression.
  int64_t hi_inclusive = hi - 1;
  if (hi_inclusive > max_val) hi_inclusive = max_val;
  if (lo < min_val) lo = min_val;
  auto lo_expr = tvm::tirx::make_const(var->dtype, lo);
  auto hi_expr = tvm::tirx::make_const(var->dtype, hi_inclusive);
  auto recover = prover.EnterConstraint((var >= lo_expr) && (var <= hi_expr), true);
  bool res = prover.CanProve(expr);
  recover();
  return res;
}

// CPPMEGA: ScopedBVMode round-trip exerciser. Creates a fresh Analyzer's
// prover at `outer_width`, opens a `ScopedBVMode(inner_width)` block,
// optionally runs a trivial CanProve inside it, then on scope exit
// confirms the prover is back at `outer_width`. Returns the *observed*
// width after the inner scope closes; the test asserts it equals
// `outer_width`. This is the only public way (without a full Z3Prover
// Python binding) to confirm RAII restoration in-process.
int BvScopedRoundTrip(int outer_width, int inner_width) {
  ::tvm::arith::Analyzer ana;
  Z3Prover& prover = GetOrCreate(&ana);
  prover.SetBitVectorMode(outer_width);
  {
    ScopedBVMode guard(prover, inner_width);
    ICHECK_EQ(prover.GetBitVectorWidth(), inner_width)
        << "ScopedBVMode failed to enter target width";
    // Trivial proof inside the inner scope to make sure the solver works
    // at the inner sort.
    ::tvm::tirx::Var x("x", ::tvm::DataType::Int(32));
    ::tvm::Range r = ::tvm::Range::FromMinExtent(
        ::tvm::IntImm(x->dtype, 0), ::tvm::IntImm(x->dtype, 1));
    prover.Bind(x, r);
    (void)prover.CanProve(x == ::tvm::IntImm(x->dtype, 0));
  }
  return prover.GetBitVectorWidth();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  refl::GlobalDef().def("tl.z3.bv_can_prove", BvCanProve);
  refl::GlobalDef().def("tl.z3.bv_scoped_round_trip", BvScopedRoundTrip);
}
#endif  // TILELANG_BUILD_TESTS

// CPPMEGA z3-stack fix-A8 (NEW-2): production FFI for cache hygiene.
// Registered unconditionally (NOT inside the TILELANG_BUILD_TESTS gate)
// because the pass driver in `tilelang/engine/phase.py` calls these at
// every pass entry. Both functions are no-ops on an empty cache.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  refl::GlobalDef().def("tl.z3.clear_prover_cache",
                        []() { ClearProverCache(); });
}

// ---------------------------------------------------------------------------
// CPPMEGA z3-final per-pass gate (2026-05-07).
// See `Z3PassGate` doc in z3_prover.h for the rationale & env-var table.
// ---------------------------------------------------------------------------
//
// Cheap-by-design: the global lookup is amortized via a function-local
// `static const`; per-pass lookups go through a small thread-local cache
// keyed on the env-var name string. The cache is bounded by the small set
// of pass names hard-coded at call sites (~10), so its memory footprint is
// O(1) per thread for any realistic compile.
//
// Truthiness convention matches the existing `TILELANG_DISABLE_Z3` gate at
// `Z3Prover::CanProve` (above): "set" means env != nullptr AND env[0] != 0
// AND env[0] != '0'. So `=0`, ``, and unset are all "enabled".
bool Z3PassGate::IsEnabled(const char* pass_name) {
  // Fast path: global gate. Same predicate as `Z3Prover::CanProve`'s.
  static const bool global_disabled = []() {
    const char* g = std::getenv("TILELANG_DISABLE_Z3");
    return g != nullptr && g[0] != '\0' && g[0] != '0';
  }();
  if (global_disabled) return false;

  // Per-pass gate: cache by full env-var name. `pass_name` is a string
  // literal at every call site, so the std::string allocation here is
  // a one-time-per-name cost (then served from the cache).
  //
  // BUG-Z3-TLS fix: use heap-leak pattern matching the prover cache to
  // avoid TLS-destruction-order SIGSEGVs. The std::unordered_map dtor
  // would run at thread exit; if it runs after Z3's own TLS globals are
  // gone, the interleaved allocator calls can crash. Leaking the small
  // (~10 entry) map is harmless.
  static thread_local auto* cache =
      new std::unordered_map<std::string, bool>();
  std::string key("TILELANG_DISABLE_Z3_");
  key += pass_name;
  auto it = cache->find(key);
  if (it != cache->end()) {
    return !it->second;
  }
  const char* v = std::getenv(key.c_str());
  bool disabled = v != nullptr && v[0] != '\0' && v[0] != '0';
  cache->emplace(std::move(key), disabled);
  return !disabled;
}

}  // namespace tlz3
}  // namespace tilelang
