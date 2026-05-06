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
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
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
      ::z3::expr e = ctx->int_const(name.c_str());
      if (dtype.is_uint() && dtype.bits() == 64) {
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
    auto side_effect_exprs = std::move(side_effect_exprs_);
    side_effect_exprs_.clear();
    if (is_assume_in) {
      return [this, side_effect_exprs]() {
        solver.pop();
        for (const auto& expr : side_effect_exprs) {
          memo_.erase(expr);
        }
        scope_stack_.pop_back();
      };
    } else {
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
    ::z3::expr_vector constr(*ctx);
    constr.push_back(!ConvertBool(expr));
    auto result = solver.check(constr);
    constr.pop_back();
    return result == ::z3::unsat;
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
    auto var_expr = Create(var.as<PrimExprNode>());
    memo_.emplace(var, var_expr);
    if (tirx_op::is_const_int(range->min) &&
        tirx_op::is_const_int(range->min + range->extent)) {
      int64_t min_value = *tirx_op::as_const_int(range->min);
      int64_t max_value = *tirx_op::as_const_int(range->min + range->extent);
      if (min_value < max_value) {
        solver.add(ctx->int_val(min_value) <= var_expr);
        solver.add(var_expr < ctx->int_val(max_value));
      }
    } else {
      solver.add(ConvertBool(range->extent <= 0 ||
                             (range->min <= var &&
                              var < range->min + range->extent)));
    }
  }

  void SetTimeoutMs(unsigned timeout_ms_in) {
    this->timeout_ms = timeout_ms_in;
    solver.set("timeout", timeout_ms_in);
  }

  void SetRLimit(unsigned rlimit_in) {
    this->rlimit = rlimit_in;
    solver.set("rlimit", rlimit_in);
  }

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

    solver.set("model", true);
    solver.push();

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
      solver.add(z3_var != ctx->int_val(val));
    }

    solver.pop();
    solver.set("model", false);

    for (const auto& expr : side_effect_exprs_) {
      memo_.erase(expr);
    }
    side_effect_exprs_.clear();

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
      return ::z3::ite(e, ctx->int_val(1), ctx->int_val(0));
    } else {
      return e;
    }
  }

  ::z3::expr VisitBool(const PrimExpr& e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) return expr;
    return expr != ctx->int_val(0);
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
    if (IsValidDType(op->value->dtype) && IsValidDType(op->dtype)) {
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
  ::z3::expr VisitExpr_(const MinNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return ::z3::ite(a < b, a, b);
  }
  ::z3::expr VisitExpr_(const MaxNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
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
    return VisitArith(floordiv, op, op->a, op->b);
  }
  ::z3::expr VisitExpr_(const FloorModNode* op) override {
    return VisitArith(floormod, op, op->a, op->b);
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
    return ::z3::ite(VisitBool(op->condition), VisitInt(op->true_value),
                     VisitInt(op->false_value));
  }
  ::z3::expr VisitExpr_(const IntImmNode* op) override {
    return ctx->int_val(op->value);
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

// ---------------------------------------------------------------------------
// Per-Analyzer cache. `thread_local` because the Z3 context inside
// Z3ProverImpl is also thread_local; mixing analyzers across threads would
// hand out provers wired to the wrong context.

Z3Prover& GetOrCreate(::tvm::arith::Analyzer* analyzer) {
  static thread_local std::unordered_map<::tvm::arith::Analyzer*,
                                         std::unique_ptr<Z3Prover>>
      cache;
  auto& slot = cache[analyzer];
  if (!slot) {
    slot = std::make_unique<Z3Prover>(analyzer);
  }
  return *slot;
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

}  // namespace tlz3
}  // namespace tilelang
