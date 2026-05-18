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
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file metal_scalar_intrinsics.cc
 * \brief Bind pure Metal scalar kernel-attribute intrinsics once per function.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vendored/let_stmt.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

namespace {

bool MetalScalarIntrinsicPrefix(const CallNode *op, std::string *prefix) {
  const auto *op_node = op->op.as<OpNode>();
  if (op_node == nullptr) {
    return false;
  }
  const std::string &name = op_node->name;
  if (name == "tir.metal.thread_position_in_grid_x" ||
      name == "tirx.metal.thread_position_in_grid_x") {
    *prefix = "grid_tid";
    return true;
  }
  if (name == "tir.metal.thread_position_in_threadgroup_x" ||
      name == "tirx.metal.thread_position_in_threadgroup_x") {
    *prefix = "threadgroup_tid";
    return true;
  }
  if (name == "tir.metal.thread_index_in_simdgroup" ||
      name == "tirx.metal.thread_index_in_simdgroup") {
    *prefix = "simd_lane";
    return true;
  }
  return false;
}

std::string CallKey(const CallNode *op) {
  const auto *op_node = op->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  return op_node->name + ":" + std::to_string(op->dtype.code()) + ":" +
         std::to_string(op->dtype.bits()) + ":" + std::to_string(op->dtype.lanes());
}

class MetalScalarIntrinsicBinder : public StmtExprMutator {
 public:
  Stmt Rewrite(Stmt body) { return VisitStmt(std::move(body)); }

  Stmt VisitStmt_(const BindNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      std::string prefix;
      if (MetalScalarIntrinsicPrefix(call, &prefix)) {
        vars_[CallKey(call)] = op->var;
        return ffi::GetRef<Stmt>(op);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    std::string prefix;
    if (!MetalScalarIntrinsicPrefix(op, &prefix)) {
      return StmtExprMutator::VisitExpr_(op);
    }

    std::string key = CallKey(op);
    auto it = vars_.find(key);
    if (it != vars_.end()) {
      return it->second;
    }

    std::string name =
        bindings_.empty() ? prefix : prefix + "_" + std::to_string(bindings_.size());
    Var var(name, op->dtype);
    vars_.emplace(std::move(key), var);
    bindings_.push_back(Bind(var, ffi::GetRef<PrimExpr>(op)));
    return var;
  }

  const std::vector<Stmt> &bindings() const { return bindings_; }

 private:
  std::unordered_map<std::string, Var> vars_;
  std::vector<Stmt> bindings_;
};

class PureScalarExprChecker : public ExprVisitor {
 public:
  static bool Check(const PrimExpr &expr) {
    if (!expr.defined() || !expr.dtype().is_scalar()) {
      return false;
    }
    PureScalarExprChecker checker;
    checker.VisitExpr(expr);
    return checker.pure_;
  }

 private:
  void VisitExpr_(const BufferLoadNode *op) final { pure_ = false; }
  void VisitExpr_(const ProducerLoadNode *op) final { pure_ = false; }
  void VisitExpr_(const ReduceNode *op) final { pure_ = false; }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_call_packed()) ||
        op->op.same_as(builtin::call_extern()) ||
        op->op.same_as(builtin::call_pure_extern())) {
      pure_ = false;
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

  bool pure_{true};
};

bool IsTrivialScalarExpr(const PrimExpr &expr) {
  return expr.as<VarNode>() || expr.as<IntImmNode>() || expr.as<FloatImmNode>() ||
         expr.as<StringImmNode>();
}

bool IsMetalThreadIndexVar(const VarNode *op) {
  if (op == nullptr) {
    return false;
  }
  const std::string &name = op->name_hint;
  return name == "thread_position_in_threadgroup.x" ||
         name == "threadgroup_position_in_grid.x" || name == "threadIdx.x" ||
         name == "blockIdx.x" || name == "lane";
}

class MetalThreadIndexExprDetector : public ExprVisitor {
 public:
  static bool Contains(const PrimExpr &expr) {
    if (!expr.defined() || !expr.dtype().is_scalar()) {
      return false;
    }
    MetalThreadIndexExprDetector detector;
    detector.VisitExpr(expr);
    return detector.found_;
  }

 private:
  void VisitExpr_(const VarNode *op) final {
    if (IsMetalThreadIndexVar(op)) {
      found_ = true;
    }
  }

  bool found_{false};
};

bool ShouldInlineMetalThreadIndexExpr(const PrimExpr &expr) {
  return PureScalarExprChecker::Check(expr) && MetalThreadIndexExprDetector::Contains(expr);
}

bool IsWideningIntegerIndexCast(const CastNode *op) {
  if (op == nullptr || !op->dtype.is_scalar() || !op->value.dtype().is_scalar()) {
    return false;
  }
  if (!(op->dtype.is_int() || op->dtype.is_uint()) ||
      !(op->value.dtype().is_int() || op->value.dtype().is_uint())) {
    return false;
  }
  return op->dtype.bits() >= 64 && op->value.dtype().bits() <= 32;
}

class MetalIndexCastNormalizer : public ExprMutator {
 public:
  static PrimExpr Normalize(const PrimExpr &expr) {
    MetalIndexCastNormalizer normalizer;
    return normalizer.VisitExpr(expr);
  }

 private:
  static bool CoerceIntImmTo(const PrimExpr &expr, DataType dtype, PrimExpr *out) {
    if (!(dtype.is_int() || dtype.is_uint()) || !dtype.is_scalar()) {
      return false;
    }
    const auto *imm = expr.as<IntImmNode>();
    if (imm == nullptr) {
      return false;
    }
    if (dtype.is_uint() && imm->value < 0) {
      return false;
    }
    *out = tirx::make_const(dtype, imm->value);
    return true;
  }

  static bool IsScalarInteger(DataType dtype) {
    return dtype.is_scalar() && (dtype.is_int() || dtype.is_uint());
  }

  static bool IsPositiveIntImm(const PrimExpr &expr) {
    const auto *imm = expr.as<IntImmNode>();
    return imm != nullptr && imm->value > 0;
  }

  static bool IsKnownNonNegativeMetalIndex(const PrimExpr &expr) {
    if (!expr.defined() || !IsScalarInteger(expr.dtype())) {
      return false;
    }
    if (expr.dtype().is_uint()) {
      return true;
    }
    if (const auto *imm = expr.as<IntImmNode>()) {
      return imm->value >= 0;
    }
    if (const auto *var = expr.as<VarNode>()) {
      return IsMetalThreadIndexVar(var) ||
             MetalScalarIntrinsicPrefixFromVarName(var->name_hint);
    }
    if (const auto *call = expr.as<CallNode>()) {
      std::string prefix;
      return MetalScalarIntrinsicPrefix(call, &prefix);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      return IsKnownNonNegativeMetalIndex(cast->value);
    }
    if (const auto *add = expr.as<AddNode>()) {
      return IsKnownNonNegativeMetalIndex(add->a) &&
             IsKnownNonNegativeMetalIndex(add->b);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return IsKnownNonNegativeMetalIndex(mul->a) &&
             IsKnownNonNegativeMetalIndex(mul->b);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return IsKnownNonNegativeMetalIndex(div->a) && IsPositiveIntImm(div->b);
    }
    if (const auto *mod = expr.as<ModNode>()) {
      return IsKnownNonNegativeMetalIndex(mod->a) && IsPositiveIntImm(mod->b);
    }
    if (const auto *floordiv = expr.as<FloorDivNode>()) {
      return IsKnownNonNegativeMetalIndex(floordiv->a) &&
             IsPositiveIntImm(floordiv->b);
    }
    if (const auto *floormod = expr.as<FloorModNode>()) {
      return IsKnownNonNegativeMetalIndex(floormod->a) &&
             IsPositiveIntImm(floormod->b);
    }
    return false;
  }

  static bool MetalScalarIntrinsicPrefixFromVarName(const std::string &name) {
    return name == "grid_tid" || name.rfind("grid_tid_", 0) == 0 ||
           name == "threadgroup_tid" ||
           name.rfind("threadgroup_tid_", 0) == 0 ||
           name == "simd_lane" || name.rfind("simd_lane_", 0) == 0;
  }

  static DataType CommonIntegerDType(DataType lhs, DataType rhs) {
    ICHECK(IsScalarInteger(lhs));
    ICHECK(IsScalarInteger(rhs));
    int bits = std::max(lhs.bits(), rhs.bits());
    if (lhs.is_uint() && rhs.is_uint()) {
      return DataType::UInt(bits);
    }
    return DataType::Int(bits);
  }

  static void CastIntegerOperandTo(PrimExpr *expr, DataType dtype) {
    if ((*expr).dtype() != dtype) {
      *expr = Cast(dtype, *expr);
    }
  }

  static void NormalizeIntegerOperands(PrimExpr *lhs, PrimExpr *rhs) {
    if ((*lhs).dtype() == (*rhs).dtype()) {
      return;
    }
    if (!IsScalarInteger((*lhs).dtype()) || !IsScalarInteger((*rhs).dtype())) {
      return;
    }
    PrimExpr coerced;
    if (CoerceIntImmTo(*lhs, (*rhs).dtype(), &coerced)) {
      *lhs = coerced;
      return;
    }
    if (CoerceIntImmTo(*rhs, (*lhs).dtype(), &coerced)) {
      *rhs = coerced;
      return;
    }

    DataType common_dtype = CommonIntegerDType((*lhs).dtype(), (*rhs).dtype());
    CastIntegerOperandTo(lhs, common_dtype);
    CastIntegerOperandTo(rhs, common_dtype);
  }

  PrimExpr VisitExpr_(const AddNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Add(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const SubNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Sub(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const MulNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Mul(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const DivNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Div(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const FloorDivNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (IsKnownNonNegativeMetalIndex(lhs) && IsPositiveIntImm(rhs)) {
      return Div(lhs, rhs, op->span);
    }
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return FloorDiv(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const ModNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Mod(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    NormalizeIntegerOperands(&lhs, &rhs);
    if (IsKnownNonNegativeMetalIndex(lhs) && IsPositiveIntImm(rhs)) {
      return Mod(lhs, rhs, op->span);
    }
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return FloorMod(lhs, rhs, op->span);
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (IsWideningIntegerIndexCast(op)) {
      return value;
    }
    if (value.same_as(op->value)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Cast(op->dtype, value, op->span);
  }
};

ffi::Array<PrimExpr> NormalizeMetalIndexArray(const ffi::Array<PrimExpr> &indices,
                                              bool *changed) {
  ffi::Array<PrimExpr> normalized;
  normalized.reserve(indices.size());
  for (const PrimExpr &index : indices) {
    PrimExpr new_index = MetalIndexCastNormalizer::Normalize(index);
    *changed = *changed || !new_index.same_as(index);
    normalized.push_back(new_index);
  }
  return normalized;
}

class MetalScalarBindCanonicalizer : public StmtExprMutator {
 public:
  Stmt Rewrite(Stmt body) { return VisitStmt(std::move(body)); }

 private:
  using ExprTable =
      std::unordered_map<PrimExpr, Var, ffi::StructuralHash, ExprDeepEqual>;

  Stmt VisitStmt(const Stmt &stmt) final {
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return VisitLetStmt(let);
    }
    return StmtExprMutator::VisitStmt(stmt);
  }

  Stmt VisitLetStmt(const LetStmtNode *op) {
    ExprTable saved_expr_to_var = expr_to_var_;
    std::unordered_map<const VarNode *, PrimExpr> saved_var_remap = var_remap_;

    PrimExpr value = VisitExpr(op->value);
    if (CanAliasBind(op->var, value) || ShouldInlineMetalThreadIndexExpr(value)) {
      var_remap_[op->var.get()] = value;
      Stmt body = VisitStmt(op->body);
      expr_to_var_ = std::move(saved_expr_to_var);
      var_remap_ = std::move(saved_var_remap);
      return body;
    }
    PrimExpr replacement = ExistingReplacement(value);
    if (replacement.defined()) {
      var_remap_[op->var.get()] = replacement;
      Stmt body = VisitStmt(op->body);
      expr_to_var_ = std::move(saved_expr_to_var);
      var_remap_ = std::move(saved_var_remap);
      return body;
    }

    Register(op->var, value);
    Stmt body = VisitStmt(op->body);
    expr_to_var_ = std::move(saved_expr_to_var);
    var_remap_ = std::move(saved_var_remap);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return LetStmt(op->var, value, body, op->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    ExprTable saved_expr_to_var = expr_to_var_;
    std::unordered_map<const VarNode *, PrimExpr> saved_var_remap = var_remap_;

    ffi::Array<Stmt> stmts;
    bool changed = false;
    for (const Stmt &stmt : op->seq) {
      if (const auto *bind = stmt.as<BindNode>()) {
        PrimExpr value = VisitExpr(bind->value);
        if (CanAliasBind(bind->var, value)) {
          var_remap_[bind->var.get()] = value;
          changed = true;
          continue;
        }
        if (ShouldInlineMetalThreadIndexExpr(value)) {
          var_remap_[bind->var.get()] = value;
          changed = true;
          continue;
        }
        PrimExpr replacement = ExistingReplacement(value);
        if (replacement.defined()) {
          var_remap_[bind->var.get()] = replacement;
          changed = true;
          continue;
        }

        Stmt new_bind = Bind(bind->var, value);
        stmts.push_back(new_bind);
        changed = changed || !new_bind.same_as(stmt);
        Register(bind->var, value);
        continue;
      }

      Stmt new_stmt = VisitStmt(stmt);
      stmts.push_back(new_stmt);
      changed = changed || !new_stmt.same_as(stmt);
    }

    expr_to_var_ = std::move(saved_expr_to_var);
    var_remap_ = std::move(saved_var_remap);

    if (!changed) {
      return ffi::GetRef<Stmt>(op);
    }
    if (stmts.empty()) {
      return Evaluate(0);
    }
    if (stmts.size() == 1) {
      return stmts[0];
    }
    return SeqStmt(stmts);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (CanAliasBind(op->var, value)) {
      var_remap_[op->var.get()] = value;
      return Evaluate(0);
    }
    if (ShouldInlineMetalThreadIndexExpr(value)) {
      var_remap_[op->var.get()] = value;
      return Evaluate(0);
    }
    PrimExpr replacement = ExistingReplacement(value);
    if (replacement.defined()) {
      var_remap_[op->var.get()] = replacement;
      return Evaluate(0);
    }
    Register(op->var, value);
    if (value.same_as(op->value)) {
      return ffi::GetRef<Stmt>(op);
    }
    return Bind(op->var, value);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool changed = false;
    ffi::Array<PrimExpr> indices = NormalizeMetalIndexArray(store->indices, &changed);
    if (!changed) {
      return store;
    }
    return BufferStore(store->buffer, store->value, indices, std::nullopt, store->span);
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr rewritten = ExprMutator::VisitExpr(expr);

    if (const auto *var = rewritten.as<VarNode>()) {
      auto it = var_remap_.find(var);
      if (it != var_remap_.end()) {
        return it->second;
      }
      return rewritten;
    }

    PrimExpr replacement = ExistingReplacement(rewritten);
    return replacement.defined() ? replacement : rewritten;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    bool changed = false;
    ffi::Array<PrimExpr> indices = NormalizeMetalIndexArray(load->indices, &changed);
    if (!changed) {
      return load;
    }
    return BufferLoad(load->buffer, indices, std::nullopt, load->span);
  }

  PrimExpr ExistingReplacement(const PrimExpr &expr) const {
    if (!CanRegister(expr)) {
      return PrimExpr();
    }
    auto it = expr_to_var_.find(expr);
    if (it == expr_to_var_.end()) {
      return PrimExpr();
    }
    return it->second;
  }

  void Register(const Var &var, const PrimExpr &value) {
    if (CanRegister(value)) {
      expr_to_var_.emplace(value, var);
    }
  }

  static bool CanRegister(const PrimExpr &expr) {
    return expr.defined() && !IsTrivialScalarExpr(expr) &&
           PureScalarExprChecker::Check(expr);
  }

  static bool CanAliasBind(const Var &var, const PrimExpr &value) {
    const auto *alias = value.as<VarNode>();
    return alias != nullptr && alias != var.get();
  }

  ExprTable expr_to_var_;
  std::unordered_map<const VarNode *, PrimExpr> var_remap_;
};

PrimFunc BindMetalScalarIntrinsicsFunc(PrimFunc f) {
  MetalScalarIntrinsicBinder binder;
  Stmt body = binder.Rewrite(f->body);
  MetalScalarBindCanonicalizer canonicalizer;

  auto canonicalize = [&](Stmt stmt) { return canonicalizer.Rewrite(std::move(stmt)); };

  if (binder.bindings().empty() && body.same_as(f->body)) {
    Stmt canonical_body = canonicalize(f->body);
    if (!canonical_body.same_as(f->body)) {
      f.CopyOnWrite()->body = canonical_body;
    }
    return f;
  }

  if (binder.bindings().empty()) {
    f.CopyOnWrite()->body = canonicalize(body);
    return f;
  }

  ffi::Array<Stmt> stmts;
  for (const Stmt &binding : binder.bindings()) {
    stmts.push_back(binding);
  }
  stmts.push_back(body);
  f.CopyOnWrite()->body = canonicalize(SeqStmt(stmts));
  return f;
}

}  // namespace

using namespace tirx::transform;
tvm::transform::Pass BindMetalScalarIntrinsics() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return BindMetalScalarIntrinsicsFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.BindMetalScalarIntrinsics", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.BindMetalScalarIntrinsics", BindMetalScalarIntrinsics);
}

}  // namespace tl
}  // namespace tvm
