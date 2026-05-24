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
 * \file metal_simdgroup_guard.cc
 * \brief Fail-closed legality checks for CUDA/HIP warp intrinsics on Metal.
 */

#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>

namespace tvm {
namespace tl {
namespace transform {
namespace {

using namespace tirx;
using namespace tirx::transform;

constexpr int64_t kFullWarpMask = 0xffffffffLL;
constexpr int64_t kMetalSimdgroupWidth = 32;
constexpr int64_t kGemmWarpPolicyFullRow = 1;
constexpr int64_t kGemmWarpPolicyFullCol = 2;
constexpr int64_t kGemmWarpPolicyFree = 3;

bool IsMetalTarget(const Target &target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "metal";
}

std::string OpName(const CallNode *op) {
  if (const auto *op_node = op->op.as<OpNode>()) {
    return op_node->name;
  }
  return "";
}

const char *UnsupportedIntrinsicReason(const std::string &name) {
  if (name == "tl.shfl_sync") {
    return "absolute-lane CUDA warp broadcast";
  }
  if (name == "tl.shfl_down_sync") {
    return "CUDA warp down-shuffle";
  }
  if (name == "tl.shfl_up_sync") {
    return "CUDA warp up-shuffle";
  }
  if (name == "tl.sync_warp") {
    return "CUDA/HIP warp-scope synchronization";
  }
  if (name == "tir.tvm_warp_shuffle" || name == "tir.tvm_warp_shuffle_up" ||
      name == "tir.tvm_warp_shuffle_down") {
    return "TVM warp-shuffle intrinsic";
  }
  return nullptr;
}

bool ConstInt(const PrimExpr &expr, int64_t *out) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    *out = imm->value;
    return true;
  }
  return false;
}

const char *GemmWarpPolicyName(int64_t policy) {
  switch (policy) {
  case kGemmWarpPolicyFullRow:
    return "GemmWarpPolicy.FullRow";
  case kGemmWarpPolicyFullCol:
    return "GemmWarpPolicy.FullCol";
  case kGemmWarpPolicyFree:
    return "GemmWarpPolicy.Free";
  default:
    return nullptr;
  }
}

void Reject(const std::string &func_name, const std::string &op_name,
            const std::string &reason) {
  TVM_FFI_THROW(ValueError)
      << "Metal SIMDgroup guard rejected " << op_name << " in " << func_name
      << ": " << reason
      << ". Use a Metal-specific simdgroup primitive or route this schedule to "
         "CUDA/HIP.";
}

class MetalSimdgroupGuard final : public StmtExprVisitor {
 public:
  explicit MetalSimdgroupGuard(std::string func_name)
      : func_name_(std::move(func_name)) {}

  void VisitExpr_(const CallNode *op) final {
    const std::string name = OpName(op);
    if (const char *reason = UnsupportedIntrinsicReason(name)) {
      Reject(func_name_, name, reason);
    }
    if (name == "tl.shfl_xor_sync") {
      ValidateShflXor(op);
    }
    if (name == "tl.tileop.gemm") {
      ValidateTileGemm(op);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

 private:
  void ValidateShflXor(const CallNode *op) const {
    if (op->args.size() != 4U) {
      Reject(func_name_, "tl.shfl_xor_sync",
             "malformed intrinsic; expected <mask, value, lane_mask, width>");
    }
    int64_t mask = 0;
    int64_t width = 0;
    if (!ConstInt(op->args[0], &mask) || !ConstInt(op->args[3], &width) ||
        mask != kFullWarpMask || width != kMetalSimdgroupWidth) {
      Reject(func_name_, "tl.shfl_xor_sync",
             "Metal lowering only preserves full-simdgroup semantics "
             "(mask=0xFFFFFFFF, width=32)");
    }
  }

  void ValidateTileGemm(const CallNode *op) const {
    if (op->args.size() <= 8U) {
      return;
    }
    int64_t policy = 0;
    if (!ConstInt(op->args[8], &policy)) {
      return;
    }
    const char *policy_name = GemmWarpPolicyName(policy);
    if (policy_name == nullptr) {
      return;
    }
    Reject(func_name_, "tl.tileop.gemm",
           std::string(policy_name) +
               " encodes a CUDA/HIP warp-partition policy. Metal schedules "
               "must use target-native simdgroup GEMM lowering or an explicit "
               "Metal policy instead of reusing a warp-policy tile GEMM");
  }

  std::string func_name_;
};

}  // namespace

tvm::transform::Pass MetalSimdgroupSemanticGuard() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    Target target;
    if (auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget)) {
      target = opt_target.value();
    } else {
      return f;
    }
    if (!IsMetalTarget(target)) {
      return f;
    }

    std::string func_name = "main";
    for (const auto &kv : m->functions) {
      if (kv.second.same_as(f)) {
        func_name = kv.first->name_hint;
        break;
      }
    }
    MetalSimdgroupGuard(std::move(func_name))(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MetalSimdgroupSemanticGuard", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MetalSimdgroupSemanticGuard",
                        MetalSimdgroupSemanticGuard);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
