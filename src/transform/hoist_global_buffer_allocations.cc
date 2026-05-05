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
 * \brief Hoist global buffer allocations to the top of the block (host side).
 * \file hoist_global_buffer_allocations.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/utils.h"
#include "common/attr.h"
#include "tirx/transform/ir_utils.h"
#include "tvm/tirx/stmt.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

// CPPMEGA: Collect buffer-data Vars that have a backing definition (AllocBuffer,
// SBlock alloc_buffers/match_buffers, DeclBuffer) and the ghost buffers that
// are referenced via BufferLoad/BufferStore without ever being defined. The
// latter occurs when LayoutInference / LowerTileOp rewrites a fragment buffer
// (e.g. ``C_local``) into a smaller per-thread "local" buffer at the *outer*
// "root" SBlock via a layout_map annotation but never adds a corresponding
// entry to any block's ``alloc_buffers``. Without an explicit Allocate the
// buffer's data Var is undefined and downstream MakePackedAPI rejects the
// PrimFunc with "variables (...) are used, but are not passed in as API
// arguments".
class GhostBufferCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &buf : op->alloc_buffers) {
      defined_vars_.insert(buf->data);
    }
    for (const auto &mb : op->match_buffers) {
      defined_vars_.insert(mb->buffer->data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    defined_vars_.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    defined_vars_.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    RecordBufferUse(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RecordBufferUse(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    if (!op->op.same_as(builtin::tvm_access_ptr()) || op->args.size() < 2) {
      return;
    }
    if (const auto *data = op->args[1].as<VarNode>()) {
      DataType elem_dtype = op->args[0].defined() ? op->args[0].dtype()
                                                  : DataType::Void();
      RecordAccessPtrUse(ffi::GetRef<Var>(data), elem_dtype);
    }
  }

  // Map of buffer-data Var -> Buffer for any buffer ever referenced.
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> used_buffers_;
  // Map of raw tvm_access_ptr data Var -> pointed-to element dtype.
  std::unordered_map<Var, DataType, ObjectPtrHash, ObjectPtrEqual>
      used_access_ptr_vars_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> defined_vars_;

private:
  void RecordBufferUse(const Buffer &buf) {
    if (!buf.defined()) return;
    used_buffers_.emplace(buf->data, buf);
  }

  void RecordAccessPtrUse(const Var &data, DataType elem_dtype) {
    if (ambiguous_access_ptr_vars_.count(data)) {
      return;
    }
    auto it = used_access_ptr_vars_.find(data);
    if (it == used_access_ptr_vars_.end()) {
      used_access_ptr_vars_.emplace(data, elem_dtype);
    } else if (it->second.is_void()) {
      it->second = elem_dtype;
    } else if (elem_dtype.is_void()) {
      return;
    } else if (it->second != elem_dtype) {
      used_access_ptr_vars_.erase(it);
      ambiguous_access_ptr_vars_.insert(data);
    }
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      ambiguous_access_ptr_vars_;
};

bool PrimExprDeepEqual(const PrimExpr &lhs, const PrimExpr &rhs) {
  if (lhs.defined() != rhs.defined()) {
    return false;
  }
  return !lhs.defined() || ExprDeepEqual()(lhs, rhs);
}

bool PrimExprArrayDeepEqual(const ffi::Array<PrimExpr> &lhs,
                            const ffi::Array<PrimExpr> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!PrimExprDeepEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

bool IntImmArrayDeepEqual(const ffi::Array<IntImm> &lhs,
                          const ffi::Array<IntImm> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!ExprDeepEqual()(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

bool MatchesCanonicalInputBuffer(const Buffer &buffer,
                                 const Buffer &canonical) {
  if (!buffer.defined() || !canonical.defined()) {
    return false;
  }
  if (!IsGlobalBuffer(buffer) || !IsGlobalBuffer(canonical)) {
    return false;
  }
  if (buffer->data->name_hint != canonical->data->name_hint ||
      buffer->data.dtype() != canonical->data.dtype()) {
    return false;
  }
  if (buffer->name != canonical->name || buffer->dtype != canonical->dtype ||
      buffer->data_alignment != canonical->data_alignment ||
      buffer->offset_factor != canonical->offset_factor ||
      buffer->buffer_type != canonical->buffer_type) {
    return false;
  }
  return PrimExprArrayDeepEqual(buffer->shape, canonical->shape) &&
         PrimExprArrayDeepEqual(buffer->strides, canonical->strides) &&
         PrimExprDeepEqual(buffer->elem_offset, canonical->elem_offset) &&
         IntImmArrayDeepEqual(buffer->axis_separators,
                              canonical->axis_separators);
}

Buffer FindCanonicalInputBuffer(const Buffer &buffer,
                                const ffi::Map<Var, Buffer> &buffer_map) {
  Buffer match;
  for (const auto &kv : buffer_map) {
    const Buffer &candidate = kv.second;
    if (!MatchesCanonicalInputBuffer(buffer, candidate)) {
      continue;
    }
    if (match.defined()) {
      // Multiple identical parameter buffers are ambiguous. Leave the IR
      // unchanged so the later UndefinedVars check reports the real problem.
      return Buffer();
    }
    match = candidate;
  }
  return match;
}

bool MatchesCanonicalInputDataVar(const Var &data, DataType elem_dtype,
                                  const Buffer &canonical) {
  if (!canonical.defined() || !IsGlobalBuffer(canonical)) {
    return false;
  }
  if (data->name_hint != canonical->data->name_hint ||
      data.dtype() != canonical->data.dtype()) {
    return false;
  }
  if (!elem_dtype.is_void() && elem_dtype != canonical->dtype) {
    return false;
  }
  return true;
}

Buffer FindCanonicalInputBufferForDataVar(
    const Var &data, DataType elem_dtype,
    const ffi::Map<Var, Buffer> &buffer_map) {
  Buffer match;
  for (const auto &kv : buffer_map) {
    const Buffer &candidate = kv.second;
    if (!MatchesCanonicalInputDataVar(data, elem_dtype, candidate)) {
      continue;
    }
    if (match.defined()) {
      return Buffer();
    }
    match = candidate;
  }
  return match;
}

class InputBufferCanonicalizer : public StmtExprMutator {
public:
  explicit InputBufferCanonicalizer(
      const std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>
          &canonical_buffers)
      : canonical_buffers_(canonical_buffers) {}

private:
  Buffer VisitBufferUse(const Buffer &buffer) final {
    if (buffer.defined()) {
      auto it = canonical_buffers_.find(buffer->data);
      if (it != canonical_buffers_.end()) {
        return it->second;
      }
    }
    return StmtExprMutator::VisitBufferUse(buffer);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(builtin::tvm_access_ptr()) || op->args.size() < 2) {
      return StmtExprMutator::VisitExpr_(op);
    }

    ffi::Array<PrimExpr> args;
    for (size_t i = 0; i < op->args.size(); ++i) {
      PrimExpr arg = this->VisitExpr(op->args[i]);
      if (i == 1) {
        if (const auto *var = arg.as<VarNode>()) {
          auto it = canonical_buffers_.find(ffi::GetRef<Var>(var));
          if (it != canonical_buffers_.end()) {
            arg = it->second->data;
          }
        }
      }
      args.push_back(arg);
    }
    return Call(op->dtype, op->op, args);
  }

  const std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>
      &canonical_buffers_;
};

class GhostBufferPlacementPlanner : public StmtExprVisitor {
public:
  explicit GhostBufferPlacementPlanner(const ffi::Array<Buffer> &ghost_buffers) {
    for (const Buffer &buf : ghost_buffers) {
      ghost_buffers_.emplace(buf->data, buf);
    }
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (!is_one(op->predicate)) {
      this->VisitExpr(op->predicate);
    }
    block_stack_.push_back(op->block.get());
    this->VisitStmt(op->block->body);
    block_stack_.pop_back();
  }

  void VisitStmt_(const SBlockNode *op) final {
    block_stack_.push_back(op);
    this->VisitStmt(op->body);
    block_stack_.pop_back();
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    RecordBufferUse(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RecordBufferUse(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() >= 2) {
      if (const auto *data = op->args[1].as<VarNode>()) {
        RecordBufferVarUse(ffi::GetRef<Var>(data));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void BuildPlacements() {
    for (const auto &kv : owner_paths_) {
      const std::vector<const SBlockNode *> &path = kv.second;
      if (path.empty()) {
        continue;
      }
      const SBlockNode *owner = path.back();
      // A non-global ghost buffer allocated at host root is not a valid
      // device-local allocation; leave it visible to downstream diagnostics.
      if (IsHostMainBlock(owner)) {
        continue;
      }
      placements_[owner].push_back(ghost_buffers_.at(kv.first));
    }
  }

  std::unordered_map<const SBlockNode *, ffi::Array<Buffer>> placements_;

private:
  void RecordBufferUse(const Buffer &buf) {
    if (!buf.defined()) return;
    RecordBufferVarUse(buf->data);
  }

  void RecordBufferVarUse(const Var &var) {
    if (!ghost_buffers_.count(var) || block_stack_.empty()) {
      return;
    }
    auto it = owner_paths_.find(var);
    if (it == owner_paths_.end()) {
      owner_paths_.emplace(var, block_stack_);
      return;
    }
    std::vector<const SBlockNode *> &path = it->second;
    size_t common = 0;
    size_t limit = std::min(path.size(), block_stack_.size());
    while (common < limit && path[common] == block_stack_[common]) {
      ++common;
    }
    path.resize(common);
  }

  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> ghost_buffers_;
  std::unordered_map<Var, std::vector<const SBlockNode *>, ObjectPtrHash,
                     ObjectPtrEqual>
      owner_paths_;
  std::vector<const SBlockNode *> block_stack_;
};

class GlobalBufferAllocationsHoister : public StmtMutator {
public:
  explicit GlobalBufferAllocationsHoister(
      std::unordered_map<const SBlockNode *, ffi::Array<Buffer>>
          ghost_buffer_placements)
      : ghost_buffer_placements_(std::move(ghost_buffer_placements)) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    auto node = Downcast<SBlock>(StmtMutator::VisitStmt_(op));

    if (IsHostMainBlock(op)) {
      for (const auto &buf : global_buffers_) {
        node.CopyOnWrite()->alloc_buffers.push_back(buf);
      }
    } else {
      ffi::Array<Buffer> new_alloc_buffers;
      for (const auto &buf : op->alloc_buffers) {
        if (IsGlobalBuffer(buf)) {
          global_buffers_.push_back(buf);
        } else {
          new_alloc_buffers.push_back(buf);
        }
      }
      node.CopyOnWrite()->alloc_buffers = std::move(new_alloc_buffers);
    }

    auto placement_it = ghost_buffer_placements_.find(op);
    if (placement_it != ghost_buffer_placements_.end()) {
      for (const auto &buf : placement_it->second) {
        node.CopyOnWrite()->alloc_buffers.push_back(buf);
      }
    }

    return node;
  }

  ffi::Array<Buffer> global_buffers_;

private:
  std::unordered_map<const SBlockNode *, ffi::Array<Buffer>>
      ghost_buffer_placements_;
};

PrimFunc HoistGlobalBufferAllocations(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  // CPPMEGA: Collect ghost buffers (referenced via BufferLoad/Store but never
  // defined by any Alloc/Decl/match site or as a function parameter handle).
  auto collect_buffers = [&]() {
    GhostBufferCollector collector;
    collector(fptr->body);
    for (const auto &kv : fptr->buffer_map) {
      if (kv.second.defined()) {
        collector.defined_vars_.insert(kv.second->data);
      }
    }
    return collector;
  };

  // Function parameter Vars are also "defined" for the purposes of binding.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> param_vars(
      fptr->params.begin(), fptr->params.end());

  GhostBufferCollector collector = collect_buffers();
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>
      canonical_buffers;
  for (const auto &kv : collector.used_buffers_) {
    const Var &v = kv.first;
    const Buffer &buf = kv.second;
    if (collector.defined_vars_.count(v) || param_vars.count(v)) continue;

    Buffer canonical = FindCanonicalInputBuffer(buf, fptr->buffer_map);
    if (canonical.defined()) {
      canonical_buffers.emplace(v, canonical);
    }
  }
  for (const auto &kv : collector.used_access_ptr_vars_) {
    const Var &v = kv.first;
    DataType elem_dtype = kv.second;
    if (collector.defined_vars_.count(v) || param_vars.count(v)) continue;
    if (canonical_buffers.count(v)) continue;

    Buffer canonical =
        FindCanonicalInputBufferForDataVar(v, elem_dtype, fptr->buffer_map);
    if (canonical.defined()) {
      canonical_buffers.emplace(v, canonical);
    }
  }

  if (!canonical_buffers.empty()) {
    fptr->body = InputBufferCanonicalizer(canonical_buffers)(fptr->body);
    collector = collect_buffers();
  }

  ffi::Array<Buffer> ghost_buffers;
  for (const auto &kv : collector.used_buffers_) {
    const Var &v = kv.first;
    const Buffer &buf = kv.second;
    if (collector.defined_vars_.count(v)) continue;
    if (param_vars.count(v)) continue;
    // Only hoist non-global, statically-shaped buffers. Global buffers are
    // expected to be match-bound from function parameters.
    if (IsGlobalBuffer(buf)) continue;
    bool all_static = true;
    for (const auto &dim : buf->shape) {
      if (!dim.as<IntImmNode>()) {
        all_static = false;
        break;
      }
    }
    if (!all_static) continue;
    ghost_buffers.push_back(buf);
  }

  GhostBufferPlacementPlanner planner(ghost_buffers);
  planner(fptr->body);
  planner.BuildPlacements();

  GlobalBufferAllocationsHoister hoister(std::move(planner.placements_));
  fptr->body = hoister(fptr->body);
  return func;
}

namespace transform {

Pass HoistGlobalBufferAllocations() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ::tvm::tl::HoistGlobalBufferAllocations(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HoistGlobalBufferAllocations",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistGlobalBufferAllocations",
                        HoistGlobalBufferAllocations);
}

} // namespace transform

} // namespace tl
} // namespace tvm
