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
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>

#include <unordered_map>
#include <unordered_set>

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

  // Map of buffer-data Var -> Buffer for any buffer ever referenced.
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> used_buffers_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> defined_vars_;

private:
  void RecordBufferUse(const Buffer &buf) {
    if (!buf.defined()) return;
    used_buffers_.emplace(buf->data, buf);
  }
};

class GlobalBufferAllocationsHoister : public StmtMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    auto node = Downcast<SBlock>(StmtMutator::VisitStmt_(op));

    if (IsHostMainBlock(op)) {
      for (const auto &buf : global_buffers_) {
        node.CopyOnWrite()->alloc_buffers.push_back(buf);
      }
      // CPPMEGA: Also materialize ghost local buffers (used but never
      // allocated) here so MakePackedAPI's UndefinedVars check succeeds.
      for (const auto &buf : ghost_buffers_) {
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

    return node;
  }

  ffi::Array<Buffer> global_buffers_;
  ffi::Array<Buffer> ghost_buffers_;
};

PrimFunc HoistGlobalBufferAllocations(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  // CPPMEGA: Collect ghost buffers (referenced via BufferLoad/Store but never
  // defined by any Alloc/Decl/match site or as a function parameter handle).
  GhostBufferCollector collector;
  collector(fptr->body);
  // Function parameter Vars are also "defined" for the purposes of binding.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> param_vars(
      fptr->params.begin(), fptr->params.end());
  // Buffer data Vars exposed via buffer_map (match_buffer on params) are
  // defined too.
  for (const auto &kv : fptr->buffer_map) {
    if (kv.second.defined()) {
      collector.defined_vars_.insert(kv.second->data);
    }
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

  GlobalBufferAllocationsHoister hoister;
  hoister.ghost_buffers_ = std::move(ghost_buffers);
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
