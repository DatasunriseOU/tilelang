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
 * \file lower_device_storage_access.cc
 * \brief Lower the special device storage access.
 */
#include "vendored/target_info.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "runtime/thread_storage_scope.h"
#include "tirx/transform/ir_utils.h"
#include "vendored/let_stmt.h"

namespace tvm {
namespace tl {
using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

using runtime::StorageRank;
using runtime::StorageScope;

class StorageAccessInfoLower : public StmtExprMutator {
public:
  // CPPMEGA: apache/tvm latest replaced AllocateNode with AllocBufferNode and
  // dropped the body field. The original pass also stripped DeclBuffer when
  // the storage info had no head_address; with no `body` to return we instead
  // emit an Evaluate(0) stub so the surrounding SeqStmt can absorb the slot.
  Stmt VisitStmt_(const AllocBufferNode *op) final {
    const Buffer &buf = op->buffer;
    auto scope = StorageScope::Create(GetPtrStorageScope(buf->data));
    if (!scope.tag.empty() && scope.tag != ".dyn" && scope.tag != ".var" &&
        scope.tag != ".barrier" && scope.tag != ".cluster_barrier" &&
        scope.tag != ".fragment" && scope.tag.find(".descriptor") != 0) {
      auto info = GetMemoryInfo(GetPtrStorageScope(buf->data));
      ICHECK(info.defined())
          << "Cannot find memory info of " << scope.to_string();
      ICHECK(storage_info_.find(buf->data.get()) == storage_info_.end())
          << "Double allocation of " << scope.to_string();
      storage_info_[buf->data.get()] = info;

      // Lower allocate to device allocate when needed.
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      const auto *new_op = stmt.as<AllocBufferNode>();
      const Buffer &new_buf = new_op ? new_op->buffer : buf;
      if (info->head_address.defined()) {
        // CPPMEGA: this pass runs in `host_codegen`/`device_codegen`
        // AFTER the `LowerTileLangLetStmt` converter, then is followed by
        // apache `tir.transform.Simplify` and target-specific codegen which
        // dispatch on tirx node types via `StmtFunctor` vtable. Emitting a
        // vendored `tilelang::tl_tir::LetStmt` here would crash apache's
        // visitor with `NodeFunctor calls un-registered function`. Use the
        // apache-equivalent `SeqStmt({Bind(var, value), Evaluate(0)})`
        // directly — see split_host_device.cc:385 for the same pattern.
        return SeqStmt({Bind(new_buf->data, info->head_address), Evaluate(0)});
      } else {
        return Evaluate(0);
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    if (auto it = storage_info_.find(node->buffer->data.get());
        it != storage_info_.end() && !it->second->head_address.defined()) {
      // CPPMEGA: DeclBuffer no longer has a body; emit a no-op stub.
      return Evaluate(0);
    } else {
      return std::move(node);
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      return MakeAccessPtr(op);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

private:
  // tvm_access_ptr
  PrimExpr MakeAccessPtr(const CallNode *op) {
    // Specially handle the buffer packed intrinsic
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CallNode>();
    ICHECK_EQ(op->args.size(), 5U);
    DataType dtype = op->args[0].dtype();
    const VarNode *buffer = op->args[1].as<VarNode>();
    Var buffer_var = Downcast<Var>(op->args[1]);
    PrimExpr offset = op->args[2];
    auto it = storage_info_.find(buffer);
    if (it != storage_info_.end() && it->second.defined()) {
      return MakeTaggedAccessPtr(op->dtype, buffer_var, dtype, offset,
                                 it->second);
    }
    ICHECK(op->dtype.is_handle());
    // Change to address_of
    return AddressOffset(buffer_var, dtype, offset);
  }

  PrimExpr MakeTaggedAccessPtr(DataType ptr_type, const Var &buffer_var,
                               DataType dtype, const PrimExpr &offset,
                               const MemoryInfo &info) {
    if (ptr_type.is_handle()) {
      ICHECK(info->head_address.defined())
          << buffer_var << " is not adddressable.";
      return AddressOffset(buffer_var, dtype, offset);
    }
    int dtype_bits = dtype.bits() * dtype.lanes();
    ICHECK_EQ(info->unit_bits % dtype_bits, 0);
    return cast(
        ptr_type,
        analyzer_.Simplify(
            offset / make_const(offset.dtype(), info->unit_bits / dtype_bits)));
  }
  // The storage scope of each buffer
  std::unordered_map<const VarNode *, MemoryInfo> storage_info_;
  // analyzer
  arith::Analyzer analyzer_;
};

Stmt LowerStorageAccessInfo(Stmt stmt) {
  return StorageAccessInfoLower()(std::move(stmt));
}

namespace transform {
using namespace tirx::transform;

Pass LowerDeviceStorageAccessInfo() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = StorageAccessInfoLower()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerDeviceStorageAccessInfo",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerDeviceStorageAccessInfo",
                        LowerDeviceStorageAccessInfo);
}

} // namespace transform
} // namespace tl
} // namespace tvm
