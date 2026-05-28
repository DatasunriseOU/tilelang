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
 * \file storage_rewrite.cc
 * \brief Memory access pattern analysis and optimization.
 *  Re-write data access to enable memory sharing when possible.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/type.h>
#include "vendored/target_info.h"
#include "vendored/tl_attr.h"
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <list>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"
#include "arith/int_operator.h"
#include "runtime/thread_storage_scope.h"
#include "tirx/ir/buffer_common.h"
#include "tirx/transform/ir_utils.h"
#include "vendored/let_stmt.h"

namespace tvm {
namespace tl {

using runtime::StorageRank;
using runtime::StorageScope;
using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

namespace {

// Metal M5 cooperative tensor buffers are register tiles owned by
// codegen_metal; the generic storage-rewrite pass must skip any function
// that allocates one. See PR tile-ai/tilelang#2252.
class MetalCooperativeTensorScopeDetector : public StmtExprVisitor {
public:
  static bool Detect(const PrimFunc &func) {
    MetalCooperativeTensorScopeDetector detector;
    detector(func->body);
    return detector.found_;
  }

private:
  void VisitStmt_(const AllocBufferNode *op) final {
    if (op->buffer.scope() == "metal.cooperative_tensor") {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      if (buffer.scope() == "metal.cooperative_tensor") {
        found_ = true;
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

} // namespace

/*!
 * \brief Perform data type legalization on the given BufferLoadNode pointer.
 * Equal to BufferLoadNode::LegalizeDType, but operates on a pointer.
 * \param n A pointer to a writable BufferLoadNode.
 */
static void LegalizeBufferLoadDType(BufferLoadNode *n) {
  // Check that all indices except the last one have a scalar dtype
  for (int i = 0; i < static_cast<int>(n->indices.size()) - 1; i++) {
    ICHECK(n->indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  // If there are no indices, set the dtype to the buffer's dtype
  if (n->indices.empty()) {
    n->dtype = n->buffer->dtype;
  } else {
    auto index_dtype = n->indices.back().dtype();
    bool is_buffer_dtype_scalable = n->buffer->dtype.is_scalable_vector();
    bool is_index_scalable = index_dtype.is_scalable_vector();

    // Do not allow both index dtype and buffer dtype to be scalable vectors
    ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
        << "Index dtype and buffer dtype cannot both be scalable.";

    if (is_index_scalable) {
      // Index is a scalable vector, while the buffer is not
      n->dtype = n->buffer->dtype.with_scalable_vscale_factor(
          index_dtype.vscale_factor() * n->buffer->dtype.lanes());
    } else if (is_buffer_dtype_scalable) {
      // The buffer is a scalable vector, while the index is not
      n->dtype = n->buffer->dtype.with_scalable_vscale_factor(
          n->buffer->dtype.vscale_factor() * index_dtype.lanes());
    } else {
      // Neither side is a scalable vector, multiply lanes
      n->dtype = n->buffer->dtype.with_lanes(index_dtype.lanes() *
                                             n->buffer->dtype.lanes());
    }
  }
}

/*!
 * \brief collect the mapping from the buffer var to its allocate
 */
class AllocateCollector : public StmtExprVisitor {
private:
  bool IsDynamicSharedMemory(Var buffer_var) {
    StorageScope storage_scope = runtime::StorageScope::Create(
        GetPtrStorageScope(std::move(buffer_var)));
    return storage_scope.rank == runtime::StorageRank::kShared &&
           storage_scope.tag == ".dyn";
  }

  bool IsStaticSharedMemory(Var buffer_var) {
    StorageScope storage_scope = runtime::StorageScope::Create(
        GetPtrStorageScope(std::move(buffer_var)));
    return storage_scope.rank == runtime::StorageRank::kShared &&
           storage_scope.tag.empty();
  }

public:
  // CPPMEGA: vendored TileLang Allocate is not in apache StmtFunctor dispatch.
  void VisitStmt_(const AllocateNode *op) {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      dyn_shmem_allocs_[op->buffer_var.get()] = op;
    } else if (IsStaticSharedMemory(op->buffer_var)) {
      static_shmem_allocs_[op->buffer_var.get()] = op;
    }
    this->VisitStmt(op->body);
  }
  void VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<AllocateNode>()) {
      VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }
  // The dynamic mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode *, const AllocateNode *> dyn_shmem_allocs_;
  // The static mapping from the original buffer var to its allocate
  std::unordered_map<const VarNode *, const AllocateNode *>
      static_shmem_allocs_;
};

// Find a linear pattern of storage access
// Used for liveness analysis.
// Composite scopes(loop/thread_launch/IfThen) is represented by two points:
// before_scope -> scope_body -> after_scope
//
// The linear_seq_ stores before_scope and after_scope.
// The access to the arrays are stored at the after_scope point.
//
// Define "scope" as the body of For/thread_launch/IfThenElse
// This pass tries to detect last point that we need to keep memory
// alive under the same scope as allocate.
// The storage need to be kept alive between allocate and last access.
// The free point is only inserted at the same scope of allocate.
//
struct AllocationRef {
  const AllocateNode *legacy_alloc{nullptr};
  const AllocBufferNode *alloc_buffer{nullptr};

  static AllocationRef Legacy(const AllocateNode *op) {
    return AllocationRef{op, nullptr};
  }

  static AllocationRef Apache(const AllocBufferNode *op) {
    return AllocationRef{nullptr, op};
  }

  explicit operator bool() const {
    return legacy_alloc != nullptr || alloc_buffer != nullptr;
  }

  const Var &buffer_var() const {
    if (legacy_alloc != nullptr) {
      return legacy_alloc->buffer_var;
    }
    ICHECK(alloc_buffer != nullptr);
    return alloc_buffer->buffer->data;
  }

  DataType dtype() const {
    if (legacy_alloc != nullptr) {
      return legacy_alloc->dtype;
    }
    ICHECK(alloc_buffer != nullptr);
    return alloc_buffer->buffer->dtype;
  }

  const Array<PrimExpr> &extents() const {
    if (legacy_alloc != nullptr) {
      return legacy_alloc->extents;
    }
    ICHECK(alloc_buffer != nullptr);
    return alloc_buffer->buffer->shape;
  }

  const Map<String, ffi::Any> &annotations() const {
    if (legacy_alloc != nullptr) {
      return legacy_alloc->annotations;
    }
    ICHECK(alloc_buffer != nullptr);
    return alloc_buffer->annotations;
  }

  Span span() const {
    if (legacy_alloc != nullptr) {
      return legacy_alloc->span;
    }
    ICHECK(alloc_buffer != nullptr);
    return alloc_buffer->span;
  }

  bool is_volatile() const {
    return alloc_buffer != nullptr &&
           alloc_buffer->annotations.count(tirx::attr::kVolatile);
  }

  int64_t ConstantAllocationSize() const {
    int64_t result = 1;
    for (const PrimExpr &extent : extents()) {
      if (const auto *imm = extent.as<IntImmNode>()) {
        result *= imm->value;
      } else {
        return -1;
      }
    }
    return result;
  }

  Buffer MakeBuffer(const Var &data, DataType alloc_type,
                    Array<PrimExpr> shape) const {
    if (alloc_buffer != nullptr) {
      Buffer buf = alloc_buffer->buffer;
      auto writer = buf.CopyOnWrite();
      writer->data = data;
      writer->dtype = alloc_type;
      writer->shape = std::move(shape);
      writer->name = data->name_hint;
      return buf;
    }
    ICHECK(legacy_alloc != nullptr);
    return Buffer(/*data=*/data, /*dtype=*/alloc_type, /*shape=*/std::move(shape),
                  /*strides=*/{}, /*elem_offset=*/PrimExpr(),
                  /*name=*/data->name_hint, /*data_alignment=*/0,
                  /*offset_factor=*/0, /*buffer_type=*/BufferType::kDefault,
                  /*axis_separators=*/{}, span());
  }

  const DeclBufferNode *LegacyDeclBuffer() const {
    if (legacy_alloc == nullptr) {
      return nullptr;
    }
    return legacy_alloc->body.as<DeclBufferNode>();
  }
};

class LinearAccessPatternFinder final : public StmtExprVisitor {
public:
  /*! \brief record the touch hist of statement. */
  struct StmtEntry {
    // The statement
    const Object *stmt{};
    // The index in the linear_seq_ to point to end of the nested scope.
    // This is only set to non-zero if stmt is a nested scope.
    // if offset > 0, means this is the begin, the end entry is current_index +
    // offset if offset < 0, means this is the end, the begin entry is
    // current_index + offset
    int64_t scope_pair_offset{0};
    // The buffer variables this statement touched.
    std::vector<const VarNode *> touched;
  };
  // The scope of each allocation
  struct AllocEntry {
    // The physical dimension of the allocation.
    size_t num_physical_dimensions{0};
    // scope level
    size_t level{0};
    // allocation stmt
    AllocationRef alloc;
  };

  void VisitStmt_(const AllocateNode *op) {
    size_t level = scope_.size();
    const VarNode *buf = op->buffer_var.get();

    AllocEntry entry;
    entry.alloc = AllocationRef::Legacy(op);
    entry.level = level;
    // Since StorageRewrite occurs after StorageFlatten/FlattenBuffer,
    // all allocations specify the extent of physical dimensions, and
    // is 1 for flat memory spaces.
    entry.num_physical_dimensions = op->extents.size();
    alloc_info_[buf] = entry;

    this->VisitStmt(op->body);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    size_t level = scope_.size();
    const VarNode *buf = op->buffer->data.get();

    AllocEntry entry;
    entry.alloc = AllocationRef::Apache(op);
    entry.level = level;
    entry.num_physical_dimensions = op->buffer->shape.size();
    alloc_info_[buf] = entry;

    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    all_buffers_accessed_.insert(op->buffer.get());

    // Add write access.
    const VarNode *buffer_var = op->buffer->data.get();
    auto it = alloc_info_.find(buffer_var);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size());
      scope_[it->second.level].touched.push_back(buffer_var);

      ICHECK_EQ(op->buffer->axis_separators.size() + 1,
                it->second.num_physical_dimensions)
          << "Buffer " << op->buffer->name << " is allocated with "
          << it->second.num_physical_dimensions
          << " physical dimensions, but is accessed as having "
          << op->buffer->axis_separators.size() + 1 << " physical dimensions"
          << '\n';
    }
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // Add write access.
    StmtExprVisitor::VisitExpr_(op);

    all_buffers_accessed_.insert(op->buffer.get());

    const VarNode *buffer_var = op->buffer->data.get();
    auto it = alloc_info_.find(buffer_var);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size())
          << "Load memory in places other than store.";
      scope_[it->second.level].touched.push_back(buffer_var);

      ICHECK_EQ(op->buffer->axis_separators.size() + 1,
                it->second.num_physical_dimensions)
          << "Buffer " << op->buffer->name << " is allocated with "
          << it->second.num_physical_dimensions
          << " physical dimensions, but is accessed as having "
          << op->buffer->axis_separators.size() + 1 << " physical dimensions"
          << '\n';
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    scope_.push_back(StmtEntry());
    // visit subexpr
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitStmt_(const BindNode *op) final {
    scope_.push_back(StmtEntry());
    // Apache tirx::Bind can carry BufferLoad values; track them as a statement
    // scope so local reuse checks do not see the load at allocation depth.
    StmtExprVisitor::VisitStmt_(op);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const VarNode *buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      ICHECK_LT(it->second.level, scope_.size()) << " buf=" << buf->name_hint;
      scope_[it->second.level].touched.push_back(buf);
    }
  }

  template <typename T> void VisitNewScope(const T *op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    StmtExprVisitor::VisitStmt_(op);
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);
    // record the pointer to end index.
    ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    // Only record the outer most thread extent.
    if (op->attr_key == tirx::attr::thread_extent && !in_thread_env_) {
      in_thread_env_ = true;
      VisitNewScope(op);
      in_thread_env_ = false;
    } else if (op->attr_key == tirx::attr::extern_scope) {
      VisitNewScope(op);
    } else if (op->attr_key == s_tir::attr::virtual_thread) {
      VisitNewScope(op);
    } else if (op->attr_key == tl::attr::kLexicalAllocScope) {
      VisitNewScope(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final { VisitNewScope(op); }

  void VisitStmt_(const ForNode *op) final { VisitNewScope(op); }

  void VisitStmt_(const WhileNode *op) final { VisitNewScope(op); }

  void VisitStmt_(const AssertStmtNode *op) final { VisitNewScope(op); }

  // apache/tvm StmtFunctor vtable does not dispatch to vendored
  // `tilelang::tl_tir::LetStmtNode`; intercept via top-level VisitStmt.
  void VisitStmt(const Stmt &n) override {
    if (const auto *op = n.as<LetStmtNode>()) {
      scope_.push_back(StmtEntry());
      StmtEntry e;
      e.stmt = op;
      int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
      linear_seq_.push_back(e);
      // Manually traverse value/body since base vtable can't dispatch.
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
      e.touched = std::move(scope_.back().touched);
      scope_.pop_back();
      int64_t end_index = static_cast<int64_t>(linear_seq_.size());
      ICHECK_GT(end_index, begin_index);
      e.scope_pair_offset = begin_index - end_index;
      linear_seq_.push_back(e);
      ICHECK_NE(end_index, 0U);
      linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
      return;
    }
    if (const auto *op = n.as<AllocateNode>()) {
      VisitStmt_(op);
      return;
    }
    StmtExprVisitor::VisitStmt(n);
  }

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode *, AllocEntry> alloc_info_;
  // A record of which Buffer objects have been accessed, to prune
  // unused DeclBuffer instances.
  std::unordered_set<const BufferNode *> all_buffers_accessed_;

private:
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
};

// Verify if the statement can be run safely via inplace fashion
//
// Detect pattern: dst[index] = f(src[index])
//
// WARNING: the current detection algorithm cannot handle the case
// when a location in an array is written multiple times
//
// For example, the following program will pass the check,
// but we cannot make A and B to be the same array.
//
//  A[0] = B[0] + 1
//  A[0] = B[0] + 1
//
// The high level code generator needs to ensure that the generated
// code only write each location of the target array once.
//
// This is the case with IR generated by the current compute schedule.
// We explicitly return false if we find there is an extern block
// which can be arbitrary IR.
//
// Neve-the-less, inplace detector should be used with care in mind.
// We may also consider introduce a condition checker that checks
// if every index only visited once for an absolute sufficient condition.
//
// The code after inplace transformation is no longer idempotent.
//
class InplaceOpVerifier : public StmtExprVisitor {
public:
  bool Check(const Object *stmt, const VarNode *dst, const VarNode *src) {
    dst_ = dst;
    src_ = src;
    result_ = true;
    if (stmt->IsInstance<AttrStmtNode>()) {
      VisitStmt_(reinterpret_cast<const AttrStmtNode *>(stmt));
    } else if (stmt->IsInstance<ForNode>()) {
      VisitStmt_(reinterpret_cast<const ForNode *>(stmt));
    } else if (stmt->IsInstance<IfThenElseNode>()) {
      VisitStmt_(reinterpret_cast<const IfThenElseNode *>(stmt));
    } else if (stmt->IsInstance<WhileNode>()) {
      VisitStmt_(reinterpret_cast<const WhileNode *>(stmt));
    } else if (stmt->IsInstance<BufferStoreNode>()) {
      VisitStmt_(reinterpret_cast<const BufferStoreNode *>(stmt));
    } else {
      return false;
    }
    return result_;
  }

  using StmtExprVisitor::VisitStmt_;

  void VisitStmt(const Stmt &n) final {
    if (!result_)
      return;
    StmtExprVisitor::VisitStmt(n);
  }
  void VisitExpr(const PrimExpr &n) final {
    if (!result_)
      return;
    StmtExprVisitor::VisitExpr(n);
  }

  void VisitExpr_(const VarNode *op) final {
    // assume all opaque access is unsafe
    if (op == dst_ || op == src_) {
      result_ = false;
      return;
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    ++mem_nest_;
    for (const auto &index : op->indices) {
      this->VisitExpr(index);
    }
    --mem_nest_;
    if (op->buffer->data.get() == dst_) {
      store_ = op;
      this->VisitExpr(op->value);
      store_ = nullptr;
    } else {
      this->VisitExpr(op->value);
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    // always reject extern code
    if (op->attr_key == tirx::attr::extern_scope ||
        op->attr_key == tilelang::tl_attr::volatile_scope) {
      result_ = false;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    const VarNode *buf = op->buffer->data.get();
    // cannot read from dst_ (no reduction)
    if (buf == dst_) {
      result_ = false;
      return;
    }
    // do not allow indirect memory load
    if (mem_nest_ != 0) {
      result_ = false;
      return;
    }
    if (src_ == buf) {
      if (store_ == nullptr || store_->value.dtype() != op->dtype) {
        result_ = false;
        return;
      }
      ICHECK_EQ(store_->indices.size(), op->indices.size())
          << "Store/Load occur to the same buffer " << buf->name_hint
          << " with differing number of indices";
      for (size_t i = 0; i < store_->indices.size(); i++) {
        if (!tirx::ExprDeepEqual()(store_->indices[i], op->indices[i])) {
          result_ = false;
          return;
        }
      }
    }
    ++mem_nest_;
    StmtExprVisitor::VisitExpr_(op);
    --mem_nest_;
  }

private:
  // result of the check
  bool result_{true};
  // destination memory
  const VarNode *dst_{};
  // source variable
  const VarNode *src_{};
  // counter of load,
  // it is not safe to inplace when there is nested load like A[B[i]]
  int mem_nest_{0};
  // The current store to be inspected
  const BufferStoreNode *store_{nullptr};
};

/* \brief Rewrite and merge memory allocation.
 *
 * Using LinearAccessPatternFinder, determines which buffers could share an
 * allocation.  This includes both sequential usage of the same buffer and
 * merging small allocations at the same scope into a single larger allocation.
 * The merging of small allocations requires the codegen to cast the resulting
 * value from the storage type to the output type after access.
 */
class StoragePlanRewriter : public StmtExprMutator {
public:
  using StmtEntry = LinearAccessPatternFinder::StmtEntry;
  using AllocEntry = LinearAccessPatternFinder::AllocEntry;

  Stmt Rewrite(Stmt stmt, bool detect_inplace, bool enable_reuse,
               bool reuse_require_exact_matched_dtype,
               bool reuse_shared_memory,
               bool reuse_large_plain_local,
               Map<Var, PrimExpr> local_var_init_map = {}) {
    detect_inplace_ = detect_inplace;
    reuse_shared_memory_ = reuse_shared_memory;
    reuse_large_plain_local_ = reuse_large_plain_local;
    local_var_init_map_ = std::move(local_var_init_map);
    // plan the rewrite
    LinearAccessPatternFinder finder;
    finder(stmt);
    this->LivenessAnalysis(finder.linear_seq_);
    this->PlanMemory(finder.linear_seq_, finder.alloc_info_, enable_reuse,
                     reuse_require_exact_matched_dtype);
    all_buffers_accessed_ = finder.all_buffers_accessed_;
    this->PrepareNewAlloc();
    // start rewrite
    stmt = operator()(std::move(stmt));
    if (attach_map_.count(nullptr)) {
      return MakeAttach(attach_map_.at(nullptr), stmt);
    }
    return stmt;
  }

  template <typename Node> Node VisitBufferAccess(Node node) {
    auto it = alloc_map_.find(node->buffer->data.get());
    if (it != alloc_map_.end()) {
      Buffer buf = RemapBuffer(node->buffer, it->second->alloc_var);

      Array<PrimExpr> indices = node->indices;
      indices.Set(indices.size() - 1,
                  RemapIndex(node->buffer->dtype, indices[indices.size() - 1],
                             it->second));

      auto writer = node.CopyOnWrite();
      writer->buffer = buf;
      writer->indices = indices;
    }
    return node;
  }

  Buffer RemapBuffer(const Buffer &buf, const Var &new_backing_array) {
    auto key = buf.get();
    auto it = buffer_remap_.find(key);
    if (it != buffer_remap_.end()) {
      ICHECK_EQ(it->second->data.get(), new_backing_array.get())
          << "Cannot remap buffer " << buf->name << " to use backing array "
          << new_backing_array->name_hint << ", previously used backing array "
          << it->second->data->name_hint;
      return it->second;
    }

    Buffer remapped = Buffer(
        new_backing_array, buf->dtype, buf->shape, buf->strides,
        buf->elem_offset, new_backing_array->name_hint, buf->data_alignment,
        buf->offset_factor, buf->buffer_type, buf->axis_separators, buf->span);
    buffer_remap_[key] = remapped;
    return remapped;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = alloc_map_.find(op);
    if (it != alloc_map_.end()) {
      if (it->second->bits_offset != 0) {
        LOG(WARNING)
            << "Use a merged buffer variable address, could cause error";
      }
      return it->second->alloc_var;
    } else {
      return tvm::ffi::GetRef<PrimExpr>(op);
    }
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      const VarNode *buffer = op->args[1].as<VarNode>();
      auto it = alloc_map_.find(buffer);
      if (it == alloc_map_.end()) {
        return StmtExprMutator::VisitExpr_(op);
      }
      const StorageEntry *se = it->second;
      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      uint64_t elem_bits = dtype.bits() * dtype.lanes();
      ICHECK_EQ(se->bits_offset % elem_bits, 0U);
      if (se->bits_offset != 0) {
        offset =
            make_const(offset.dtype(), se->bits_offset / elem_bits) + offset;
      }
      return Call(op->dtype, op->op,
                  {op->args[0], se->alloc_var, offset, extent, op->args[4]});
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent ||
        op->attr_key == s_tir::attr::virtual_thread ||
        op->attr_key == tl::attr::kLexicalAllocScope ||
        tirx::attr::IsPragmaKey(op->attr_key)) {
      // remake all the allocation at the attach scope.
      if (attach_map_.count(op)) {
        auto &svec = attach_map_[op];
        Stmt stmt = StmtExprMutator::VisitStmt_(op);
        op = stmt.as<AttrStmtNode>();
        return AttrStmt(op->node, op->attr_key, op->value,
                        MakeAttach(svec, op->body));
      } else {
        return StmtExprMutator::VisitStmt_(op);
      }
    } else if (op->attr_key == tilelang::tl_attr::volatile_scope) {
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AttrStmtNode>();
      auto it = alloc_map_.find(op->node.as<VarNode>());
      if (it == alloc_map_.end())
        return stmt;
      return AttrStmt(it->second->alloc_var, op->attr_key, op->value, op->body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode *op) final {
    ICHECK(op->kind != ForKind::kVectorized)
        << "VectorizeLoop before LiftStorageAlloc";
    // remake all the allocation at the attach scope.
    if (attach_map_.count(op)) {
      auto &svec = attach_map_[op];
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<ForNode>();
      return For(op->loop_var, op->min, op->extent, op->kind,
                 MakeAttach(svec, op->body), op->thread_binding,
                 op->annotations);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AllocateNode *op) {
    return this->VisitStmt(op->body);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    // AllocBuffer combines the allocation and its buffer declaration.  The
    // replacement allocation has already been hoisted by PrepareNewAlloc.
    if (auto it = alloc_map_.find(op->buffer->data.get());
        it != alloc_map_.end()) {
      if (it->second->alloc_var.get() == op->buffer->data.get()) {
        // Winner allocation: strip the original duplicate AllocBuffer.
        return Evaluate(0);
      }
      // Merged allocation: keep only a buffer declaration aliasing the winner.
      Buffer buf = RemapBuffer(op->buffer, it->second->alloc_var);
      return DeclBuffer(buf);
    }
    // CPPMEGA: This TileLang pass still plans storage reuse primarily from
    // legacy body-carrying AllocateNode inputs.  Apache/tirx plans over
    // AllocBufferNode directly, so its "not in alloc_map" case really means
    // unused.  Until this pass is fully ported to AllocBuffer-based planning,
    // an unmatched AllocBuffer may be the only real storage definition for
    // device scratch created by LowerOpaqueBlock.  Preserve it fail-closed
    // instead of dropping storage and leaking the scratch Var into
    // SplitHostDevice/MakePackedAPI as an undefined ABI parameter.
    return Downcast<AllocBuffer>(StmtExprMutator::VisitStmt_(op));
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    if (hoisted_buffer_decls_.count(op->buffer.get()) ||
        !all_buffers_accessed_.count(op->buffer.get())) {
      // CPPMEGA: DeclBufferNode lost its body field; the SeqStmt visitor
      // will continue visiting the next stmt after this one, so we just
      // emit a no-op to drop this DeclBuffer.
      return Evaluate(0);
    }
    auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));

    if (auto it = alloc_map_.find(op->buffer->data.get());
        it != alloc_map_.end()) {
      Buffer buf = RemapBuffer(op->buffer, it->second->alloc_var);
      node.CopyOnWrite()->buffer = buf;
    }
    return std::move(node);
  }

private:
  struct StorageEntry {
    // The scope that this alloc attaches after
    // For shared/local memory it is beginning of the thread extent.
    // for global memory it is nullptr, means beginning of everything.
    const Object *attach_scope_{nullptr};
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // The storage scope.
    StorageScope scope;
    // The physical dimensionality of the allocations.  Since
    // StorageRewrite is applied after StorageFlatten/FlattenBuffer,
    // this is the number of physical extent dimensions.
    size_t ndim{};
    // Allocs that shares this entry.
    std::vector<AllocationRef> allocs;
    // The children of this entry, not including itself.
    std::vector<StorageEntry *> merged_children;
    // The replacement Allocate, if any.  May also include associated
    // DeclBuffer statement.
    std::vector<Stmt> alloc_nest;
    // The var expr of new allocation.
    Var alloc_var;
    // The allocation element type.
    DataType elem_type;
    // Whether any constituent allocation was marked volatile.
    bool is_volatile{false};
    // This is non-zero if this allocate is folded into another one
    // the address(in bits) becomes alloc_var + bits_offset;
    // can be effectively converted to the element type.
    // We need to convert bit_offset to offset of specific element type later.
    //
    // We use bits(instead of bytes) to support non-conventional indexing in
    // hardware. When we are merging buffer together, the bits_offset are set to
    // be aligned to certain value given by the max_simd_bits property of the
    // special memory.
    //
    // This allows effective sharing among different types as long as their
    // alignment requirement fits into the max_simd_bits.
    uint64_t bits_offset{0};
  };

  // Checks whether the storage_scope is especially tagged for a specific
  // memory. Special memory is all combined into a single allocation.
  bool IsSpecialTaggedMemory(const StorageScope &scope) {
    return !scope.tag.empty() && scope.tag != ".dyn" &&
           scope.tag != ".barrier" && scope.tag != ".cluster_barrier" &&
           scope.tag != ".workspace" && scope.tag != ".vtcm" &&
           scope.tag != ".var" && scope.tag.find(".descriptor") != 0;
  }

  // Allocate entry of node.
  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarNode *> gen;
    // variables we kill
    std::vector<const VarNode *> kill;
  };

  Stmt MakeAttach(const std::vector<StorageEntry *> &svec, Stmt body) {
    for (auto it = svec.rbegin(); it != svec.rend(); it++) {
      body = MergeNest((*it)->alloc_nest, body);
    }
    return body;
  }
  Map<String, ffi::Any> MakeAllocateAnnotations(const Var &buffer_var) const {
    Map<String, ffi::Any> annotations;
    if (local_var_init_map_.defined()) {
      auto it = local_var_init_map_.find(buffer_var);
      if (it != local_var_init_map_.end()) {
        const PrimExpr &init = (*it).second;
        annotations.Set(tl::attr::kLocalVarInit, init);
      }
    }
    return annotations;
  }
  Map<String, ffi::Any> MergeAllocAnnotations(
      const Map<String, ffi::Any> &base, const Var &buffer_var) const {
    Map<String, ffi::Any> annotations;
    for (const auto &kv : base) {
      annotations.Set(kv.first, kv.second);
    }
    Map<String, ffi::Any> generated = MakeAllocateAnnotations(buffer_var);
    for (const auto &kv : generated) {
      annotations.Set(kv.first, kv.second);
    }
    return annotations;
  }
  // Remap the index
  PrimExpr RemapIndex(DataType dtype, PrimExpr index, StorageEntry *e) {
    if (e->bits_offset == 0)
      return index;
    uint64_t elem_bits = dtype.bits();
    ICHECK_EQ(e->bits_offset % elem_bits, 0U);
    return make_const(index.dtype(), e->bits_offset / elem_bits) + index;
  }
  // Prepare the new allocations
  void PrepareNewAlloc() {
    for (size_t i = 0; i < alloc_vec_.size(); ++i) {
      StorageEntry *e = alloc_vec_[i].get();
      attach_map_[e->attach_scope_].push_back(e);
    }
    // find allocation via attach map.
    for (auto &kv : attach_map_) {
      // find the element with the most amount of bytes.
      std::vector<StorageEntry *> &vec = kv.second;
      // try to find merge, for tagged memory
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry *e = vec[i];
        if (IsSpecialTaggedMemory(e->scope)) {
          ICHECK_NE(e->const_nbits, 0U)
              << "Special tagged memory must be const size";
          for (size_t j = 0; j < i; ++j) {
            if (e->scope == vec[j]->scope) {
              vec[j]->merged_children.push_back(e);
              break;
            }
          }
        }
      }
      // Start allocation
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry *e = vec[i];
        // already merged
        if (e->bits_offset != 0)
          continue;
        if (!e->merged_children.empty()) {
          NewAllocTagMerged(e);
          continue;
        }
        // Get the allocation size;
        e->alloc_var = e->allocs[0].buffer_var();
        DataType alloc_type = e->allocs[0].dtype();
        for (const AllocationRef &op : e->allocs) {
          if (op.dtype().lanes() > alloc_type.lanes()) {
            alloc_type = op.dtype();
          }
        }

        bool all_allocs_identical = std::all_of(
            e->allocs.begin() + 1, e->allocs.end(),
            [&](const AllocationRef &op) -> bool {
              const AllocationRef &first = e->allocs.front();
              if (op.dtype() != first.dtype()) {
                return false;
              }
              if (op.extents().size() != first.extents().size()) {
                return false;
              }
              ExprDeepEqual expr_equal;
              for (size_t i = 0; i < op.extents().size(); i++) {
                if (!expr_equal(op.extents()[i], first.extents()[i])) {
                  return false;
                }
              }
              return true;
            });

        if (all_allocs_identical) {
          // simply use the original allocation.
          Map<String, ffi::Any> annotations =
              MergeAllocAnnotations(e->allocs[0].annotations(), e->alloc_var);
          // CPPMEGA: emit apache `AllocBuffer` instead of vendored
          // `tl_tir::Allocate`. apache `StmtFunctor` has no dispatch entry
          // for `tilelang.Allocate`; downstream apache passes
          // (tir.transform.Simplify, RenormalizeSplitPattern, RemoveNoOp,
          // HoistIfThenElse, ...) crash with "NodeFunctor calls
          // un-registered function on type tilelang.Allocate". MergeNest
          // accepts `AllocBuffer` natively (see ir_utils.cc:73).
          // The legacy condition was always `const_true()` here in practice
          // because merge candidates have already been filtered, but if
          // non-trivial we honor it via IfThenElse later in the nest fold.
          Buffer alloc_buf_obj =
              e->allocs[0].MakeBuffer(e->alloc_var, alloc_type,
                                       e->allocs[0].extents());
          e->alloc_nest.push_back(
              AllocBuffer(alloc_buf_obj, std::move(annotations)));
          if (auto ptr = e->allocs[0].LegacyDeclBuffer()) {
            // CPPMEGA: DeclBuffer is body-less in apache/tvm; drop the trailing
            // Evaluate(0).
            e->alloc_nest.push_back(
                DeclBuffer(RemapBuffer(ptr->buffer, e->alloc_var)));
            hoisted_buffer_decls_.insert(ptr->buffer.get());
          }
          if (IsSpecialTaggedMemory(e->scope)) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            if (info.defined()) {
              uint64_t total_elem = e->const_nbits / e->elem_type.bits();
              ICHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                  << "Allocation exceed bound of memory tag "
                  << e->scope.to_string();
            }
          }
        } else {
          // Build a merged allocation
          PrimExpr combo_size;
          for (const AllocationRef &op : e->allocs) {
            ICHECK_EQ(op.extents().size(), 1)
                << "Buffer var " << op.buffer_var()->name_hint
                << " was identified as a reusable allocation, but has "
                << op.extents().size() << " physical dimensions.  "
                << "Currently, only flat 1-d memory spaces should be "
                   "identified as reusable "
                   "allocations.";
            PrimExpr sz = op.extents()[0];
            auto nbits = op.dtype().bits() * op.dtype().lanes();
            if (const auto *imm = sz.as<IntImmNode>()) {
              if (imm->value > std::numeric_limits<int>::max() / nbits) {
                LOG(WARNING) << "The allocation requires : " << imm->value
                             << " * " << nbits
                             << " bits, which is greater than the maximum of"
                                " int32. The size is cast to int64."
                             << "\n";
                sz = make_const(DataType::Int(64), imm->value);
              }
            }
            // transform to bits
            auto sz_nbits = sz * nbits;
            if (combo_size.defined()) {
              combo_size = max(combo_size, sz_nbits);
            } else {
              combo_size = sz_nbits;
            }
          }
          // transform to alloc bytes
          auto type_bits = alloc_type.bits() * alloc_type.lanes();
          bool divided =
              analyzer_.CanProve(indexmod(combo_size, type_bits) == 0);
          combo_size = indexdiv(combo_size, type_bits);
          // round up for can not divided
          if (!divided) {
            combo_size = combo_size + make_const(DataType::Int(32), 1);
          }
          combo_size = analyzer_.Simplify(combo_size);
          Map<String, ffi::Any> annotations =
              MergeAllocAnnotations(e->allocs[0].annotations(), e->alloc_var);
          // CPPMEGA: emit apache `AllocBuffer` instead of vendored
          // `tl_tir::Allocate`. See rationale at the all_allocs_identical
          // site above.
          Buffer alloc_buf_combo =
              e->allocs[0].MakeBuffer(e->alloc_var, alloc_type, {combo_size});
          e->alloc_nest.push_back(
              AllocBuffer(alloc_buf_combo, std::move(annotations)));
          if (IsSpecialTaggedMemory(e->scope)) {
            MemoryInfo info = GetMemoryInfo(e->scope.to_string());
            if (info.defined()) {
              uint64_t total_elem = e->const_nbits / e->elem_type.bits();
              ICHECK_LE(total_elem * e->elem_type.bits(), info->max_num_bits)
                  << "Allocation exceed bound of memory tag "
                  << e->scope.to_string();
            }
          }
        }
      }
    }
  }
  // New allocation for merged data
  void NewAllocTagMerged(StorageEntry *e) {
    ICHECK_NE(e->scope.tag.length(), 0U);
    // allocate with element type.
    ICHECK_NE(e->const_nbits, 0U);
    MemoryInfo info;
    if (e->scope.tag != ".barrier" && e->scope.tag != ".var" &&
        e->scope.tag.find(".descriptor") != 0) {
      info = GetMemoryInfo(e->scope.to_string());
    }
    uint64_t total_bits = e->const_nbits;
    // By default, align to 32 bits.
    size_t align = 32;
    if (info.defined()) {
      align = info->max_simd_bits;
    }
    // Always align to max_simd_bits
    // so we can remap types by keeping this property
    if (total_bits % align != 0) {
      total_bits += align - (total_bits % align);
    }
    e->alloc_var = e->allocs[0].buffer_var();
    for (StorageEntry *child : e->merged_children) {
      ICHECK_NE(child->const_nbits, 0U);
      ICHECK_NE(total_bits, 0U);
      child->bits_offset = total_bits;
      child->alloc_var = e->alloc_var;
      total_bits += child->const_nbits;
      if (total_bits % align != 0) {
        total_bits += align - (total_bits % align);
      }
    }
    uint64_t type_bits = e->elem_type.bits() * e->elem_type.lanes();
    PrimExpr alloc_size = make_const(e->allocs[0].extents()[0].dtype(),
                                     (total_bits + type_bits - 1) / type_bits);
    Map<String, ffi::Any> annotations =
        MergeAllocAnnotations(e->allocs[0].annotations(), e->alloc_var);
    // CPPMEGA: emit apache `AllocBuffer` instead of vendored
    // `tl_tir::Allocate`. See rationale at line ~895 (NewAlloc).
    Buffer alloc_buf_tag =
        e->allocs[0].MakeBuffer(e->alloc_var, e->elem_type, {alloc_size});
    e->alloc_nest.push_back(
        AllocBuffer(alloc_buf_tag, std::move(annotations)));
    if (info.defined()) {
      ICHECK_LE(total_bits, info->max_num_bits)
          << "Allocation exceed bound of memory tag " << e->scope.to_string();
    }
  }
  // Liveness analysis to find gen and kill point of each variable.
  void LivenessAnalysis(const std::vector<StmtEntry> &seq) {
    // find kill point, do a reverse linear scan.
    std::unordered_set<const VarNode *> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry &s = seq[i - 1];
      for (const VarNode *buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].kill.push_back(buffer);
        }
      }
    }
    // find gen point, do forward scan
    touched.clear();
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset < 0)
        continue;
      const StmtEntry &s = seq[i + offset];
      for (const VarNode *buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].gen.push_back(buffer);
        }
      }
    }
  }
  void PlanNewScope(const Object *op) {
    if (thread_scope_ != nullptr) {
      ICHECK(thread_scope_ == op);
      // erase all memory attached to this scope.
      for (auto it = const_free_map_.begin(); it != const_free_map_.end();) {
        if (it->second->attach_scope_ == op) {
          it = const_free_map_.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end();) {
        if ((*it)->attach_scope_ == op) {
          it = sym_free_list_.erase(it);
        } else {
          ++it;
        }
      }
      thread_scope_ = nullptr;
    } else {
      thread_scope_ = op;
    }
  }

  /*! \brief Return the effective attach scope for the given storage scope.
   *
   * lexical_alloc_scope is intended to bound register/local-like allocations.
   * Shared/global allocations should continue to follow thread_scope_ so we do
   * not accidentally re-scope shared buffers nested inside a lexical block.
   */
  const Object *effective_scope(const StorageScope &storage_scope) const {
    if (lexical_scope_ != nullptr &&
        storage_scope.rank != StorageRank::kGlobal &&
        storage_scope.rank != StorageRank::kShared) {
      return lexical_scope_;
    }
    return thread_scope_;
  }

  // Memory plan algorithm
  void
  PlanMemory(const std::vector<StmtEntry> &seq,
             const std::unordered_map<const VarNode *, AllocEntry> &alloc_info,
             bool enable_reuse, bool reuse_require_exact_matched_dtype) {
    std::unordered_set<const VarNode *> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry &s = seq[i];
      auto it = event_map_.find(seq[i].stmt);

      // scope_pair_offset >= 0 means it is either
      // - leaf stmt(offset = 0)
      // - beginning of scope(offset < 0)
      // In both cases, we need to handle the gen event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset >= 0) {
        // Inplace operation detection
        // specially handle this
        bool detect_inplace = detect_inplace_ && (it->second.gen.size() <= 2);

        for (const VarNode *var : it->second.gen) {
          ICHECK(alloc_info.count(var));
          const AllocEntry &entry = alloc_info.at(var);
          const AllocationRef &alloc = entry.alloc;
          auto storage_scope = StorageScope::Create(
              GetPtrStorageScope(tvm::ffi::GetRef<Var>(var)));
          StorageEntry *dst_entry = nullptr;
          // inplace detection
          if (detect_inplace) {
            // only one inplace var for s.stmt
            bool inplace_found = false;
            for (const VarNode *src : it->second.kill) {
              if (!inplace_flag.count(src) && alloc_map_.count(src)) {
                InplaceOpVerifier visitor;
                StorageEntry *src_entry = alloc_map_.at(src);
                if (src_entry->scope == storage_scope &&
                    src_entry->attach_scope_ ==
                        effective_scope(storage_scope) &&
                    src_entry->elem_type == alloc.dtype().element_of() &&
                    visitor.Check(s.stmt, var, src)) {
                  uint64_t const_nbits =
                      static_cast<uint64_t>(alloc.ConstantAllocationSize()) *
                      alloc.dtype().bits() * alloc.dtype().lanes();
                  if (src_entry->const_nbits == const_nbits && !inplace_found) {
                    // successfully inplace
                    dst_entry = src_entry;
                    inplace_flag.insert(src);
                    inplace_found = true;
                  }
                }
              }
            }
          }
          if (dst_entry == nullptr) {
            dst_entry =
                FindAlloc(alloc, effective_scope(storage_scope), storage_scope,
                          entry.num_physical_dimensions, enable_reuse,
                          reuse_require_exact_matched_dtype);
          }
          dst_entry->allocs.emplace_back(alloc);
          alloc_map_[var] = dst_entry;
          non_reusable_alloc_vars_[var] =
              (!reuse_shared_memory_ &&
               storage_scope.rank == StorageRank::kShared) ||
              IsUntaggedNonReusableAlloc(alloc, storage_scope,
                                         reuse_large_plain_local_);
        }
      }
      // enter/exit new scope
      if (s.stmt->IsInstance<AttrStmtNode>()) {
        const auto *op = reinterpret_cast<const AttrStmtNode *>(s.stmt);
        if (op->attr_key == tirx::attr::thread_extent ||
            op->attr_key == s_tir::attr::virtual_thread ||
            tirx::attr::IsPragmaKey(op->attr_key)) {
          PlanNewScope(op);
        } else if (op->attr_key == tl::attr::kLexicalAllocScope) {
          if (s.scope_pair_offset > 0) {
            // Entering: redirect allocation attachment to this scope.
            // thread_scope_ is NOT touched so PlanNewScope keeps working.
            lexical_scope_stack_.push_back(lexical_scope_);
            lexical_scope_ = op;
          } else {
            // Exiting: clear free lists for this scope and restore.
            for (auto it = const_free_map_.begin();
                 it != const_free_map_.end();) {
              if (it->second->attach_scope_ == op) {
                it = const_free_map_.erase(it);
              } else {
                ++it;
              }
            }
            for (auto it = sym_free_list_.begin();
                 it != sym_free_list_.end();) {
              if ((*it)->attach_scope_ == op) {
                it = sym_free_list_.erase(it);
              } else {
                ++it;
              }
            }
            lexical_scope_ = lexical_scope_stack_.back();
            lexical_scope_stack_.pop_back();
          }
        } else {
          ICHECK(op->attr_key == tirx::attr::extern_scope);
        }
      } else if (s.stmt->IsInstance<ForNode>()) {
        const auto *op = reinterpret_cast<const ForNode *>(s.stmt);
        if (op->kind == ForKind::kParallel) {
          if (thread_scope_ == nullptr || thread_scope_ == op) {
            PlanNewScope(op);
          }
        }
      }
      // scope_pair_offset <= 0 means it is either
      // - leaf stmt(offset = 0)
      // - end of scope(offset < 0)
      // In both cases, we need to handle the kill event correctly
      if (it != event_map_.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode *var : it->second.kill) {
          // skip space which are already replaced by inplace
          if (!inplace_flag.count(var)) {
            this->Free(var);
          }
        }
      }
    }
  }
  // Allocate new storage entry.
  StorageEntry *NewAlloc(const AllocationRef &op, const Object *attach_scope,
                         const StorageScope &scope, size_t const_nbits) {
    ICHECK(op);
    // Reuse not successful, allocate a new buffer.
    auto entry = std::make_unique<StorageEntry>();
    entry->attach_scope_ = attach_scope;
    entry->scope = scope;
    entry->elem_type = op.dtype().element_of();
    entry->const_nbits = const_nbits;
    entry->is_volatile = op.is_volatile();
    StorageEntry *e = entry.get();
    alloc_vec_.emplace_back(std::move(entry));
    return e;
  }

  static uint64_t ConstantAllocationNBits(const AllocationRef &op) {
    uint64_t op_elem_bits = op.dtype().bits() * op.dtype().lanes();
    return static_cast<uint64_t>(op.ConstantAllocationSize() * op_elem_bits);
  }

  static bool IsUntaggedNonReusableAlloc(const AllocationRef &op,
                                          const StorageScope &scope,
                                          bool reuse_large_plain_local) {
    uint64_t const_nbits = ConstantAllocationNBits(op);
    bool is_known_size = (const_nbits != 0);
    if (reuse_large_plain_local && scope.rank == StorageRank::kLocal &&
        scope.tag.empty() && !op.dtype().is_handle() &&
        is_known_size && const_nbits > 32) {
      return false;
    }
    return scope.tag.empty() &&
           (scope.rank >= StorageRank::kWarp || op.dtype().is_handle() ||
            (is_known_size && const_nbits <= 32));
  }

  StorageEntry *FindAlloc(const AllocationRef &op, const Object *attach_scope,
                           const StorageScope &scope,
                           size_t num_physical_dimensions, bool enable_reuse,
                           bool reuse_require_exact_matched_dtype) {
    ICHECK(op);
    // skip plan for local variable,
    // compiler can do a better job with register allocation.
    const uint64_t match_range = 16;
    uint64_t op_elem_bits = op.dtype().bits() * op.dtype().lanes();
    uint64_t const_nbits = ConstantAllocationNBits(op);

    // If the size of the array isn't known at compile-time, it must
    // have its own allocation with size determined at runtime.
    bool is_known_size = (const_nbits != 0);

    // Currently, only flat memory spaces can be reused.  Packing
    // into N-d space (e.g. 2-d texture memory on GPUs) will require
    // more in-depth algorithms.
    bool is_flat_memory_space = (num_physical_dimensions == 1);

    // Disable reuse of untagged local/handle/small arrays; small arrays are
    // lowered to registers in LLVM.
    bool is_non_reusable_untagged_alloc =
        IsUntaggedNonReusableAlloc(op, scope, reuse_large_plain_local_);

    if (!reuse_shared_memory_ && scope.rank == StorageRank::kShared) {
      return NewAlloc(op, attach_scope, scope, const_nbits);
    }

    if (!enable_reuse || is_non_reusable_untagged_alloc ||
        !is_flat_memory_space) {
      return NewAlloc(op, attach_scope, scope, const_nbits);
    }

    if (is_known_size) {
      // constant allocation.
      auto begin = const_free_map_.lower_bound(const_nbits / match_range);
      auto mid = const_free_map_.lower_bound(const_nbits);
      auto end = const_free_map_.upper_bound(const_nbits * match_range);
      // start looking at the buffer that is bigger than the required size first
      for (auto it = mid; it != end; ++it) {
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope)
          continue;
        if (e->scope != scope)
          continue;
        // when not divided, no reuse, eg, float4 vs float3
        if (e->bits_offset % op_elem_bits != 0)
          continue;
        // must check element type to avoid type mismatch in codegen
        if (e->elem_type != op.dtype().element_of())
          continue;
        if (reuse_require_exact_matched_dtype && e->elem_type != op.dtype()) {
          continue;
        }
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }
      // then start looking at smaller buffers.
      for (auto it = mid; it != begin;) {
        --it;
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope)
          continue;
        if (e->scope != scope)
          continue;
        if (e->elem_type != op.dtype().element_of())
          continue;
        if (reuse_require_exact_matched_dtype && e->elem_type != op.dtype()) {
          continue;
        }
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }
    } else {
      // Simple strategy: round roubin.
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end(); ++it) {
        StorageEntry *e = *it;
        if (e->attach_scope_ != attach_scope)
          continue;
        if (e->scope != scope)
          continue;
        if (e->elem_type != op.dtype().element_of())
          continue;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, attach_scope, scope, const_nbits);
  }
  // simulated free.
  void Free(const VarNode *var) {
    auto it = alloc_map_.find(var);
    ICHECK(it != alloc_map_.end());
    StorageEntry *e = it->second;
    ICHECK_NE(e->allocs.size(), 0U);
    auto non_reusable_it = non_reusable_alloc_vars_.find(var);
    ICHECK(non_reusable_it != non_reusable_alloc_vars_.end());

    // Mirror FindAlloc's allocation-local reuse predicate. The StorageEntry
    // size can grow after reuse, so do not use e->const_nbits to decide whether
    // the allocation currently being freed is reusable.
    if (non_reusable_it->second) {
      return;
    }
    // normal free.
    if (e->const_nbits != 0) {
      const_free_map_.insert({e->const_nbits, e});
    } else {
      sym_free_list_.push_back(e);
    }
  }
  // thread scope.
  const Object *thread_scope_{nullptr};
  // Current lexical scope (set by lexical_alloc_scope, independent of
  // thread_scope_ so that PlanNewScope's toggle protocol is preserved).
  const Object *lexical_scope_{nullptr};
  // Stack for nested lexical scopes.
  std::vector<const Object *> lexical_scope_stack_;
  // whether enable inplace detection.
  bool detect_inplace_{false};
  // Locations of free ops.
  std::unordered_map<const Object *, EventEntry> event_map_;
  // constant size free map.
  std::multimap<uint64_t, StorageEntry *> const_free_map_;
  // symbolic free list, for non constant items.
  std::list<StorageEntry *> sym_free_list_;
  // The allocation attach map
  std::unordered_map<const Object *, std::vector<StorageEntry *>> attach_map_;
  // The allocation assign map
  std::unordered_map<const VarNode *, StorageEntry *> alloc_map_;
  // Allocation-local free-list exclusion. StorageEntry state can be shared and
  // grow after reuse, so keep this decision per original buffer var.
  std::unordered_map<const VarNode *, bool> non_reusable_alloc_vars_;
  // The allocations
  std::vector<std::unique_ptr<StorageEntry>> alloc_vec_;
  // The buffer objects being remapped
  std::unordered_map<const BufferNode *, Buffer> buffer_remap_;
  // Buffers whose DeclBuffer has been hoisted to be adjacent to the new
  // Allocate location
  std::unordered_set<const BufferNode *> hoisted_buffer_decls_;
  // Any buffers that is accessed at some point.  DeclBuffer instances
  // that do not appear in this list may be removed.
  std::unordered_set<const BufferNode *> all_buffers_accessed_;
  // Initial values for local variable buffers.
  Map<Var, PrimExpr> local_var_init_map_;
  // Static shared memory has its own target-aware merge pass. On Metal, keep
  // shared buffers distinct here so that pass can pack by real liveness.
  bool reuse_shared_memory_{true};
  // Metal compilers materialize large plain-local arrays as stack memory.
  // Let StorageRewrite reuse non-overlapping large local buffers there instead
  // of relying on backend stack allocation cleanup.
  bool reuse_large_plain_local_{false};
  // analyzer
  arith::Analyzer analyzer_;
};

/* Helper struct containing information on how a buffer is declared and used
 *
 */
struct BufferVarInfo {
  enum DeclarationLocation : uint8_t {
    kPrimFuncParam = (1 << 0),
    kPrimFuncBufferMap = (1 << 1),
    kAllocBufferNode = (1 << 2),
    kDeclBufferNode = (1 << 3),
    kLetNode = (1 << 4),
  };

  // The tirx::Var that represents this buffer.
  Var var;

  // The data type of an element of the buffer.
  DataType element_dtype;

  /* The extent of the buffer.
   *
   * If multidimensional, the extent of the last dimension of the buffer.  If
   * the size is unknown (e.g. pointer arguments to PrimFunc with no
   * corresponding entry in buffer_map), then extent is zero.
   */
  PrimExpr extent;

  // Where the buffer was declared
  DeclarationLocation declaration_location;

  // When accessed, which element type is it accessed as.  This may
  // differ both in base type (e.g. int32* cast to float32* after
  // packing in StorageRewrite) or in number of lanes (e.g. float16*
  // cast to float16x4*).
  std::unordered_set<DataType> access_dtype;
  // Data types used for scalar reads. This is used to record vectorized read
  // dtypes that can be shuffled for scalar reads when
  // rewrite_scalar_read_to_vector_shuffle is enabled.
  std::unordered_set<DataType> scalar_read_dtype;

  DataType get_preferred_dtype() const {
    std::unordered_set<DataType> base_access_dtype;
    for (auto dtype : access_dtype) {
      base_access_dtype.insert(dtype.element_of());
    }
    for (auto dtype : scalar_read_dtype) {
      base_access_dtype.insert(dtype.element_of());
    }
    // If the array is accessed as multiple base types within a
    // function, no point in changing the declared type.  CodeGenC can
    // handle this with a type-cast prior to indexing.  Vulkan will
    // raise an error at code-gen time, if a later pass doesn't split
    // it out.
    if (base_access_dtype.size() != 1) {
      return element_dtype;
    }

    DataType preferred_base_type = *base_access_dtype.begin();

    // If there is only one vectorizable size used to access the
    // buffer, and if that access size is compatible with the array
    // size, then the buffer is vectorizable.  In the future, this
    // could be improved to allow vectorized buffer access of size
    // GCD(*lanes_used), if necessary.
    // When there are scalar reads and no writes, access_dtype can be empty and
    // we should avoid rewriting.
    int preferred_lanes = element_dtype.lanes();
    if (element_dtype.lanes() == 1 && (access_dtype.size() == 1)) {
      int lanes = access_dtype.begin()->lanes();
      // Check the scalar read dtypes are compatible with the vectorized access
      // dtype.
      for (auto dtype : scalar_read_dtype) {
        if (dtype.lanes() % lanes != 0) {
          return element_dtype;
        }
      }
      arith::Analyzer analyzer_;
      arith::ModularSet me = analyzer_.modular_set(extent);
      if ((me->coeff % lanes == 0) && (me->base % lanes == 0)) {
        preferred_lanes = lanes;
      }
    }

    return preferred_base_type.with_lanes(preferred_lanes);
  }
};

/* Checks whether buffers are accessed as scalar or vector parameters in a
 * function.
 *
 */
class VectorTypeAccessChecker : public StmtExprVisitor {
public:
  /* Constructor
   *
   * @param params The parameters passed to a PrimFunc
   *
   * @param buffer_map The buffer_map associated with a PrimFunc
   *
   * @param allow_untyped_handles If a buffer or pointer variable is
   * missing a type annotation, assume that it has the same underlying
   * type as it is later accessed, with scalar element types.
   */
  VectorTypeAccessChecker(const Array<tirx::Var> &params,
                          const Map<Var, Buffer> &buffer_map,
                          bool allow_untyped_pointers = false,
                          bool detect_scalar_read_patterns = true)
      : allow_untyped_pointers_(allow_untyped_pointers),
        detect_scalar_read_patterns_(detect_scalar_read_patterns) {
    // If a parameter is in the buffer map, we want to track the
    // version in the map.
    for (auto it : buffer_map) {
      Buffer &buffer = it.second;
      Var buffer_var = buffer->data;
      DataType dtype = buffer->dtype;
      PrimExpr extent =
          !buffer->shape.empty() ? buffer->shape[buffer->shape.size() - 1] : 0;
      OnArrayDeclaration(buffer_var, dtype, extent,
                         BufferVarInfo::kPrimFuncParam);
    }

    // If a pointer parameter isn't in the buffer map, then we want to
    // track the parameter itself.
    for (Var buffer_var : params) {
      auto pointer_type = GetPointerType(buffer_var->type_annotation);
      if (pointer_type.has_value() && (buffer_map.count(buffer_var) == 0)) {
        DataType dtype = pointer_type.value();
        PrimExpr extent = 0;
        OnArrayDeclaration(buffer_var, dtype, extent,
                           BufferVarInfo::kPrimFuncBufferMap);
      }
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    OnArrayAccess(op->dtype, op->buffer->data.get(), op->indices,
                  /*is_buffer_load=*/true);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    OnArrayAccess(op->value.dtype(), op->buffer->data.get(), op->indices,
                  /*is_buffer_load=*/false);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      DataType dtype = op->args[0].dtype();
      const VarNode *buffer = op->args[1].as<VarNode>();
      PrimExpr index = op->args[2];
      OnArrayAccess(dtype, buffer, {index}, false);
    } else if (op->op.same_as(builtin::address_of())) {
      if (auto load = op->args[0].as<BufferLoadNode>()) {
        OnArrayAccess(load->dtype, load->buffer->data.get(), load->indices,
                      /*is_buffer_load=*/false);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocateNode *op) {
    const Array<PrimExpr> &extents = op->extents;
    PrimExpr extent = extents[extents.size() - 1];
    OnArrayDeclaration(op->buffer_var, op->dtype, extent,
                       BufferVarInfo::kAllocBufferNode);

    this->VisitStmt(op->body);
  }

  // CPPMEGA: After tl.LowerTileLangAllocate runs, the vendored TileLang
  // Allocate (with body) is rewritten into `SeqStmt({AllocBuffer(buf), body})`.
  // Register the buffer var here so that subsequent BufferLoad/Store accesses
  // pass the "occurred before its declaration" check.
  void VisitStmt_(const tirx::AllocBufferNode *op) final {
    const tirx::Buffer &buf = op->buffer;
    PrimExpr extent =
        !buf->shape.empty() ? buf->shape[buf->shape.size() - 1] : 0;
    OnArrayDeclaration(buf->data, buf->dtype, extent,
                       BufferVarInfo::kAllocBufferNode);
    StmtExprVisitor::VisitStmt_(op);
  }

  // CPPMEGA: lower_opaque_block also emits a standalone DeclBuffer alongside
  // the AllocBuffer (apache made DeclBuffer body-less). Register the buffer
  // var on encounter so accesses inside the enclosing SeqStmt are tracked even
  // when only a DeclBuffer is in scope (e.g., function-parameter buffers
  // re-decl'd by sub-passes).
  void VisitStmt_(const tirx::DeclBufferNode *op) final {
    const tirx::Buffer &buf = op->buffer;
    PrimExpr extent =
        !buf->shape.empty() ? buf->shape[buf->shape.size() - 1] : 0;
    OnArrayDeclaration(buf->data, buf->dtype, extent,
                       BufferVarInfo::kDeclBufferNode);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LetNode *op) final {
    HandleLetNode(op->var);
    StmtExprVisitor::VisitExpr_(op);
  }

  // apache/tvm StmtFunctor vtable does not dispatch to vendored
  // `tilelang::tl_tir::{LetStmtNode,AllocateNode}`; intercept via top-level
  // VisitStmt.
  void VisitStmt(const Stmt &n) override {
    if (const auto *op = n.as<LetStmtNode>()) {
      HandleLetNode(op->var);
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
      return;
    }
    if (const auto *op = n.as<AllocateNode>()) {
      VisitStmt_(op);
      return;
    }
    // AllocBufferNode / DeclBufferNode dispatch is handled by the apache/tvm
    // StmtFunctor vtable (these are real tirx nodes), so no manual interception
    // is required here.
    StmtExprVisitor::VisitStmt(n);
  }

  void HandleLetNode(const Var &let_var) {
    if (let_var->dtype.is_handle()) {
      auto pointer_type = GetPointerType(let_var->type_annotation);
      if (pointer_type.has_value()) {
        OnArrayDeclaration(let_var, pointer_type.value(), 0,
                           BufferVarInfo::kLetNode);
      } else if (allow_untyped_pointers_) {
        OnArrayDeclaration(let_var, let_var->dtype, 0, BufferVarInfo::kLetNode);
      } else {
        LOG(FATAL) << "Let statement of variable " << let_var->name_hint
                   << " is missing a type annotation, "
                   << "or type annotation is not a pointer to primitive";
      }
    }
  }

  /* Update the type map for a buffer based on its declaration
   *
   * @param buffer The VarNode representing the buffer.
   *
   * @param element_dtype The dtype of a single element of the buffer.
   * If unknown, when used with the allow_untyped_handles option,
   * should be a handle dtype.
   *
   * @param extent The extent of the buffer.  Zero if size is unknown.
   *
   * @param declaration_location How the buffer was allocated, so that
   * some locations can be rewritten without others.
   */
  void
  OnArrayDeclaration(const Var &buffer, DataType element_dtype, PrimExpr extent,
                     BufferVarInfo::DeclarationLocation declaration_location) {
    auto it = info_map_.find(buffer.get());
    if (it != info_map_.end()) {
      // The same buffer var may appear in more than one Allocate due to
      // upstream transforms (e.g., storage planning/merging). Treat repeated
      // declarations as benign and merge metadata instead of erroring.
      BufferVarInfo &existing = it->second;
      // Prefer a concrete element dtype if the previous one was a handle.
      if (existing.element_dtype.is_handle() && !element_dtype.is_handle()) {
        existing.element_dtype =
            element_dtype == DataType::Bool()
                ? DataType::Int(8).with_lanes(element_dtype.lanes())
                : element_dtype;
      }
      // If extent was previously unknown (0) and a concrete extent is
      // provided now, record it.
      if (!existing.extent.defined() || is_zero(existing.extent)) {
        existing.extent = extent;
      }
      // Merge declaration locations (bitwise OR of flags).
      existing.declaration_location =
          static_cast<BufferVarInfo::DeclarationLocation>(
              existing.declaration_location | declaration_location);
      return;
    }

    if (element_dtype == DataType::Bool()) {
      element_dtype = DataType::Int(8).with_lanes(element_dtype.lanes());
    }
    info_map_[buffer.get()] = BufferVarInfo{
        buffer, element_dtype, std::move(extent), declaration_location};
  }

  /* Update the type map for a buffer based on its usage
   *
   * @param value_dtype The dtype of the value being stored to or
   * loaded from the buffer.
   *
   * @param buffer The VarNode representing the buffer.
   *
   * @param indices The index at which the value is being stored/loaded.
   *
   * @param is_buffer_load Whether the access is BufferLoad
   */
  void OnArrayAccess(DataType value_dtype, const VarNode *buffer,
                     const Array<PrimExpr> &indices, bool is_buffer_load) {
    auto it = info_map_.find(buffer);
    if (it == info_map_.end()) {
      // CPPMEGA: lenient registration on miss. Some upstream passes
      // (e.g. lower_tile_op's layout remap) introduce free-standing
      // `T.Buffer(...)` aliases that reference a Var without an
      // accompanying Allocate / AllocBuffer / DeclBuffer / Let. The
      // legacy storage_rewrite tolerated this because the Allocate's
      // body-recursion implicitly declared the var. With apache/tvm's
      // body-less AllocBuffer, that lexical guarantee is gone. When
      // `allow_untyped_pointers_` is set (PointerValueTypeRewrite is
      // invoked this way from `tl.StorageRewrite`), treat the access as
      // declaring a let-style pointer of the access dtype so subsequent
      // uses are accepted. The rewriter will leave such buffers alone.
      if (allow_untyped_pointers_) {
        Var var(ffi::GetRef<Var>(buffer));
        OnArrayDeclaration(var, value_dtype, /*extent=*/0,
                           BufferVarInfo::kLetNode);
        it = info_map_.find(buffer);
      }
    }
    ICHECK(it != info_map_.end())
        << "Load/Store of buffer " << buffer->name_hint << " (" << buffer
        << ") occurred before its declaration.";

    if (value_dtype.is_scalable_vector()) {
      // Scalable types are not currently supported in storage_rewrite. Scalable
      // buffer accesses are not currently checked and therefore are not
      // rewritten.
      return;
    }

    BufferVarInfo &var_info = it->second;

    if (value_dtype.element_of() == DataType::Bool()) {
      value_dtype = DataType::Int(8).with_lanes(value_dtype.lanes());
    }

    if (var_info.element_dtype.is_handle()) {
      ICHECK(allow_untyped_pointers_)
          << "Variable " << buffer->name_hint
          << " was missing a type annotation in its declaration";
      var_info.element_dtype = value_dtype.element_of();
    }

    for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
      ICHECK(indices[i].dtype().is_scalar())
          << "Only the last index of a buffer access may be a vector type.";
    }
    int index_lanes = !indices.empty() ? indices.back().dtype().lanes() : 1;

    DataType access_dtype = value_dtype;

    int lanes_used = var_info.element_dtype.lanes();

    // This can happen due to a previous pass that had rewrite_store_load =
    // false.  This occurs from the StorageRewrite in tvm::lower, followed by
    // the PointerValueTypeRewrite in BuildSPIRV.  The rewrite_store_load =
    // false is necessary because the C-based codegens do not yet support
    // vectorized pointer types (e.g. float16x4*).  Once they do, this if
    // statement should instead be replaced by the below ICHECK_EQ.
    if (index_lanes * var_info.element_dtype.lanes() != value_dtype.lanes()) {
      // If the total element sizes differ (e.g. a bfloat16 view of a
      // bfloat16x2 buffer where each bfloat16x2 = 4 bytes but bfloat16 = 2
      // bytes), this is a reinterpret-cast view access with a finer-grained
      // element type.  The buffer's declared element dtype must not be
      // downgraded in this case; just skip lane tracking for this access.
      int declared_bytes =
          var_info.element_dtype.bits() * var_info.element_dtype.lanes() / 8;
      int access_bytes = value_dtype.bits() * value_dtype.lanes() / 8;
      if (access_bytes != declared_bytes) {
        return;
      }
      ICHECK_EQ(index_lanes, value_dtype.lanes());
      lanes_used = 1;
      var_info.element_dtype = var_info.element_dtype.with_lanes(1);
    }

    ICHECK_EQ(index_lanes * var_info.element_dtype.lanes(), value_dtype.lanes())
        << "Attempting to retrieve " << value_dtype.lanes() << " lanes of data with "
        << index_lanes << " indices into an array whose elements have "
        << var_info.element_dtype.lanes() << " lanes.  "
        << "Expected output with " << index_lanes * var_info.element_dtype.lanes()
        << " lanes.";

    // If the index is a RampNode with stride of 1 and offset
    // divisible by the number of number of lanes, and the predicate
    // does not apply any masking, then this array access could be
    // vectorized.
    if (!indices.empty()) {
      const RampNode *ramp_index = indices[indices.size() - 1].as<RampNode>();
      if (ramp_index && is_one(ramp_index->stride)) {
        if (ramp_index->lanes->IsInstance<IntImmNode>()) {
          int lanes =
              static_cast<int>(Downcast<IntImm>(ramp_index->lanes)->value);
          arith::ModularSet me = analyzer_.modular_set(ramp_index->base);
          if ((me->coeff % lanes == 0) && (me->base % lanes == 0)) {
            lanes_used = lanes;
          }
        }
      }
    }

    if (detect_scalar_read_patterns_ && is_buffer_load && !indices.empty()) {
      const PrimExpr last_dim_index = indices[indices.size() - 1];
      if (last_dim_index.dtype().lanes() == 1) {
        arith::ModularSet me = analyzer_.modular_set(last_dim_index);
        var_info.scalar_read_dtype.emplace(access_dtype.with_lanes(me->coeff));
        return;
      }
    }
    var_info.access_dtype.insert(access_dtype.with_lanes(lanes_used));
  }

  // Map of buffer variable information determined
  std::unordered_map<const VarNode *, BufferVarInfo> info_map_;

  //
  bool allow_untyped_pointers_{false};
  // Whether to detect scalar read patterns for rewriting to vector shuffle
  bool detect_scalar_read_patterns_{true};

  // internal analyzer
  arith::Analyzer analyzer_;
};

/* \brief Rewrites buffer/pointer variables from scalar types to vectorized
 * types.
 *
 * Some runtimes do not allow casting between composite types and the underlying
 * base type (e.g. Vulkan, casting from 1-lane float16* to 4-lane float16x4*).
 * In these cases, in order to have vectorized load/store on an array, the
 * element type of that array must be vectorized.  This is in contrast to
 * C-style runtimes, in which `float16x4* vec = *(float16x4*)(float_arr +
 * offset)` is valid.
 *
 * By default, VectorTypeRewriter will attempt to rewrite all buffer variables
 * to vectorized access, if the load/store occurring in the PrimFunc are all
 * vectorized.  This includes adjusting the indices being used to access the
 * array.  (e.g. If `float16* scalar_arr` is being converted to `float16x4*
 * vec_arr`, then `scalar_arr[Ramp(offset, 1, 4)]` will be converted to
 * `vec_arr[offset/4]`.)
 *
 * Currently, several of the C-style runtimes do not support buffers whose
 * elements are vectorized types, or rely on the presence of the Ramp nodes to
 * identify vectorized loads.  The boolean parameters in the constructor are to
 * mimic the previous behavior of VectorTypeRewriter, to avoid breaking these
 * runtimes.  Once all runtimes support vectorized buffer elements, these
 * parameters can be removed.
 */
class VectorTypeRewriter : public StmtExprMutator {
public:
  /* Constructor
   *
   * @param checker The VectorTypeAccessChecker that has previously read out
   * information from the PrimFunc
   *
   * @param rewrite_params Whether pointer-type parameters passed into the
   * function should be rewritten from scalar types to vectorized types.
   *
   * @param rewrite_buffer_map Whether buffers present in the buffer_map should
   * have their data variable be rewritten from scalar types to vectorized
   * types.
   *
   * @param rewrite_alloc_buffer_node Whether the buffer variable associated
   * with AllocBufferNodes should be rewritten from scalar types to vectorized
   * types.
   *
   * @param rewrite_indices Whether the indices to the Load and Store nodes
   * should be rewritten to correspond to the new buffer_var type.
   *
   * @param rewrite_let_node Whether pointer declarations in let nodes
   * should be re-written.
   */
  VectorTypeRewriter(
      const std::unordered_map<const VarNode *, BufferVarInfo> &info_map,
      bool rewrite_params = true, bool rewrite_buffer_map = true,
      bool rewrite_alloc_buffer_node = true, bool rewrite_indices = true,
      bool rewrite_let_node = true,
      bool rewrite_scalar_read_to_vector_shuffle = true)
      : rewrite_indices_(rewrite_indices) {
    int rewrite_mask = 0;
    if (rewrite_params) {
      rewrite_mask |= BufferVarInfo::kPrimFuncParam;
    }
    if (rewrite_buffer_map) {
      rewrite_mask |= BufferVarInfo::kPrimFuncBufferMap;
    }
    if (rewrite_alloc_buffer_node) {
      rewrite_mask |= BufferVarInfo::kAllocBufferNode;
    }
    if (rewrite_let_node) {
      rewrite_mask |= BufferVarInfo::kLetNode;
    }

    // Rewrite any buffer variables whose preferred type isn't their current
    // type.
    for (const auto &pair : info_map) {
      const auto &var_info = pair.second;
      DataType preferred = var_info.get_preferred_dtype();
      if (preferred != var_info.element_dtype &&
          (rewrite_mask & var_info.declaration_location)) {
        Var old_buffer_var = var_info.var;
        Var new_buffer_var(old_buffer_var->name_hint,
                           PointerType(PrimType(preferred),
                                       GetPtrStorageScope(old_buffer_var)),
                           old_buffer_var->span);

        rewrite_map_[var_info.var.get()] = {var_info.var, new_buffer_var,
                                            var_info.element_dtype, preferred};
      }
    }
  }

  /*!
   * \brief Mutator for BufferLoad or BufferStore.
   * \return The rewritten node and the shuffle index. (Only for BufferLoad)
   * When the shuffle index is non-negative, the caller should generate Shuffle
   * to extract the element from the vector.
   */
  template <typename Node> std::pair<Node, int> VisitBufferAccess(Node node) {
    int shuffle_index = -1;
    if (!rewrite_indices_) {
      return {node, shuffle_index};
    }

    auto it = rewrite_map_.find(node->buffer->data.get());
    if (it == rewrite_map_.end()) {
      return {node, shuffle_index};
    }
    const auto &info = it->second;

    Array<PrimExpr> indices = node->indices;
    const PrimExpr &last_dim_index = indices[indices.size() - 1];
    const RampNode *ramp_index = indices[indices.size() - 1].as<RampNode>();

    if (node->buffer->dtype.is_scalable_vector() ||
        last_dim_index.dtype().is_scalable_vector()) {
      // Scalable types are not currently supported in storage_rewrite. Scalable
      // buffer accesses are not currently checked and therefore are not
      // rewritten.
      return {node, shuffle_index};
    }

    if (ramp_index && is_one(ramp_index->stride) &&
        ramp_index->lanes->IsInstance<IntImmNode>()) {
      int lanes = static_cast<int>(Downcast<IntImm>(ramp_index->lanes)->value);
      PrimExpr new_index =
          ramp_index->base / make_const(ramp_index->base.dtype(), lanes);
      if (lanes != info.factor()) {
        ICHECK(info.factor() && lanes % info.factor() == 0);
        int new_lanes = lanes / info.factor();
        new_index = Ramp(new_index * new_lanes, ramp_index->stride, new_lanes,
                         ramp_index->span);
      }
      indices.Set(indices.size() - 1, new_index);
    } else if (last_dim_index.dtype().lanes() == 1 && info.factor() > 1) {
      arith::ModularSet me = analyzer_.modular_set(last_dim_index);
      ICHECK(me->coeff == 0 || info.factor() % me->coeff == 0);
      PrimExpr new_index =
          last_dim_index / make_const(last_dim_index.dtype(), info.factor());
      shuffle_index = me->base % info.factor();
      ;
      indices.Set(indices.size() - 1, new_index);
    }

    auto writer = node.CopyOnWrite();
    writer->buffer = RemapBuffer(node->buffer);
    writer->indices = indices;
    return {node, shuffle_index};
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto [modified, shuffle_index] = VisitBufferAccess(node);

    // Not needed for BufferStoreNode, so we can't just call
    // LegalizeDtype() in VisitBufferAccess.
    if (node.same_as(modified)) {
      return std::move(node);
    } else {
      auto writer = modified.CopyOnWrite();
      // writer->LegalizeDType();
      LegalizeBufferLoadDType(writer);
      if (shuffle_index >= 0) {
        return Shuffle::ExtractElement(std::move(modified), shuffle_index);
      }
      return std::move(modified);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto [modified, shuffle_index] = VisitBufferAccess(std::move(node));
    ICHECK(shuffle_index < 0);
    return std::move(modified);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    Buffer new_buf = RemapBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    auto node = CopyOnWrite(op);
    node->buffer = std::move(new_buf);
    return Stmt(node);
  }

  // apache/tvm StmtFunctor vtable does not dispatch to vendored
  // `tilelang::tl_tir::{LetStmtNode,AllocateNode}`; intercept here.
  Stmt VisitStmt(const Stmt &n) override {
    if (const auto *op = n.as<LetStmtNode>()) {
      auto it = rewrite_map_.find(op->var.get());
      PrimExpr value = this->VisitExpr(op->value);
      Stmt body = this->VisitStmt(op->body);
      Var var =
          (it == rewrite_map_.end()) ? op->var : it->second.new_buffer_var;
      if (var.same_as(op->var) && value.same_as(op->value) &&
          body.same_as(op->body)) {
        return tvm::ffi::GetRef<Stmt>(op);
      }
      return LetStmt(var, value, body);
    }
    if (const auto *op = n.as<AllocateNode>()) {
      return VisitStmt_(op);
    }
    return StmtExprMutator::VisitStmt(n);
  }

  Buffer RemapBuffer(Buffer buf) {
    auto cache_key = buf.get();

    auto cache_it = buffer_map_.find(cache_key);
    if (cache_it != buffer_map_.end()) {
      return cache_it->second;
    }

    auto info_it = rewrite_map_.find(buf->data.get());
    if (info_it != rewrite_map_.end()) {
      auto &info = info_it->second;

      Array<PrimExpr> shape = buf->shape;
      PrimExpr last_dim = shape[shape.size() - 1];
      shape.Set(shape.size() - 1,
                last_dim / make_const(last_dim.dtype(), info.factor()));

      auto writer = buf.CopyOnWrite();
      writer->data = info.new_buffer_var;
      writer->dtype = info.new_element_dtype;
      writer->shape = shape;
    }

    buffer_map_[cache_key] = buf;
    return buf;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      op = expr.as<CallNode>();

      if (!rewrite_indices_) {
        return expr;
      }

      const VarNode *buffer_var = op->args[1].as<VarNode>();
      auto it = rewrite_map_.find(buffer_var);
      if (it == rewrite_map_.end()) {
        return expr;
      }
      const auto &info = it->second;

      PrimExpr index = op->args[2];
      PrimExpr extent = op->args[3];
      PrimExpr flag = op->args[4];

      PrimExpr e_dtype = tirx::TypeAnnotation(info.new_element_dtype);
      int factor = info.factor();
      extent = extent / make_const(extent.dtype(), factor);
      index = index / make_const(index.dtype(), factor);
      Array<PrimExpr> acc_args{e_dtype, info.new_buffer_var, index, extent,
                               flag};
      return Call(info.new_element_dtype, builtin::tvm_access_ptr(), acc_args);

    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AllocateNode *op) {
    Stmt body = this->VisitStmt(op->body);

    // CPPMEGA: emit apache `AllocBuffer + SeqStmt` instead of vendored
    // `tl_tir::Allocate`. apache `StmtFunctor` has no dispatch entry for
    // `tilelang.Allocate`; downstream apache passes (tir.transform.Simplify,
    // RemoveNoOp, HoistIfThenElse, etc.) crash with "NodeFunctor calls
    // un-registered function on type tilelang.Allocate". Mirrors
    // `lower_allocate.cc` converter logic.
    auto build_alloc_seq = [&](Var buf_var, DataType dtype,
                               Array<PrimExpr> extents, PrimExpr condition,
                               Stmt inner_body,
                               Map<String, ffi::Any> annotations) -> Stmt {
      Buffer alloc_buf_obj(/*data=*/buf_var,
                           /*dtype=*/dtype,
                           /*shape=*/extents,
                           /*strides=*/{},
                           /*elem_offset=*/PrimExpr(),
                           /*name=*/buf_var->name_hint,
                           /*data_alignment=*/0,
                           /*offset_factor=*/0,
                           /*buffer_type=*/BufferType::kDefault);
      Stmt seq =
          SeqStmt({AllocBuffer(alloc_buf_obj, annotations), inner_body});
      bool trivial = false;
      if (const auto *imm = condition.as<IntImmNode>()) {
        trivial = (imm->value != 0);
      }
      if (!trivial) {
        seq = IfThenElse(condition, seq);
      }
      return seq;
    };

    auto it = rewrite_map_.find(op->buffer_var.get());
    if (it == rewrite_map_.end()) {
      return build_alloc_seq(op->buffer_var, op->dtype, op->extents,
                             op->condition, body, op->annotations);
    }

    const auto &info = it->second;

    Var new_buffer_var = info.new_buffer_var;

    Array<PrimExpr> extents = op->extents;
    PrimExpr last_extent = extents[extents.size() - 1];
    extents.Set(extents.size() - 1,
                last_extent / make_const(last_extent.dtype(), info.factor()));
    DLOG(INFO) << "Allocate with " << new_buffer_var << " and "
               << info.new_element_dtype << " extents: " << extents;
    return build_alloc_seq(new_buffer_var, info.new_element_dtype, extents,
                           op->condition, body, op->annotations);
  }
  /* Update the parameters and all remaining variable references
   *
   * Should be called after calling operator() on the body of the
   * function.
   *
   * @param func A pointer to the PrimFunc being modified.
   */
  void Finalize(PrimFunc *func_ptr) {
    ICHECK(func_ptr) << "Finalize expects a non-null pointer";
    auto &func = *func_ptr;
    auto *n = func.CopyOnWrite();

    // Remap any remaining references to the old buffer variables
    Map<Var, Var> var_remap;
    for (const auto &pair : rewrite_map_) {
      const auto &info = pair.second;
      var_remap.Set(info.old_buffer_var, info.new_buffer_var);
    }
    n->body = Substitute(n->body, var_remap);

    // Remap the argument list to use the new buffer variables.
    Array<Var> new_params;
    for (const auto &old_param : n->params) {
      auto it = rewrite_map_.find(old_param.get());
      if (it == rewrite_map_.end()) {
        new_params.push_back(old_param);
      } else {
        const auto &info = it->second;
        new_params.push_back(info.new_buffer_var);
      }
    }
    n->params = new_params;

    // Remap the Buffer objects in PrimFunc::buffer_map so that the
    // buffers use the new buffer variables
    Map<Var, Buffer> new_buffer_map;
    for (const auto &pair : n->buffer_map) {
      Var key = pair.first;
      Buffer old_buffer = pair.second;
      Var old_var = old_buffer->data;
      Buffer new_buffer = RemapBuffer(old_buffer);
      new_buffer_map.Set(key, new_buffer);
    }
    n->buffer_map = new_buffer_map;
  }

private:
  struct RewriteInfo {
    Var old_buffer_var;
    Var new_buffer_var;
    DataType old_element_dtype;
    DataType new_element_dtype;

    int factor() const {
      int old_lanes = old_element_dtype.lanes();
      int new_lanes = new_element_dtype.lanes();
      ICHECK_EQ(new_lanes % old_lanes, 0);
      return new_lanes / old_lanes;
    }
  };

  bool rewrite_indices_{true};
  std::unordered_map<const VarNode *, RewriteInfo> rewrite_map_;
  std::unordered_map<const BufferNode *, Buffer> buffer_map_;
  arith::Analyzer analyzer_;
};

// Rewrite allocates, pointer parameters, and buffer map into vectorized
// versions if each access into a buffer is the same vector type.
PrimFunc PointerValueTypeRewrite(
    PrimFunc f, bool allow_untyped_pointers = false, bool rewrite_params = true,
    bool rewrite_buffer_map = true, bool rewrite_alloc_buffer_node = true,
    bool rewrite_indices = true, bool rewrite_let_node = true,
    bool rewrite_scalar_read_to_vector_shuffle = true) {
  VectorTypeAccessChecker checker(f->params, f->buffer_map,
                                  allow_untyped_pointers,
                                  rewrite_scalar_read_to_vector_shuffle);
  checker(f->body);

  VectorTypeRewriter rewriter(
      checker.info_map_, rewrite_params, rewrite_buffer_map,
      rewrite_alloc_buffer_node, rewrite_indices, rewrite_let_node,
      rewrite_scalar_read_to_vector_shuffle);
  PrimFuncNode *n = f.CopyOnWrite();
  n->body = rewriter(std::move(n->body));
  rewriter.Finalize(&f);

  return f;
}

using namespace tirx::transform;
namespace transform {
Pass StorageRewrite() {
  auto pass_func = [](PrimFunc f, const IRModule &m, PassContext ctx) {
    if (MetalCooperativeTensorScopeDetector::Detect(f)) {
      return f;
    }
    bool detect_inplace =
        ctx->GetConfig<Bool>(kStorageRewriteDetectInplace, Bool(false)).value();
    bool enable_reuse = true;
    bool reuse_require_exact_matched_dtype = false;
    AllocateCollector collector;
    collector(f->body);
    // Always disable reuse currently, for shared memory reuse we depend on
    // MergeSharedMemoryAllocations pass, for register reuse we depend on nvcc
    // or other compiler its self.
    enable_reuse = false;

    Optional<Target> target = f->GetAttr<Target>("target");
    bool reuse_shared_memory = true;
    bool reuse_large_plain_local = false;
    if (target.defined() && target.value()->kind->name == "metal") {
      enable_reuse = true;
      reuse_shared_memory = false;
      reuse_large_plain_local = true;
    }
    if (target.defined() && (target.value()->kind->name == "vulkan" ||
                             target.value()->kind->name == "webgpu")) {
      // Require exactly same-dtype matching in smem reuse for Vulkan and WebGPU
      reuse_require_exact_matched_dtype = true;
    }
    Map<Var, PrimExpr> local_var_init_map;
    if (auto init_map =
            f->attrs.GetAttr<Map<Var, PrimExpr>>(tl::attr::kLocalVarInit)) {
      local_var_init_map = init_map.value();
    }
    auto *n = f.CopyOnWrite();
    StoragePlanRewriter plan_rewriter;
    n->body = plan_rewriter.Rewrite(
        std::move(n->body), detect_inplace, enable_reuse,
        reuse_require_exact_matched_dtype, reuse_shared_memory,
        reuse_large_plain_local,
        std::move(local_var_init_map));
    // Parameters may not be rewritten, but internal allocations may.
    return PointerValueTypeRewrite(std::move(f), true, false, false, true,
                                   true, true, false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.StorageRewrite", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.StorageRewrite", StorageRewrite);
}

Pass PointerValueTypeRewrite() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return tl::PointerValueTypeRewrite(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PointerValueTypeRewrite", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PointerValueTypeRewrite",
                        PointerValueTypeRewrite);
}

} // namespace transform
} // namespace tl
} // namespace tvm
