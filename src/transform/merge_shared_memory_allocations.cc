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
 * \file merge_shared_memory_allocations.cc
 * \brief Each GPU kernel is allowed to have only one dynamic or static shared
 * memory allocation. This pass merges multiple TIR-level dynamic or static
 * shared memory allocations into one allocation.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"
#include "../target/utils.h"
#include "runtime/thread_storage_scope.h"
#include "tirx/transform/ir_utils.h"
#include "tvm/tirx/function.h"
#include "vendored/let_stmt.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

using runtime::StorageRank;
using runtime::StorageScope;

namespace {

static bool IsDynamicSharedMemory(Var buffer_var) {
  StorageScope storage_scope =
      runtime::StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  return storage_scope.rank == runtime::StorageRank::kShared &&
         storage_scope.tag == ".dyn";
}

static bool IsStaticSharedMemory(Var buffer_var) {
  StorageScope storage_scope =
      runtime::StorageScope::Create(GetPtrStorageScope(std::move(buffer_var)));
  return storage_scope.rank == runtime::StorageRank::kShared &&
         storage_scope.tag.empty();
}

struct SharedAllocInfo {
  Var buffer_var;
  DataType dtype{DataType::UInt(8)};
  Array<PrimExpr> extents;
  ffi::Map<ffi::String, ffi::Any> annotations;
  Span span;

  int64_t ConstantAllocationSize() const {
    int64_t result = 1;
    for (const PrimExpr &extent : extents) {
      if (const auto *imm = extent.as<IntImmNode>()) {
        result *= imm->value;
      } else {
        return -1;
      }
    }
    return result;
  }
};

using SharedAllocMap = std::unordered_map<const VarNode *, SharedAllocInfo>;

static SharedAllocInfo MakeSharedAllocInfo(const AllocateNode *op) {
  return SharedAllocInfo{op->buffer_var, op->dtype, op->extents,
                         op->annotations, op->span};
}

static SharedAllocInfo MakeSharedAllocInfo(const AllocBufferNode *op) {
  return SharedAllocInfo{op->buffer->data, op->buffer->dtype, op->buffer->shape,
                         op->annotations, op->span};
}

/*!
 * \brief collect the mapping from the buffer var to its allocate
 */
class AllocateCollector : public StmtExprVisitor {
public:
  // CPPMEGA: vendored Allocate is not in apache StmtFunctor dispatch — manual
  // intercept via VisitStmt(const Stmt&).
  void VisitStmt_(const AllocateNode *op) {
    if (IsDynamicSharedMemory(op->buffer_var)) {
      dyn_shmem_allocs_[op->buffer_var.get()] = MakeSharedAllocInfo(op);
    } else if (IsStaticSharedMemory(op->buffer_var)) {
      static_shmem_allocs_[op->buffer_var.get()] = MakeSharedAllocInfo(op);
    }
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    if (IsDynamicSharedMemory(op->buffer->data)) {
      dyn_shmem_allocs_[op->buffer->data.get()] = MakeSharedAllocInfo(op);
    } else if (IsStaticSharedMemory(op->buffer->data)) {
      static_shmem_allocs_[op->buffer->data.get()] = MakeSharedAllocInfo(op);
    }
    // Apache AllocBuffer is bodyless.  The default visitor only walks Buffer
    // definition metadata, which is not needed for allocation collection and
    // can contain legacy TileLang buffer handles during this migration.
  }

  void VisitStmt_(const SBlockNode *op) final {
    // CPPMEGA: MergeSharedMemoryAllocations runs after TileLang has converted
    // semantic blocks to executable statements.  If a later pass leaves an
    // SBlock around, only its executable body matters here.  Do not walk block
    // metadata (`iter_vars`, `alloc_buffers`, regions, match buffers): apache's
    // default visitor assumes fully-normalized SBlock invariants and can crash
    // on migration-era TileLang metadata.
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (!is_one(op->predicate)) {
      this->VisitExpr(op->predicate);
    }
    this->VisitStmt(op->block);
  }

  void VisitStmt_(const EvaluateNode *op) final { this->VisitExpr(op->value); }

  void VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<AllocateNode>()) {
      VisitStmt_(op);
    } else if (const auto *op = stmt.as<LetStmtNode>()) {
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
    } else if (const auto *op = stmt.as<EvaluateNode>()) {
      this->VisitExpr(op->value);
    } else {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }
  // The dynamic mapping from the original buffer var to its allocate
  SharedAllocMap dyn_shmem_allocs_;
  // The static mapping from the original buffer var to its allocate
  SharedAllocMap static_shmem_allocs_;
};

// Classify which appropriate-shared buffers are loop-carried.
//
// A shared buffer is "loop-carried" when, in forward program order, its FIRST
// touch is a READ rather than a WRITE.  In that case the value consumed at the
// top of the body was produced by the PREVIOUS loop iteration (via the loop
// back-edge), so its storage must stay live across the whole allocation scope.
// Such a buffer is unsafe to give a tight per-statement sub-interval and must
// keep the conservative allocation-level liveness window.
//
// Conversely, a buffer whose first touch is a WRITE is fully produced and then
// consumed inside the same iteration; its storage is dead before the next
// iteration overwrites it, so a tight per-statement interval is exact and lets
// the arena packer alias it with other disjoint transients.  This is the key
// signal that lets load-A -> use-A -> load-B -> mma -> store-C reuse storage.
//
// This classification is generic and backend-neutral (the same read-before-
// write rule defines loop-carried liveness on CUDA, ROCm, and Metal alike).
class LoopCarryClassifier final : public StmtExprVisitor {
public:
  explicit LoopCarryClassifier(bool is_dynamic) : is_dynamic_(is_dynamic) {}

  // Returns the set of buffer vars that are loop-carried (read-before-write).
  std::unordered_set<const VarNode *> loop_carried_;

private:
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  // Record the first-touch kind for `buf`.  `is_read` is true for loads /
  // direct var references, false for stores.  Only the first observation per
  // buffer matters; later touches do not change the classification.
  void Observe(const VarNode *buf, bool is_read) {
    if (!buf)
      return;
    // Buffer-definition metadata (shape/stride/elem_offset vars) and other
    // non-pointer vars reach this visitor; GetPtrStorageScope ICHECKs pointer
    // type, so guard here exactly like the alignment planner does before
    // querying the storage scope.
    if (!buf->type_annotation.as<PointerTypeNode>())
      return;
    Var var = tvm::ffi::GetRef<Var>(buf);
    if (!IsAppropriateSharedMemory(var))
      return;
    if (seen_.count(buf))
      return;
    seen_.insert(buf);
    if (is_read) {
      // First touch is a read => value comes from a previous iteration / before
      // the allocation scope; the buffer is loop-carried and must stay live.
      loop_carried_.insert(buf);
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // Visit the RHS value and indices first so that a read of `buf` on the RHS
    // that precedes its own store is observed as read-before-write.
    this->VisitExpr(op->value);
    for (const PrimExpr &index : op->indices) {
      this->VisitExpr(index);
    }
    Observe(op->buffer->data.get(), /*is_read=*/false);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    Observe(op->buffer->data.get(), /*is_read=*/true);
  }

  // Extract the underlying buffer Var from a pointer expression of the form
  // address_of(BufferLoad(buf, ...)).  Returns nullptr if it is not that shape.
  static const VarNode *BufferVarOfAddressOf(const PrimExpr &ptr) {
    const auto *call = ptr.as<CallNode>();
    if (call == nullptr || !call->op.same_as(builtin::address_of()) ||
        call->args.size() != 1U)
      return nullptr;
    if (const auto *load = call->args[0].as<BufferLoadNode>())
      return load->buffer->data.get();
    return nullptr;
  }

  void VisitExpr_(const CallNode *op) final {
    // tvm_access_ptr(type, data, offset, extent, rw_mask) carries an explicit
    // direction in rw_mask (1=read, 2=write, 3=read|write).  Use it so a buffer
    // first WRITTEN through an access pointer is correctly classified
    // write-first (not loop-carried), while a buffer first READ (rw_mask=1) is
    // classified read-first (loop-carried).  Without this, the bare `data`
    // VarNode below would treat every access_ptr touch as a read, needlessly
    // pinning written-first staging buffers live.
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() == 5U) {
      if (const auto *data = op->args[1].as<VarNode>()) {
        // Visit the offset/extent expressions (they may touch other buffers),
        // but classify `data` by the mask instead of as a bare var read.
        this->VisitExpr(op->args[2]);
        this->VisitExpr(op->args[3]);
        int64_t rw_mask = 0;
        if (const auto *imm = op->args[4].as<IntImmNode>()) {
          rw_mask = imm->value;
        }
        // mask&1 (read) present, or unknown mask => treat as read (the
        // conservative, loop-carried direction).  Pure write (mask==2) only is
        // the sole case classified as a write.
        bool is_read = (rw_mask & 1) != 0 || rw_mask == 0;
        Observe(data, is_read);
        return;
      }
    }

    // simdgroup_store(frag, idx, address_of(buf[..]), ...) WRITES the buffer
    // pointed to by args[2]; simdgroup_load(...) READS it.  The pointer arrives
    // as address_of(BufferLoad(buf)), so the bare BufferLoad would otherwise be
    // misread as a load.  Classify by the intrinsic's direction instead.  This
    // is the path the Metal fragment->shared C epilogue uses to write its
    // staging tile, which must be seen as write-first so it is NOT pinned
    // loop-carried (and can therefore alias the freed A/B transients).
    bool is_sg_store = op->op.same_as(builtin::simdgroup_store());
    bool is_sg_load = op->op.same_as(builtin::simdgroup_load());
    if ((is_sg_store || is_sg_load) && op->args.size() >= 3U) {
      // Register the directional touch for the pointer arg FIRST so it wins the
      // first-observation race over the bare BufferLoad inside the address_of
      // (which the generic descent below would otherwise see as a read).
      if (const VarNode *buf = BufferVarOfAddressOf(op->args[2])) {
        Observe(buf, /*is_read=*/is_sg_load);
      }
      // Then visit all args generically so any other nested buffer touches are
      // still observed (the already-seen pointer buffer is a no-op).
      StmtExprVisitor::VisitExpr_(op);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final { Observe(op, /*is_read=*/true); }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (!is_one(op->predicate)) {
      this->VisitExpr(op->predicate);
    }
    this->VisitStmt(op->block);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    // Bodyless; do not descend into buffer-definition metadata (mirrors the
    // finder's handling for migration-era TileLang buffers).
  }

  void VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<AllocateNode>()) {
      this->VisitStmt(op->body);
      return;
    }
    if (const auto *op = stmt.as<LetStmtNode>()) {
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
      return;
    }
    if (const auto *op = stmt.as<EvaluateNode>()) {
      this->VisitExpr(op->value);
      return;
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

  bool is_dynamic_{true};
  std::unordered_set<const VarNode *> seen_;
};

// Find a linear pattern of storage access
// Used for liveness analysis.
// "linear" means fitting a complex access pattern into an array of StmtEntry
//
// Define "scope" as the body of For/thread_launch/IfThenElse
// Composite scopes(loop/thread_launch/IfThen) is represented by three
// StmtEntry: before_scope -> scope_body -> after_scope
//
// This pass tries to detect last point that we need to keep memory
// alive under the same scope as Allocate.
// The storage need to be kept alive between Allocate and last access.
// The free point is only inserted at the same scope of Allocate.
//
class SharedMemLinearAccessPatternFinder final : public StmtExprVisitor {
public:
  explicit SharedMemLinearAccessPatternFinder(
      bool is_dynamic = true, bool enable_aggressive_merge = false,
      bool verbose = false,
      std::unordered_set<const VarNode *> loop_carried = {})
      : is_dynamic_(is_dynamic),
        enable_aggressive_merge_(enable_aggressive_merge), verbose_(verbose),
        loop_carried_(std::move(loop_carried)) {}
  /*! \brief record the touch list of statement. */
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
    // the level in the scope stack
    size_t level{0};
    // Whether this variable is defined by a shared allocation.
    bool defined{false};
  };

  struct StmtAttr {
    // the level in the scope stack
    size_t level{0};
  };

  void UpdateStmtAttr(const Object *stmt, size_t level) {
    if (stmt_attrs_.find(stmt) == stmt_attrs_.end()) {
      stmt_attrs_[stmt] = StmtAttr{level};
    } else {
      stmt_attrs_[stmt].level = level;
    }
  }

  void VisitStmt_(const AllocateNode *op) {
    size_t level = scope_.size();
    const VarNode *buf = op->buffer_var.get();
    // Record the allocation site and depth so liveness can reason about the
    // original scope.
    alloc_info_[buf].defined = true;
    alloc_info_[buf].level = level;
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    size_t level = scope_.size();
    const VarNode *buf = op->buffer->data.get();
    alloc_info_[buf].defined = true;
    alloc_info_[buf].level = level;
    // Apache AllocBuffer has no body.  Do not descend into Buffer definition
    // metadata here; liveness only needs the allocation site, and default
    // metadata traversal can trip on legacy TileLang buffer objects.
  }

  void VisitStmt_(const SBlockNode *op) final {
    // See AllocateCollector above.  Liveness needs executable effects, not
    // block metadata; preserving metadata traversal out of this pass avoids
    // apache SBlock invariant assumptions during the migration.
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    VisitNewScopeBody(op, [&]() {
      if (!is_one(op->predicate)) {
        this->VisitExpr(op->predicate);
      }
      this->VisitStmt(op->block);
    });
  }

  void VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<AllocateNode>()) {
      VisitStmt_(op);
    } else if (const auto *op = stmt.as<LetStmtNode>()) {
      VisitNewScopeBody(op, [&]() {
        this->VisitExpr(op->value);
        this->VisitStmt(op->body);
      });
    } else if (const auto *op = stmt.as<EvaluateNode>()) {
      VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    scope_.push_back(StmtEntry());
    // Visit children explicitly.  A qualified base-call here bypasses this
    // migration pass's VisitStmt guard and can land on apache's default
    // tirx.Evaluate/tilelang node dispatch.
    this->VisitBufferUse(op->buffer);
    this->VisitExpr(op->value);
    for (const PrimExpr &index : op->indices) {
      this->VisitExpr(index);
    }
    // Add write access.
    const VarNode *buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.defined) {
      ICHECK_LT(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        // Non-loop-carried buffers get a tight per-statement interval at the
        // innermost frame; loop-carried buffers keep allocation-level liveness.
        scope_[TouchFrameLevel(buf, it->second.level)].touched.push_back(buf);
      }
    }

    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      UpdateStmtAttr(op, scope_level_);
      linear_seq_.push_back(e);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    scope_.push_back(StmtEntry());
    this->VisitExpr(op->value);
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (!e.touched.empty()) {
      e.stmt = op;
      UpdateStmtAttr(op, scope_level_);
      linear_seq_.push_back(e);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // Add write access.
    StmtExprVisitor::VisitExpr_(op);
    const VarNode *buf = op->buffer->data.get();
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.defined) {
      if (scope_.empty()) {
        // AllocBuffer/DeclBuffer metadata can visit the backing Var before any
        // statement frame exists. That declaration is not a live buffer use.
        return;
      }
      // Earlier we required `alloc_level < scope_.size()`, assuming every load
      // would occur strictly inside a nested scope.  In practice the lowering
      // pipeline may materialise reads in the very same frame that owns the
      // allocation (e.g. when the buffer value is passed directly to a call),
      // which used to trigger the CHECK.  Treat same-level accesses as valid so
      // the merged allocator can reason about their lifetime correctly.
      ICHECK_LE(it->second.level, scope_.size())
          << "Load memory in places other than store.";
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        // Tight per-statement interval for non-loop-carried buffers (innermost
        // frame); allocation-level liveness for loop-carried ones.  The old
        // `min(alloc_level, scope-1)` clamp is subsumed by TouchFrameLevel.
        scope_[TouchFrameLevel(buf, it->second.level)].touched.push_back(buf);
      }
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::address_of()) && op->args.size() == 1U) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        for (const PrimExpr &index : load->indices) {
          this->VisitExpr(index);
        }
        return;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *buf) final {
    // Directly reference to the variable count as a read.
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.defined) {
      if (scope_.empty()) {
        // AllocBuffer metadata visits its own data Var after registration.
        // That declaration is not a live use, and apache's liveness model only
        // records touches inside statement/scope frames.
        return;
      }
      // Same rationale as the BufferLoad path above: direct references can be
      // emitted at the allocation level after flattening, so accept them and
      // record the touch for liveness planning.
      ICHECK_LE(it->second.level, scope_.size());
      if (IsAppropriateSharedMemory(tvm::ffi::GetRef<Var>(buf))) {
        // Mirror the BufferLoad handling: tight innermost-frame interval for
        // non-loop-carried buffers, allocation-level for loop-carried ones.
        scope_[TouchFrameLevel(buf, it->second.level)].touched.push_back(buf);
      }
    }
  }

  void VisitNewScope(const AttrStmtNode *op) {
    VisitNewScopeBody(op, [&]() {
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
    });
  }

  void VisitNewScope(const IfThenElseNode *op) {
    VisitNewScopeBody(op, [&]() {
      this->VisitExpr(op->condition);
      this->VisitStmt(op->then_case);
      if (op->else_case.defined()) {
        this->VisitStmt(op->else_case.value());
      }
    });
  }

  void VisitNewScope(const ForNode *op) {
    VisitNewScopeBody(op, [&]() {
      this->VisitExpr(op->min);
      this->VisitExpr(op->extent);
      if (op->step.has_value()) {
        this->VisitExpr(op->step.value());
      }
      this->VisitStmt(op->body);
    });
  }

  void VisitNewScope(const WhileNode *op) {
    VisitNewScopeBody(op, [&]() {
      this->VisitExpr(op->condition);
      this->VisitStmt(op->body);
    });
  }

  void VisitNewScope(const AssertStmtNode *op) {
    VisitNewScopeBody(op, [&]() {
      this->VisitExpr(op->condition);
      this->VisitExpr(op->error_kind);
      for (const StringImm &message_part : op->message_parts) {
        this->VisitExpr(message_part);
      }
    });
  }

  template <typename T, typename F> void VisitNewScopeBody(const T *op, F f) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    UpdateStmtAttr(op, scope_level_);
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
    // before scope.
    linear_seq_.push_back(e);
    f();
    // after scope.
    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    ICHECK_GT(end_index, begin_index);
    // The paired entries serve as scope sentinels once we flatten the
    // control-flow tree.
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
    } else if (op->attr_key == tirx::attr::virtual_thread) {
      VisitNewScope(op);
    } else if (op->attr_key == "kWarpSpecializationScope") {
      VisitWarpSpecializationBody(op->body);
    } else {
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final { VisitNewScope(op); }

  bool ContainsSeqStmt(const Stmt &stmt) {
    if (stmt->IsInstance<SeqStmtNode>()) {
      return true;
    }
    if (const auto *if_node = stmt.as<IfThenElseNode>()) {
      return ContainsSeqStmt(if_node->then_case) ||
             (if_node->else_case.defined() &&
              ContainsSeqStmt(if_node->else_case.value()));
    }
    return false;
  }

  void VisitStmt_(const ForNode *op) final {
    if (ContainsSeqStmt(op->body)) {
      scope_level_++;
      VisitNewScope(op);
      scope_level_--;
    } else {
      VisitNewScope(op);
    }
  }

  void VisitStmt_(const WhileNode *op) final { VisitNewScope(op); }

  void VisitStmt_(const AssertStmtNode *op) final { VisitNewScope(op); }

  // linearized access sequence.
  std::vector<StmtEntry> linear_seq_;
  // The storage scope of each buffer
  std::unordered_map<const VarNode *, AllocEntry> alloc_info_;
  // The attribute of each statement
  std::unordered_map<const Object *, StmtAttr> stmt_attrs_;

private:
  void VisitWarpSpecializationBody(const Stmt &stmt) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &sub_stmt : seq->seq) {
        VisitWarpSpecializationBody(sub_stmt);
      }
      return;
    }
    if (const auto *if_node = stmt.as<IfThenElseNode>()) {
      this->VisitStmt(if_node->then_case);
      if (if_node->else_case.defined()) {
        this->VisitStmt(if_node->else_case.value());
      }
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      VisitWarpSpecializationBody(attr->body);
      return;
    }
    if (const auto *let_node = stmt.as<LetStmtNode>()) {
      this->VisitExpr(let_node->value);
      VisitWarpSpecializationBody(let_node->body);
      return;
    }
    this->VisitStmt(stmt);
  }

  // Wrapper function to determine if the shared memory allocation for a
  // variable is appropriate.
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  // Pick the scope-stack frame that should own this touch.
  //
  // The historical conservative behaviour attributed every touch to the
  // buffer's ALLOCATION frame (`alloc_level`).  When a buffer is allocated
  // outside a loop but accessed by several distinct statements inside the loop
  // body, that collapses ALL of those accesses onto the single coarse loop
  // statement, so every transient gets the identical loop-wide liveness window
  // and the arena packer cannot reuse storage between disjoint transients.
  //
  // For buffers that are NOT loop-carried (their first touch in the body is a
  // write -- see LoopCarryClassifier) the value is produced and consumed within
  // one iteration, so attributing each touch to the INNERMOST live scope frame
  // yields a tight, exact per-statement sub-interval.  load-A, use-A, load-B,
  // mma, store-C then become sequential intervals that LinearScanPack aliases
  // down to their true peak.
  //
  // Loop-carried buffers keep the conservative allocation-level attribution so
  // their storage stays reserved across the back-edge.  This is the only case
  // where the inner sub-interval would be unsafe (it would free storage that a
  // later iteration still reads), so it is exactly the case we exclude.
  size_t TouchFrameLevel(const VarNode *buf, size_t alloc_level) const {
    ICHECK(!scope_.empty());
    size_t innermost = scope_.size() - 1;
    if (enable_aggressive_merge_) {
      // Aggressive mode already attributes everything to the innermost frame.
      return innermost;
    }
    if (loop_carried_.count(buf)) {
      // Loop-carried: keep the value live across the whole allocation scope by
      // attributing the touch to the allocation frame (clamped to the current
      // stack depth, matching the historical conservative behaviour).
      return std::min(alloc_level, innermost);
    }
    // Not loop-carried: tight per-statement interval attributed to the
    // innermost live scope frame at the access site.
    return innermost;
  }

  // Whether do dynamic analysis.
  bool is_dynamic_{true};
  // Whether do aggressive merge.
  bool enable_aggressive_merge_{false};
  // Whether do verbose logging.
  bool verbose_{false};
  // Buffers that are read-before-written within their allocation scope and so
  // must retain conservative (allocation-level) liveness.
  std::unordered_set<const VarNode *> loop_carried_;
  // Whether already in thread env.
  bool in_thread_env_{false};
  // The scope stack.
  std::vector<StmtEntry> scope_;
  // The size of the scope.
  size_t scope_level_{0};
};

class SharedMemoryAlignmentPlanner : public StmtExprVisitor {

public:
  static std::unordered_map<const VarNode *, int> Plan(const Stmt &stmt) {
    SharedMemoryAlignmentPlanner planner;
    planner(stmt);
    return planner.shmem_alignment_map_;
  }

private:
  // Helper to record alignment for a shared/shared.dyn Var under alignment
  // scope
  void MarkSharedVarIfNeeded(const VarNode *op) {
    if (!op || !under_alignment_scope_)
      return;
    auto ptr_type = op->type_annotation.as<PointerTypeNode>();
    if (!ptr_type)
      return;
    auto scope = GetPtrStorageScope(tvm::ffi::GetRef<Var>(op));
    if (scope == "shared" || scope == "shared.dyn") {
      auto target = Target::Current();
      ICHECK(target.defined()) << "Target is not defined";
      const int alignment = TargetHasBulkCopy(target) ? 1024 : 16;
      shmem_alignment_map_[op] = alignment;
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::tl_gemm()) || op->op.same_as(tl::tl_gemm_sp()) ||
        op->op.same_as(tl::tma_load()) || op->op.same_as(tl::tma_store()) ||
        op->op.same_as(tl::initialize_wgmma_descriptor()) ||
        op->op.same_as(tl::initialize_tcgen05_descriptor())) {
      // These intrinsics introduce stricter SMEM alignment requirements; mark
      // the subtree.
      under_alignment_scope_ = true;
      StmtExprVisitor::VisitExpr_(op);
      under_alignment_scope_ = false;
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitExpr_(const VarNode *op) final {
    MarkSharedVarIfNeeded(op);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // If we encounter address_of(BufferLoad(...)) or any direct BufferLoad
    // within an alignment scope, make sure we mark the underlying shared var.
    if (op && under_alignment_scope_) {
      const VarNode *data_var = op->buffer->data.get();
      MarkSharedVarIfNeeded(data_var);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    // Apache AllocBuffer is bodyless. Alignment planning only needs shared
    // vars touched by alignment-sensitive intrinsics, not declaration metadata.
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (!is_one(op->predicate)) {
      this->VisitExpr(op->predicate);
    }
    this->VisitStmt(op->block);
  }

  void VisitStmt_(const EvaluateNode *op) final { this->VisitExpr(op->value); }

  void VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<AllocateNode>()) {
      this->VisitStmt(op->body);
      return;
    }
    if (const auto *op = stmt.as<LetStmtNode>()) {
      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
      return;
    }
    if (const auto *op = stmt.as<EvaluateNode>()) {
      this->VisitExpr(op->value);
      return;
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

  bool under_alignment_scope_{false};

  std::unordered_map<const VarNode *, int> shmem_alignment_map_;
};

/*!
 * \brief merge the buffers whose live range has no intersection and rewrite the
 * body
 */
class SharedMemoryRewriter : public StmtExprMutator {
public:
  explicit SharedMemoryRewriter(const SharedAllocMap &shmem_allocs,
                                bool is_dynamic = true,
                                bool verbose = false, int align_bytes = 0)
      : is_dynamic_{is_dynamic}, shmem_allocs_{shmem_allocs}, verbose_{verbose},
        align_bytes_{align_bytes} {
    if (!is_dynamic) {
      merged_buf_var_ =
          Var("buf_shmem", PointerType(PrimType(DataType::UInt(8)), "shared"));
    }
  }

  /*!
   * \brief plan the memory reuse for all the buffer allocated in the statement
   * \param stmt the statement
   */
  void PlanReuse(const Stmt &stmt, bool is_dynamic = true,
                 bool enable_aggressive_merge = false, bool verbose = false) {
    // Classify which shared buffers are loop-carried (read-before-write) so the
    // finder can give the remaining transients tight per-statement liveness
    // sub-intervals while keeping loop-carried buffers conservatively live.
    LoopCarryClassifier classifier(is_dynamic);
    classifier(stmt);
    SharedMemLinearAccessPatternFinder finder(
        is_dynamic, enable_aggressive_merge, verbose,
        std::move(classifier.loop_carried_));
    finder(stmt);
    shmem_alignment_map_ = SharedMemoryAlignmentPlanner::Plan(stmt);
    // First compute liveness over the flattened schedule, then feed it into the
    // arena packer.
    this->LivenessAnalysis(finder.linear_seq_, finder.stmt_attrs_);
    this->PlanMemory(finder.linear_seq_, finder.stmt_attrs_);
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent && !allocated_) {
      // Allocate one dynamic shared memory allocation at the beginning of
      // thread scope

      if (verbose_) {

        LOG(DEBUG) << "Memory Allocation Plan for "
                   << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
        LOG(DEBUG) << "  Merged Buffer Name: " << merged_buf_var_->name_hint;
        LOG(DEBUG) << "  Total Merged Size: " << merged_alloc_size_ << " bytes";
        LOG(DEBUG) << "  Individual Buffer Allocations:";
        for (const auto &pair : buffer_byte_offsets_) {
          const VarNode *buffer_var_node = pair.first;
          PrimExpr byte_offset = pair.second;
          auto alloc_it = shmem_allocs_.find(buffer_var_node);
          if (alloc_it != shmem_allocs_.end()) {
            const SharedAllocInfo &alloc = alloc_it->second;
            PrimExpr buffer_size_bytes = make_const(DataType::Int(64),
                                                    alloc.dtype.bytes() *
                                                        alloc.dtype.lanes());
            for (const PrimExpr &extent : alloc.extents) {
              buffer_size_bytes = buffer_size_bytes * extent;
            }
            LOG(DEBUG) << "    Buffer: " << buffer_var_node->name_hint
                       << " (Type: " << alloc.dtype << ")"
                       << ", Start Offset: " << byte_offset
                       << ", Size: " << buffer_size_bytes << " bytes"
                       << ", End Offset: "
                       << (byte_offset + buffer_size_bytes - 1);
          } else {
            LOG(DEBUG) << "    Buffer: " << buffer_var_node->name_hint
                       << ", Start Offset: " << byte_offset
                       << " (Original allocation info not found)";
          }
        }
        LOG(DEBUG) << "End of Memory Allocation Plan.";
      }

      allocated_ = true;
      // CPPMEGA: emit apache `AllocBuffer + SeqStmt` instead of the vendored
      // `tl_tir::Allocate`. apache `StmtFunctor` has no dispatch entry for
      // `tilelang.Allocate`; downstream passes
      // (tilelang.transform.Simplify only intercepts LetStmt, then
      // tir.transform.* and the host/device codegen entry points) would
      // crash with "NodeFunctor calls un-registered function on type
      // tilelang.Allocate". Mirrors apache's own merge_shared_memory_allocations
      // (3rdparty/tvm/src/s_tir/transform/merge_shared_memory_allocations.cc:341).
      Buffer merged_buf(merged_buf_var_, DataType::UInt(8),
                        {merged_alloc_size_}, {}, PrimExpr(),
                        merged_buf_var_->name_hint, 0, 0, BufferType::kDefault);
      Stmt visited_body = this->VisitStmt(op->body);
      ffi::Map<ffi::String, ffi::Any> annotations;
      if (has_volatile_alloc_) {
        annotations.Set(tirx::attr::kVolatile, Bool(true));
      }
      Stmt alloc_stmt = AllocBuffer(merged_buf, annotations);
      Stmt new_body = SeqStmt::Flatten(alloc_stmt, visited_body);
      return AttrStmt(op->node, op->attr_key, op->value, new_body, op->span);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode *op) {
    if (IsAppropriateSharedMemory(op->buffer_var)) {
      if (op->annotations.count(tirx::attr::kVolatile)) {
        has_volatile_alloc_ = true;
      }
      return this->VisitStmt(op->body);
    }
    Stmt body = this->VisitStmt(op->body);
    // CPPMEGA: emit apache `AllocBuffer + SeqStmt` instead of vendored
    // `tl_tir::Allocate`. See rationale at the merged-buffer site above.
    Buffer alloc_buf_obj(/*data=*/op->buffer_var,
                         /*dtype=*/op->dtype,
                         /*shape=*/op->extents,
                         /*strides=*/{},
                         /*elem_offset=*/PrimExpr(),
                         /*name=*/op->buffer_var->name_hint,
                         /*data_alignment=*/0,
                         /*offset_factor=*/0,
                         /*buffer_type=*/BufferType::kDefault,
                         /*axis_separators=*/{},
                         /*span=*/op->span);
    Stmt seq = SeqStmt({AllocBuffer(alloc_buf_obj, op->annotations, op->span),
                        body},
                       op->span);
    bool trivial = false;
    if (const auto *imm = op->condition.as<IntImmNode>()) {
      trivial = (imm->value != 0);
    }
    if (!trivial) {
      seq = IfThenElse(op->condition, seq, std::nullopt, op->span);
    }
    return seq;
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    if (IsAppropriateSharedMemory(op->buffer->data)) {
      if (op->annotations.count(tirx::attr::kVolatile)) {
        has_volatile_alloc_ = true;
      }
      return Evaluate(0);
    }
    // Avoid default BufferDef mutation for unrelated buffers.  This pass only
    // rewrites selected shared allocations into the merged arena, and apache's
    // metadata visitor is too strict for some TileLang migration-era buffers.
    return ffi::GetRef<Stmt>(op);
  }

  Stmt VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<AllocateNode>()) {
      return VisitStmt_(op);
    }
    if (const auto *op = stmt.as<LetStmtNode>()) {
      PrimExpr value = this->VisitExpr(op->value);
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      return LetStmt(op->var, value, body);
    }
    if (const auto *op = stmt.as<EvaluateNode>()) {
      PrimExpr value = this->VisitExpr(op->value);
      if (value.same_as(op->value)) {
        return ffi::GetRef<Stmt>(op);
      }
      return Evaluate(value, op->span);
    }
    return StmtExprMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto new_buf = GetUpdatedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto node = ffi::GetRef<DeclBuffer>(op);
    node.CopyOnWrite()->buffer = new_buf;
    return std::move(node);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    ffi::Optional<Stmt> init = std::nullopt;
    if (op->init.defined()) {
      init = this->VisitStmt(op->init.value());
    }
    Stmt body = this->VisitStmt(op->body);
    if (init.same_as(op->init) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto node = ffi::GetRef<SBlock>(op);
    node.CopyOnWrite()->init = std::move(init);
    node.CopyOnWrite()->body = std::move(body);
    return std::move(node);
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    PrimExpr predicate =
        is_one(op->predicate) ? op->predicate : this->VisitExpr(op->predicate);
    Stmt block_stmt = this->VisitStmt(op->block);
    SBlock block = Downcast<SBlock>(block_stmt);
    if (predicate.same_as(op->predicate) && block.same_as(op->block)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto node = ffi::GetRef<SBlockRealize>(op);
    node.CopyOnWrite()->predicate = std::move(predicate);
    node.CopyOnWrite()->block = std::move(block);
    return std::move(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node> Node VisitBufferAccess(Node node) {
    if (IsAppropriateSharedMemory(node->buffer->data)) {
      ICHECK_EQ(node->indices.size(), 1)
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      Array<PrimExpr> indices = {
          node->indices[0] +
          this->GetBufferOffset(node->buffer->data, node->buffer->dtype)};

      auto writer = node.CopyOnWrite();
      writer->buffer = GetUpdatedBuffer(node->buffer);
      writer->indices = indices;
    }

    return node;
  }

  Buffer GetUpdatedBuffer(Buffer buffer) {
    Buffer original_buffer = buffer;
    auto it = merged_buffer_remap_.find(original_buffer);
    if (it != merged_buffer_remap_.end()) {
      return it->second;
    }

    Buffer updated_buffer = buffer;
    if (IsAppropriateSharedMemory(buffer->data)) {
      ICHECK_EQ(buffer->shape.size(), 1)
          << "Buffer " << buffer << " has shape " << buffer->shape << ".  "
          << "MergeSharedMemoryAllocations expects flat memory buffers, "
          << "and is to be run after "
          << "StorageFlatten (TE schedules) or FlattenBuffer (TIR schedules)";
      auto writer = updated_buffer.CopyOnWrite();
      writer->data = merged_buf_var_;
    }

    merged_buffer_remap_[original_buffer] = updated_buffer;
    return updated_buffer;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      DataType dtype = op->args[0].dtype();
      Var buffer = Downcast<Var>(op->args[1]);
      if (!IsAppropriateSharedMemory(buffer)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      PrimExpr extra_offset = GetBufferOffset(buffer, dtype);

      PrimExpr offset = this->VisitExpr(op->args[2]);
      PrimExpr extent = this->VisitExpr(op->args[3]);
      return Call(op->dtype, op->op,
                  {op->args[0], merged_buf_var_, extra_offset + offset, extent,
                   op->args[4]});
    } else if (op->op.same_as(builtin::ptx_cp_async()) ||
               op->op.same_as(tl::ptx_cp_async())) {
      DataType dtype = op->dtype;
      if (op->args.size() == 3U || op->args.size() == 4U) {
        const auto *dst_access_ptr = op->args[0].as<CallNode>();
        if (dst_access_ptr == nullptr ||
            !dst_access_ptr->op.same_as(builtin::tvm_access_ptr()) ||
            dst_access_ptr->args.size() != 5U ||
            !dst_access_ptr->args[1].as<VarNode>()) {
          return StmtExprMutator::VisitExpr_(op);
        }

        // tvm_access_ptr(ptype, data, offset, extent, rw_mask)
        Var buffer = Downcast<Var>(dst_access_ptr->args[1]);
        if (!IsAppropriateSharedMemory(buffer)) {
          return StmtExprMutator::VisitExpr_(op);
        }

        DataType ptr_dtype = dst_access_ptr->args[0].dtype();
        PrimExpr extra_offset = GetBufferOffset(buffer, ptr_dtype);
        PrimExpr offset = this->VisitExpr(dst_access_ptr->args[2]);

        // Create new dst_access_ptr with merged buffer and adjusted offset.
        auto new_dst_access_ptr =
            Call(DataType::Handle(), builtin::tvm_access_ptr(),
                 {
                     dst_access_ptr->args[0], // ptype
                     merged_buf_var_,         // merged buffer
                     extra_offset + offset,   // adjusted offset
                     dst_access_ptr->args[3], // extent
                     dst_access_ptr->args[4]  // rw_mask
                 });

        Array<PrimExpr> cp_async_args = {new_dst_access_ptr, op->args[1],
                                         op->args[2]};
        if (op->args.size() == 4U) {
          cp_async_args.push_back(op->args[3]);
        }
        return Call(dtype, op->op, cp_async_args);
      }

      if (op->op.same_as(builtin::ptx_cp_async()) &&
          (op->args.size() == 5U || op->args.size() == 6U) &&
          op->args[0].as<VarNode>()) {
        Var buffer = Downcast<Var>(op->args[0]);
        if (!IsAppropriateSharedMemory(buffer)) {
          return StmtExprMutator::VisitExpr_(op);
        }
        PrimExpr extra_offset = GetBufferOffset(buffer, dtype);
        PrimExpr offset = this->VisitExpr(op->args[1]);
        int index_factor = dtype.bytes();
        Array<PrimExpr> cp_async_args = {
            merged_buf_var_,
            mul(extra_offset + offset, PrimExpr(index_factor)),
            op->args[2],
            op->args[3],
            op->args[4],
        };
        if (op->args.size() == 6U) {
          cp_async_args.push_back(op->args[5]);
        }
        return Call(dtype, op->op, cp_async_args);
      }

      return StmtExprMutator::VisitExpr_(op);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr GetBufferOffset(const Var &buffer_var, DataType dtype) {
    auto it = buffer_byte_offsets_.find(buffer_var.get());
    ICHECK(it != buffer_byte_offsets_.end())
        << "buffer_var = " << buffer_var->name_hint << ", dtype = " << dtype;
    return indexdiv(it->second, dtype.bytes() * dtype.lanes());
  }

  // Wrapper function to determine if the shared memory allocation for a
  // variable is appropriate.
  bool IsAppropriateSharedMemory(const Var &var) {
    return is_dynamic_ ? IsDynamicSharedMemory(var) : IsStaticSharedMemory(var);
  }

  using StmtEntry = SharedMemLinearAccessPatternFinder::StmtEntry;
  using StmtAttr = SharedMemLinearAccessPatternFinder::StmtAttr;

  // Metadata about a single shared-memory allocation prior to merging.  This
  // is used to build lifetimes, alignment requirements, and final offsets.
  struct BufInfo {
    const VarNode *var{nullptr};
    std::string name;
    PrimExpr size_expr;
    std::optional<int64_t> const_size_bytes; // in bytes if compile-time known.
    int alignment{0};                        // required byte alignment.
    int start{0}; // first statement index touching the buf.
    int end{0};   // one-past-last statement index.
    DataType size_dtype{DataType::Int(64)};
  };

  // Interval describing the liveness window of a (constant-sized) allocation.
  struct Interval {
    int start{0};
    int end{0};
    size_t size_bytes{0};
    int alignment{0};
    const VarNode *var{nullptr};
  };

  // Result of a linear-scan arena packing.  Offsets contain the byte offset for
  // each constant-sized buffer, arena_size is the total constant footprint.
  struct ArenaPlan {
    size_t arena_size{0};
    std::unordered_map<const VarNode *, size_t> offsets;
  };

  static size_t AlignUpSize(size_t value, size_t alignment) {
    if (alignment == 0) {
      return value;
    }
    size_t remainder = value % alignment;
    if (remainder == 0) {
      return value;
    }
    return value + (alignment - remainder);
  }

  struct FreeBlock {
    size_t offset{0};
    size_t size{0};
  };

  class FreeList {
  public:
    std::optional<size_t> Allocate(size_t need, size_t alignment) {
      // Best-fit search: pick the slot that wastes the least space after
      // alignment.
      int best = -1;
      size_t best_waste = std::numeric_limits<size_t>::max();
      for (int i = 0, n = static_cast<int>(blocks_.size()); i < n; ++i) {
        size_t aligned = AlignUpSize(blocks_[i].offset, alignment);
        size_t head = aligned - blocks_[i].offset;
        if (head > blocks_[i].size)
          continue;
        size_t usable = blocks_[i].size - head;
        if (usable < need)
          continue;
        size_t waste = blocks_[i].size - need;
        if (waste < best_waste) {
          best_waste = waste;
          best = i;
        }
      }
      if (best < 0)
        return std::nullopt;
      return CarveBlock(best, need, alignment);
    }

    // Try to allocate from the free block whose end touches arena_top.
    // The block may be smaller than need; the caller grows the arena to
    // cover the deficit.  Returns the aligned start offset on success.
    std::optional<size_t> AllocateFromTail(size_t need, size_t alignment,
                                           size_t arena_top) {
      if (blocks_.empty())
        return std::nullopt;
      int tail_idx = static_cast<int>(blocks_.size()) - 1;
      if (blocks_[tail_idx].offset + blocks_[tail_idx].size != arena_top)
        return std::nullopt;

      size_t aligned = AlignUpSize(blocks_[tail_idx].offset, alignment);
      if (aligned >= arena_top)
        return std::nullopt;

      FreeBlock blk = blocks_[tail_idx];
      size_t head = aligned - blk.offset;

      blocks_.erase(blocks_.begin() + tail_idx);
      if (head) {
        InsertBlock(blk.offset, head);
      }
      return aligned;
    }

    void Free(size_t offset, size_t size) {
      if (size == 0)
        return;
      InsertBlock(offset, size);
    }

  private:
    // Insert a block at the correct sorted position and merge with adjacent
    // neighbours so the sorted-and-coalesced invariant is preserved.
    void InsertBlock(size_t offset, size_t size) {
      FreeBlock entry{offset, size};
      auto it = std::lower_bound(
          blocks_.begin(), blocks_.end(), offset,
          [](const FreeBlock &b, size_t off) { return b.offset < off; });
      it = blocks_.insert(it, entry);

      // Merge with the next neighbour.
      auto next = std::next(it);
      if (next != blocks_.end() && it->offset + it->size >= next->offset) {
        size_t merged_end =
            std::max(it->offset + it->size, next->offset + next->size);
        it->size = merged_end - it->offset;
        blocks_.erase(next);
      }
      // Merge with the previous neighbour.
      if (it != blocks_.begin()) {
        auto prev = std::prev(it);
        if (prev->offset + prev->size >= it->offset) {
          size_t merged_end =
              std::max(prev->offset + prev->size, it->offset + it->size);
          prev->size = merged_end - prev->offset;
          blocks_.erase(it);
        }
      }
    }

    // Remove blocks_[idx], allocate `need` bytes at the aligned offset
    // within it, and return any head/tail fragments to the free list.
    size_t CarveBlock(int idx, size_t need, size_t alignment) {
      FreeBlock blk = blocks_[idx];
      blocks_.erase(blocks_.begin() + idx);

      size_t aligned = AlignUpSize(blk.offset, alignment);
      size_t head = aligned - blk.offset;
      size_t tail = blk.size - head - need;

      // InsertBlock uses lower_bound + coalesce, so insertion order is
      // irrelevant for correctness.
      if (tail)
        InsertBlock(aligned + need, tail);
      if (head)
        InsertBlock(blk.offset, head);
      return aligned;
    }

    std::vector<FreeBlock> blocks_;
  };

  struct ActiveInterval {
    int end{0};
    size_t offset{0};
    size_t size{0};
    const VarNode *var{nullptr};
    bool operator>(const ActiveInterval &other) const {
      return end > other.end;
    }
  };

  static ArenaPlan LinearScanPack(std::vector<Interval> intervals) {
    // Process intervals in program order so lifetimes correspond to the
    // linearised CFG.
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval &lhs, const Interval &rhs) {
                if (lhs.start != rhs.start) {
                  return lhs.start < rhs.start;
                }
                if (lhs.size_bytes != rhs.size_bytes) {
                  return lhs.size_bytes > rhs.size_bytes;
                }
                return lhs.var->name_hint < rhs.var->name_hint;
              });

    std::priority_queue<ActiveInterval, std::vector<ActiveInterval>,
                        std::greater<ActiveInterval>>
        active;
    FreeList freelist;
    size_t arena_top = 0;
    std::unordered_map<const VarNode *, size_t> offsets;

    auto retire = [&](int pc) {
      while (!active.empty() && active.top().end <= pc) {
        const ActiveInterval top = active.top();
        active.pop();
        freelist.Free(top.offset, top.size);
      }
    };

    for (const Interval &interval : intervals) {
      retire(interval.start);
      size_t offset = 0;
      // 1) Reuse a fully fitting free block (best-fit).
      // 2) Extend the tail free block that touches arena_top.
      // 3) Bump-allocate at arena_top (reclaim alignment gap).
      if (auto slot =
              freelist.Allocate(interval.size_bytes, interval.alignment)) {
        offset = slot.value();
      } else if (auto tail_slot = freelist.AllocateFromTail(
                     interval.size_bytes, interval.alignment, arena_top)) {
        offset = tail_slot.value();
        arena_top = offset + interval.size_bytes;
      } else {
        offset = AlignUpSize(arena_top, interval.alignment);
        // Reclaim the alignment gap [arena_top, offset) so future small
        // allocations can reuse it.
        if (offset > arena_top) {
          freelist.Free(arena_top, offset - arena_top);
        }
        arena_top = offset + interval.size_bytes;
      }
      active.push(ActiveInterval{interval.end, offset, interval.size_bytes,
                                 interval.var});
      offsets[interval.var] = offset;
    }

    return ArenaPlan{arena_top, std::move(offsets)};
  }

  PrimExpr AlignPrimExpr(const PrimExpr &value, int alignment) const {
    if (alignment <= 1) {
      return value;
    }
    DataType dtype = value.dtype();
    ICHECK(dtype.is_int() || dtype.is_uint())
        << "Expected integer dtype for alignment, but got " << dtype;
    PrimExpr align_expr = make_const(dtype, alignment);
    PrimExpr adjust = make_const(dtype, alignment - 1);
    return indexdiv(value + adjust, align_expr) * align_expr;
  }

  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<const VarNode *> gen;
    // variables we kill
    std::vector<const VarNode *> kill;
  };

  /*!
   * \brief Liveness analysis to find gen and kill point of each variable.
   * \param seq the linear pattern of storage access
   */
  void LivenessAnalysis(
      const std::vector<StmtEntry> &seq,
      const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) {
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

    if (verbose_) {
      std::vector<const Object *> stmt_keys;
      for (const auto &stmt_entry : seq) {
        auto stmt = stmt_entry.stmt;
        if (std::find(stmt_keys.begin(), stmt_keys.end(), stmt) ==
            stmt_keys.end()) {
          stmt_keys.push_back(stmt);
        }
      }
      LOG(DEBUG) << "Before reorder kill points, Liveness Analysis Results for "
                 << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
      for (const auto &stmt_key : stmt_keys) {
        auto it = event_map_.find(stmt_key);
        if (it == event_map_.end())
          continue;

        const EventEntry &entry = it->second;
        if (entry.gen.empty() && entry.kill.empty())
          continue;
        ICHECK(stmt_attrs.count(stmt_key))
            << "stmt_key = " << stmt_key->GetTypeKey();
        auto level = stmt_attrs.at(stmt_key).level;
        LOG(DEBUG) << "  Statement: " << stmt_key->GetTypeKey()
                   << " (scope_level: " << level << ")";

        std::stringstream gen_vars_ss;
        bool x_generated = false;
        for (const VarNode *var : entry.gen) {
          gen_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_generated = true;
          }
        }
        if (!entry.gen.empty()) {
          std::string gen_log_msg = "    GEN: " + gen_vars_ss.str();
          if (x_generated) {
            gen_log_msg += " <-- Buffer 'x' generated";
          }
          LOG(DEBUG) << gen_log_msg;
        }

        std::stringstream kill_vars_ss;
        bool x_killed = false;
        for (const VarNode *var : entry.kill) {
          kill_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_killed = true;
          }
        }
        if (!entry.kill.empty()) {
          std::string kill_log_msg = "    KILL: " + kill_vars_ss.str();
          if (x_killed) {
            kill_log_msg += " <-- Buffer 'x' killed";
          }
          LOG(DEBUG) << kill_log_msg;
        }
      }
      LOG(DEBUG) << "End of Liveness Analysis Results.";
    }

    // Reorder kill points:
    // For each buffer, if its kill statement is at a deeper scope level than
    // its gen statement, we need to move the kill point to the end of the gen
    // statement's scope level. This ensures proper memory deallocation at the
    // right scope boundary.
    std::vector<StmtEntry> gen_kill_seq;
    for (const auto &stmt_entry : seq) {
      // if has gen and kill, add to gen_kill_seq
      if (!event_map_[stmt_entry.stmt].gen.empty() ||
          !event_map_[stmt_entry.stmt].kill.empty()) {
        gen_kill_seq.push_back(stmt_entry);
      }
    }

    // Do not append kill points into event_map_ while iterating it.  In
    // particular, last_stmt_at_level may be the same statement whose kill vector
    // is being erased below, and push_back would invalidate the active iterator.
    std::vector<std::pair<const Object *, const VarNode *>> pending_kill_moves;
    for (auto &event_pair : event_map_) {
      const Object *stmt = event_pair.first;
      EventEntry &event = event_pair.second;

      // Skip if no kill points to process
      if (event.kill.empty())
        continue;

      // Get scope level of current statement
      ICHECK(stmt_attrs.count(stmt));
      int kill_level = stmt_attrs.at(stmt).level;

      std::unordered_set<const VarNode *> visited_buffers;

      // For each killed buffer, find its gen statement and check scope levels
      for (auto it = event.kill.begin(); it != event.kill.end();) {
        const VarNode *buffer = *it;
        bool found_gen = false;
        int gen_level = 0;

        // Find the gen statement for this buffer
        for (const auto &gen_pair : event_map_) {
          const auto &gen_event = gen_pair.second;
          if (std::find(gen_event.gen.begin(), gen_event.gen.end(), buffer) !=
              gen_event.gen.end()) {
            found_gen = true;
            gen_level = stmt_attrs.at(gen_pair.first).level;
            break;
          }
        }

        if (found_gen && kill_level > gen_level) {
          if (visited_buffers.count(buffer)) {
            ++it;
            continue;
          }
          // Need to move kill point - remove from current event
          it = event.kill.erase(it);

          // Find the last statement at gen_level and add kill point there
          // Find the last statement at gen_level in the sequence
          const Object *last_stmt_at_level = nullptr;
          auto stmt_it = gen_kill_seq.begin();
          for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
            if (stmt_it->stmt == stmt) {
              break;
            }
          }
          // start from current statement and find the last statement at
          // gen_level
          //
          // Additionally, stop if the next statement generates (births) a
          // different shared-memory buffer.  Without this check the
          // reordered kill can land *past* another buffer's gen, creating
          // a false liveness overlap that blocks memory reuse even when the
          // two buffers' true lifetimes are disjoint (e.g., Q_shared and
          // O_shared in Flash Attention can share the same shared memory
          // region).
          //
          // This is safe because shared-memory allocations (T.alloc_shared)
          // are always placed *outside* pipelined loop bodies — no new
          // shared buffer is born inside the deep scope where kills are
          // being reordered from.

          for (; stmt_it != gen_kill_seq.end(); ++stmt_it) {
            auto next_it = stmt_it + 1;
            if (next_it == gen_kill_seq.end() ||
                stmt_attrs.at(next_it->stmt).level == gen_level) {
              last_stmt_at_level = stmt_it->stmt;
              break;
            }
            // Stop if the next statement births a different shared buffer --
            // but ONLY when that buffer is born at a scope no deeper than
            // `buffer`'s own gen scope.  A sibling born at `gen_level` (e.g.
            // Q_shared / O_shared in Flash-Attention, both declared outside the
            // pipelined loop) has a disjoint lifetime and may legitimately reuse
            // the region, so we stop and let it share.  But a buffer born at a
            // level DEEPER than gen_level is born *inside* the loop body, while
            // `buffer` (gen at gen_level, outside the loop) stays live across
            // EVERY iteration via the loop back-edge.  Stopping `buffer`'s kill
            // there would let the inner buffer overlay storage that the next
            // iteration still reads (e.g. the loop-invariant Q_shared overlaid
            // by the per-iteration fp32 acc_dkv_shared staging buffer in
            // sparse-MLA backward), corrupting results.  In that case we must
            // NOT stop -- continue scanning so the kill is pushed to the real
            // gen_level boundary (the end of the loop).
            auto next_event_it = event_map_.find(next_it->stmt);
            if (next_event_it != event_map_.end() &&
                !next_event_it->second.gen.empty()) {
              int next_level = stmt_attrs.at(next_it->stmt).level;
              bool has_other_gen = false;
              for (const VarNode *gen_buf : next_event_it->second.gen) {
                if (gen_buf != buffer) {
                  has_other_gen = true;
                  break;
                }
              }
              if (has_other_gen && next_level <= gen_level) {
                last_stmt_at_level = stmt_it->stmt;
                break;
              }
            }
          }
          if (last_stmt_at_level) {
            pending_kill_moves.emplace_back(last_stmt_at_level, buffer);
            visited_buffers.insert(buffer);
          }
        } else {
          ++it;
        }
      }
    }
    for (const auto &move : pending_kill_moves) {
      event_map_[move.first].kill.push_back(move.second);
    }

    std::vector<const Object *> stmt_keys;
    for (const auto &stmt_entry : seq) {
      auto stmt = stmt_entry.stmt;
      if (std::find(stmt_keys.begin(), stmt_keys.end(), stmt) ==
          stmt_keys.end()) {
        stmt_keys.push_back(stmt);
      }
    }

    if (verbose_) {
      LOG(DEBUG) << "Liveness Analysis Results for "
                 << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
      for (const auto &stmt_key : stmt_keys) {
        auto it = event_map_.find(stmt_key);
        if (it == event_map_.end())
          continue;

        const EventEntry &entry = it->second;
        if (entry.gen.empty() && entry.kill.empty())
          continue;
        ICHECK(stmt_attrs.count(stmt_key))
            << "stmt_key = " << stmt_key->GetTypeKey();
        auto level = stmt_attrs.at(stmt_key).level;
        LOG(DEBUG) << "  Statement: " << stmt_key->GetTypeKey()
                   << " (scope_level: " << level << ")";

        std::stringstream gen_vars_ss;
        bool x_generated = false;
        for (const VarNode *var : entry.gen) {
          gen_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_generated = true;
          }
        }
        if (!entry.gen.empty()) {
          std::string gen_log_msg = "    GEN: " + gen_vars_ss.str();
          if (x_generated) {
            gen_log_msg += " <-- Buffer 'x' generated";
          }
          LOG(DEBUG) << gen_log_msg;
        }

        std::stringstream kill_vars_ss;
        bool x_killed = false;
        for (const VarNode *var : entry.kill) {
          kill_vars_ss << var->name_hint << " ";
          if (var->name_hint == "x") {
            x_killed = true;
          }
        }
        if (!entry.kill.empty()) {
          std::string kill_log_msg = "    KILL: " + kill_vars_ss.str();
          if (x_killed) {
            kill_log_msg += " <-- Buffer 'x' killed";
          }
          LOG(DEBUG) << kill_log_msg;
        }
      }
      LOG(DEBUG) << "End of Liveness Analysis Results.";
    }
  }

  /*!
   * \brief Memory plan algorithm
   * \param seq the linear pattern of storage access
   * \param alloc_info
   */
  void
  PlanMemory(const std::vector<StmtEntry> &seq,
             const std::unordered_map<const Object *, StmtAttr> &stmt_attrs) {
    buffer_byte_offsets_.clear();
    (void)stmt_attrs;

    if (shmem_allocs_.empty()) {
      merged_alloc_size_ = make_const(DataType::Int(64), 0);
      return;
    }

    // Discover the first and last touch for every allocation.
    std::unordered_map<const VarNode *, int> start_index;
    std::unordered_map<const VarNode *, int> end_index;

    for (size_t i = 0; i < seq.size(); ++i) {
      auto it = event_map_.find(seq[i].stmt);
      if (it == event_map_.end())
        continue;
      for (const VarNode *var : it->second.gen) {
        start_index.emplace(var, static_cast<int>(i));
      }
      for (const VarNode *var : it->second.kill) {
        end_index[var] = std::max(end_index[var], static_cast<int>(i) + 1);
      }
    }

    const int seq_len = static_cast<int>(seq.size());
    for (const auto &kv : start_index) {
      if (!end_index.count(kv.first)) {
        end_index[kv.first] = seq_len;
      }
    }

    // Create a sorted vector of keys from shmem_allocs_ for deterministic
    // iteration
    std::vector<const VarNode *> sorted_vars;
    sorted_vars.reserve(shmem_allocs_.size());
    for (const auto &kv : shmem_allocs_) {
      sorted_vars.push_back(kv.first);
    }
    std::sort(sorted_vars.begin(), sorted_vars.end(),
              [](const VarNode *a, const VarNode *b) {
                return a->name_hint < b->name_hint;
              });

    std::vector<BufInfo> buf_infos;
    buf_infos.reserve(shmem_allocs_.size());
    // Build a BufInfo for every collected allocation.  Some buffers are only
    // passed by pointer to an intrinsic, so they may not produce gen/kill
    // events in the flattened liveness stream.  They still must be removed and
    // remapped, otherwise lower_device_kernel_launch sees multiple
    // `shared.dyn` AllocBuffers.
    for (const VarNode *var : sorted_vars) {
      auto start_it = start_index.find(var);

      BufInfo info;
      info.var = var;
      info.name = var->name_hint;
      info.start =
          start_it == start_index.end() ? seq_len : start_it->second;
      info.end = std::max(end_index[var], info.start + 1);
      info.alignment = align_bytes_;
      auto align_it = shmem_alignment_map_.find(var);
      if (align_it != shmem_alignment_map_.end()) {
        info.alignment = std::max(info.alignment, align_it->second);
      }

      const SharedAllocInfo &alloc = shmem_allocs_.at(var);
      int64_t bytes_per_elem =
          static_cast<int64_t>(alloc.dtype.bytes() * alloc.dtype.lanes());
      // Shared-memory arena sizes and offsets are byte counts.  Large fused
      // kernels can exceed int32 at compile time, so keep the arena math in
      // int64 even when individual allocation extents are int32.
      DataType size_dtype = DataType::Int(64);

      PrimExpr size_expr = make_const(size_dtype, bytes_per_elem);
      for (const PrimExpr &extent : alloc.extents) {
        PrimExpr e = extent;
        if (e.dtype() != size_dtype) {
          e = cast(size_dtype, e);
        }
        size_expr = size_expr * e;
      }
      info.size_dtype = size_dtype;
      info.size_expr = size_expr;

      int64_t const_extent = alloc.ConstantAllocationSize();
      if (const_extent >= 0) {
        info.const_size_bytes = const_extent * bytes_per_elem;
      }

      buf_infos.push_back(std::move(info));
    }

    // Stable order so the later passes have deterministic behaviour.
    std::sort(buf_infos.begin(), buf_infos.end(),
              [](const BufInfo &a, const BufInfo &b) {
                if (a.start != b.start)
                  return a.start < b.start;
                if (a.end != b.end)
                  return a.end < b.end;
                return a.name < b.name;
              });

    std::vector<Interval> intervals;
    intervals.reserve(buf_infos.size());
    for (const BufInfo &info : buf_infos) {
      if (!info.const_size_bytes.has_value())
        continue;
      // Only constant-sized buffers participate in the arena packing because
      // dynamic sizes must be placed sequentially later.
      Interval interval;
      interval.start = info.start;
      interval.end = info.end;
      interval.size_bytes = static_cast<size_t>(
          std::max<int64_t>(0, info.const_size_bytes.value()));
      interval.alignment = info.alignment;
      interval.var = info.var;
      intervals.push_back(interval);
    }

    ArenaPlan plan = LinearScanPack(std::move(intervals));
    size_t arena_size_const = plan.arena_size;

    if (verbose_) {
      LOG(DEBUG) << "ArenaPlan (constant buffers): arena_size="
                 << arena_size_const;
      for (const auto &kv : plan.offsets) {
        const VarNode *var = kv.first;
        LOG(DEBUG) << "  " << var->name_hint << " -> offset=" << kv.second;
      }
    }

    // Cursor tracks the running byte offset within the merged arena.
    DataType offset_dtype = DataType::Int(64);
    PrimExpr total_size = make_const(offset_dtype, 0);
    PrimExpr cursor = AlignPrimExpr(
        make_const(offset_dtype, static_cast<int64_t>(arena_size_const)),
        align_bytes_);

    auto CastToOffset = [&](PrimExpr expr) -> PrimExpr {
      if (expr.dtype() == offset_dtype) {
        return expr;
      }
      return cast(offset_dtype, expr);
    };

    for (const BufInfo &info : buf_infos) {
      PrimExpr offset_expr;
      auto it = plan.offsets.find(info.var);
      if (it != plan.offsets.end()) {
        offset_expr =
            make_const(offset_dtype, static_cast<int64_t>(it->second));
      } else {
        // Dynamic-sized buffers are appended after the constant arena.
        cursor = AlignPrimExpr(cursor, info.alignment);
        PrimExpr size_expr = CastToOffset(info.size_expr);
        offset_expr = cursor;
        cursor = offset_expr + size_expr;
      }

      buffer_byte_offsets_[info.var] = offset_expr;
      PrimExpr buf_end = offset_expr + CastToOffset(info.size_expr);
      total_size = max(total_size, buf_end);
    }

    merged_alloc_size_ = buf_infos.empty()
                             ? make_const(offset_dtype, 0)
                             : AlignPrimExpr(total_size, align_bytes_);

    if (verbose_) {
      LOG(DEBUG) << "Memory Allocation Plan for "
                 << (is_dynamic_ ? "Dynamic" : "Static") << " Shared Memory:";
      LOG(DEBUG) << "  Total Merged Size (aligned): " << merged_alloc_size_;
      for (const BufInfo &info : buf_infos) {
        const PrimExpr &offset = buffer_byte_offsets_.at(info.var);
        LOG(DEBUG) << "    Buffer: " << info.name << " start=" << info.start
                   << " end=" << info.end << " alignment=" << info.alignment
                   << " offset=" << offset << " size=" << info.size_expr;
      }
    }

    // Correctness guard: verify no two SIMULTANEOUSLY-LIVE constant buffers were
    // placed at overlapping byte ranges.  A liveness-window overlap combined
    // with a byte-range overlap means the arena packer aliased two buffers that
    // are both live at the same program point -- one would silently clobber the
    // other, producing wrong results.  This must NEVER happen with a correct
    // liveness computation, so it is a hard error rather than a silent
    // sequential-layout fallback (RULE#1: fail loud, do not paper over a bug).
    //
    // This runs unconditionally (not only under verbose logging) precisely
    // because it is the safety net for the per-statement sub-interval liveness:
    // any mistake there surfaces here as a crash instead of corrupt output.
    for (size_t i = 0; i < buf_infos.size(); ++i) {
      const BufInfo &a = buf_infos[i];
      auto a_off_imm = buffer_byte_offsets_.at(a.var).as<IntImmNode>();
      if (!a.const_size_bytes.has_value() || a_off_imm == nullptr)
        continue;
      int64_t a_off = a_off_imm->value;
      int64_t a_end = a_off + a.const_size_bytes.value();
      for (size_t j = i + 1; j < buf_infos.size(); ++j) {
        const BufInfo &b = buf_infos[j];
        auto b_off_imm = buffer_byte_offsets_.at(b.var).as<IntImmNode>();
        if (!b.const_size_bytes.has_value() || b_off_imm == nullptr)
          continue;
        bool live_overlap = !(a.end <= b.start || b.end <= a.start);
        if (!live_overlap)
          continue;
        int64_t b_off = b_off_imm->value;
        int64_t b_end = b_off + b.const_size_bytes.value();
        bool mem_overlap = !(a_end <= b_off || b_end <= a_off);
        if (mem_overlap) {
          LOG(FATAL)
              << "MergeSharedMemoryAllocations produced an INVALID plan: shared "
              << "buffers '" << a.name << "' and '" << b.name << "' have "
              << "overlapping lifetimes (liveness windows [" << a.start << ", "
              << a.end << ") and [" << b.start << ", " << b.end << ")) yet were "
              << "placed at overlapping byte ranges ([" << a_off << ", " << a_end
              << ") and [" << b_off << ", " << b_end << ")). This would alias "
              << "two simultaneously-live buffers and corrupt results. This is "
              << "a liveness/packing bug -- failing loud instead of emitting a "
              << "silently-wrong kernel.";
        }
      }
    }
  }

  // Whether enable dynamic analysis.
  bool is_dynamic_{true};

  // Whether enable verbose logging.
  bool verbose_{false};
  // The alignment bytes for the merged buffer
  int align_bytes_{16};
  // The var for the merged buffer
  Var merged_buf_var_{"buf_dyn_shmem",
                      PointerType(PrimType(DataType::UInt(8)), "shared.dyn")};
  // The mapping from the original buffer var to its allocate
  SharedAllocMap shmem_allocs_;
  // The size of the merged buffer
  PrimExpr merged_alloc_size_{0};
  // The mapping from the original buffer var to its offset in the merged buffer
  std::unordered_map<const VarNode *, PrimExpr> buffer_byte_offsets_;
  // The mapping from the original buffer objects to their location in the
  // merged buffer.
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      merged_buffer_remap_;
  // The flag indicating whether the merged buffer has been allocated
  bool allocated_{false};
  // Whether any merged allocation was marked volatile.
  bool has_volatile_alloc_{false};
  // Locations of free ops.
  std::unordered_map<const Object *, EventEntry> event_map_;
  // The mapping of buffer bytes alignment
  std::unordered_map<const VarNode *, int> shmem_alignment_map_;
};

} // namespace

Stmt MergeSharedMemoryAllocations(Stmt stmt, bool merge_static_smem,
                                  bool enable_aggressive_merge,
                                  int align_bytes = 16, bool verbose = false) {
  AllocateCollector collector;
  collector(stmt);
  // Host wrapper PrimFuncs can reach this pass after SplitHostDevice with a
  // plain tirx.Evaluate body and no shared allocations.  There is nothing to
  // merge in that case, and skipping the rest avoids driving migration-era
  // mixed IR through apache's strict statement dispatch tables.
  if (collector.dyn_shmem_allocs_.size() <= 1 &&
      (!merge_static_smem || collector.static_shmem_allocs_.size() <= 1)) {
    return stmt;
  }
  if (collector.dyn_shmem_allocs_.size() > 1) {
    SharedMemoryRewriter rewriter(collector.dyn_shmem_allocs_, true, verbose,
                                  align_bytes);
    rewriter.PlanReuse(stmt, true, enable_aggressive_merge);
    stmt = rewriter(std::move(stmt));
  }
  if (merge_static_smem && collector.static_shmem_allocs_.size() > 1) {
    SharedMemoryRewriter rewriter(collector.static_shmem_allocs_, false,
                                  verbose, align_bytes);
    rewriter.PlanReuse(stmt, false, enable_aggressive_merge);
    stmt = rewriter(std::move(stmt));
  }
  return stmt;
}

using namespace tirx::transform;

namespace transform {

Pass MergeSharedMemoryAllocations(bool enable_aggressive_merge = false,
                                  int align_bytes = 16) {
  auto pass_func = [enable_aggressive_merge, align_bytes](
                       PrimFunc f, const IRModule &m, PassContext ctx) {
    bool default_merge_static_smem = false;
    Optional<Target> target = f->GetAttr<Target>("target");
    if (target.defined() && target.value()->kind->name == "metal") {
      default_merge_static_smem = true;
    }
    bool merge_static_smem =
        ctx->GetConfig<Bool>("tirx.merge_static_smem",
                             Bool(default_merge_static_smem))
            .value();
    bool debug_merge_shared_memory_allocations =
        ctx->GetConfig<Bool>(kDebugMergeSharedMemoryAllocations, Bool(false))
            .value();
    auto *n = f.CopyOnWrite();
    n->body = tl::MergeSharedMemoryAllocations(
        std::move(n->body), merge_static_smem, enable_aggressive_merge,
        align_bytes, debug_merge_shared_memory_allocations);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeSharedMemoryAllocations",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MergeSharedMemoryAllocations",
                        MergeSharedMemoryAllocations);
}

} // namespace transform
} // namespace tl
} // namespace tvm
