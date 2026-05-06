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
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file loop_vectorize.cc
 * \brief A tool to automatically vectorize a for loop
 */

#include "loop_vectorize.h"
#include "../config.h"
#include "../op/builtin.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include "tvm/tirx/analysis.h"
#include "tvm/tirx/var.h"
#include "vendored/allocate_visit_passthrough.h"
#include "vendored/let_stmt.h"
#include "vendored/z3_constraint_scope.h"
#include "vendored/z3_prover.h"
#include <iostream>
#include <optional>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ::tilelang::tl_tir::LetStmt;
using ::tilelang::tl_tir::LetStmtNode;

/*!
 * \brief Check if buffer strides represent a contiguous (row-major) layout.
 * \param buffer The buffer to check.
 * \param analyzer The analyzer for symbolic comparison.
 * \return True if strides are empty (implicitly contiguous) or match row-major
 * layout.
 */
bool IsBufferContiguous(const Buffer &buffer, arith::Analyzer *analyzer) {
  if (buffer->strides.empty()) {
    return true;
  }
  if (buffer->strides.size() != buffer->shape.size()) {
    return false;
  }
  // For row-major layout:
  // strides[n-1] = 1
  // strides[i] = strides[i+1] * shape[i+1]
  int n = buffer->shape.size();
  PrimExpr expected_stride = make_const(buffer->shape[0].dtype(), 1);
  for (int i = n - 1; i >= 0; --i) {
    if (!analyzer->CanProveEqual(buffer->strides[i], expected_stride)) {
      return false;
    }
    if (i > 0) {
      expected_stride = expected_stride * buffer->shape[i];
    }
  }
  return true;
}

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

struct BufferVectorInfo {
  Buffer buffer;
  int vector_size;
  bool is_store;
  Array<PrimExpr> indices;
  bool is_cast = false; // true for CastNode constraints (vs CallNode)
};

Array<PrimExpr> GetBufferStrides(const Buffer &buffer) {
  if (!buffer->strides.empty()) {
    return buffer->strides;
  }
  Array<PrimExpr> strides;
  PrimExpr stride = 1;
  for (int i = buffer->shape.size() - 1; i >= 0; --i) {
    strides.push_back(stride);
    stride = stride * buffer->shape[i];
  }
  return Array<PrimExpr>{strides.rbegin(), strides.rend()};
}

class VectorizeFindMemoryAccess : public StmtExprVisitor {
public:
  VectorizeFindMemoryAccess() = default;

  bool HasGlobalAccess(const Stmt &stmt) {
    this->operator()(stmt);
    return has_global_access_;
  }

  bool HasSharedAccess(const Stmt &stmt) {
    this->operator()(stmt);
    return has_shared_access_;
  }

  static bool MaySupportVectorize256(const Stmt &stmt) {
    VectorizeFindMemoryAccess visitor;
    visitor(stmt);
    return visitor.has_global_access_ && !visitor.has_shared_access_;
  }

private:
  bool has_global_access_ = false;
  bool has_shared_access_ = false;

  void VisitStmt_(const BufferStoreNode *node) final {
    if (IsGlobalBuffer(node->buffer))
      has_global_access_ = true;
    if (IsSharedBuffer(node->buffer))
      has_shared_access_ = true;
    return StmtExprVisitor::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (IsGlobalBuffer(node->buffer))
      has_global_access_ = true;
    if (IsSharedBuffer(node->buffer))
      has_shared_access_ = true;
    return StmtExprVisitor::VisitExpr_(node);
  }
};

/*!
 * \brief Check if a For loop body contains SeqStmt (multiple statements).
 *
 * When the For body has SeqStmt, the vectorization analysis is more complex
 * and we should be conservative - treating local buffers the same as memory
 * buffers instead of ignoring their constraints.
 *
 * Currently we only handle simple single BufferStore cases specially for
 * local buffer optimization.
 */
bool ForBodyContainsSeqStmt(const For &loop) {
  bool has_seq_stmt = false;
  PostOrderVisit(loop->body, [&](const ObjectRef &obj) {
    if (obj.as<SeqStmtNode>()) {
      has_seq_stmt = true;
    }
  });
  return has_seq_stmt;
}

class VectorizePlanner : public arith::IRMutatorWithAnalyzer {
public:
  explicit VectorizePlanner(arith::Analyzer *analyzer,
                            const LayoutMap &layout_map = {})
      : arith::IRMutatorWithAnalyzer(analyzer), layout_map_(layout_map) {}

  int Plan(const For &node) {
    bool disable_vectorize_256 = tl_config::Vectorize256Disabled();
    bool verbose = tl_config::VectorizePlannerVerboseEnabled();

    if (TargetSupportVectorize256(Target::Current(false)) &&
        !disable_vectorize_256 &&
        VectorizeFindMemoryAccess::MaySupportVectorize256(node)) {
      vector_load_bits_max_ = initial_vector_size_ = loop_extent_vector_size_ =
          256;
    } else {
      vector_load_bits_max_ = initial_vector_size_ = loop_extent_vector_size_ =
          128;
    }

    // Check if For body contains SeqStmt (multiple statements).
    // When there's SeqStmt, we use conservative strategy - treating local
    // buffers the same as memory buffers. The special local buffer optimization
    // (ignoring local buffer constraints) only applies to simple single
    // BufferStore cases.
    bool has_seq_stmt = ForBodyContainsSeqStmt(node);

    // Clear previous buffer info and collect new ones
    buffer_vector_infos_.clear();
    this->operator()(node);

    // Compute final vector size from collected buffer infos
    // Strategy:
    // - If For body contains SeqStmt: take min of all buffers (conservative)
    // - Else if all buffers are local/fragment: take min of all
    // - Else if there are global/shared buffers: ignore local/fragment
    //   constraints and only take min of global/shared buffers
    // Rationale: local/fragment are register-level, no memory alignment
    // constraints. But for complex cases (SeqStmt), we stay conservative.
    vector_size_ = initial_vector_size_;

    if (verbose) {
      std::cerr << "=== VectorizePlanner: Collected buffer vector sizes ==="
                << "\n";
      std::cerr << "  initial_vector_size=" << initial_vector_size_
                << ", loop_extent_vector_size=" << loop_extent_vector_size_
                << ", has_seq_stmt=" << (has_seq_stmt ? "true" : "false")
                << "\n";
    }

    // Separate buffers into local/fragment vs memory (global/shared) vs
    // call/cast
    int local_fragment_min = initial_vector_size_;
    int memory_min = initial_vector_size_;
    int call_node_min = initial_vector_size_;
    int non_cast_call_node_min = initial_vector_size_;
    bool has_global_or_shared_buffer = false;

    auto is_local_or_fragment = [](const Buffer &buf) {
      return IsLocalBuffer(buf, /*allow_var=*/true) || IsFragmentBuffer(buf);
    };

    std::vector<BufferVectorInfo> local_fragment_buffers;

    for (const auto &info : buffer_vector_infos_) {
      auto buffer = info.buffer;
      if (verbose) {
        if (buffer.defined()) {
          std::cerr << "  Buffer: " << buffer->name
                    << " (scope=" << buffer.scope() << ")"
                    << " -> vector_size=" << info.vector_size
                    << (info.is_store ? " [store]" : " [load]") << "\n";
        } else {
          std::cerr << "  [" << (info.is_cast ? "cast" : "call")
                    << "] -> vector_size=" << info.vector_size << "\n";
        }
      }
      if (!buffer.defined()) {
        call_node_min = arith::ZeroAwareGCD(call_node_min, info.vector_size);
        if (!info.is_cast) {
          non_cast_call_node_min =
              arith::ZeroAwareGCD(non_cast_call_node_min, info.vector_size);
        }
      } else if (is_local_or_fragment(buffer)) {
        local_fragment_min =
            arith::ZeroAwareGCD(local_fragment_min, info.vector_size);
        local_fragment_buffers.push_back(info);
      } else {
        // global, shared, shared.dyn
        // If a *load*'s indices don't depend on loop var (e.g. b[0]), treat
        // as local — it will become a scalar broadcast, not a vector memory
        // access, and DecoupleTypeCast won't create a cast buffer for it.
        // Stores must stay in the memory bucket: a loop-invariant store is a
        // reduction-like pattern where ComputeBufferVectorSize has already
        // returned 1 to disable vectorization, and that constraint must not
        // be dropped (memory strategy ignores local_fragment_min).
        bool depends_on_loop_var = true;
        if (!info.indices.empty() && inner_for_) {
          Array<PrimExpr> strides = GetBufferStrides(info.buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < info.indices.size(); ++i) {
            elem_offset += info.indices[i] * strides[i];
          }
          depends_on_loop_var = !IsExprInvariantInVectorBoundary(
              elem_offset, inner_for_->loop_var, vector_size_, analyzer_);
        }
        if (depends_on_loop_var) {
          memory_min = arith::ZeroAwareGCD(memory_min, info.vector_size);
          has_global_or_shared_buffer = true;
        } else {
          local_fragment_min =
              arith::ZeroAwareGCD(local_fragment_min, info.vector_size);
          local_fragment_buffers.push_back(info);
        }
      }
    }

    if (verbose) {
      std::cerr << "  Computed mins: local_fragment_min=" << local_fragment_min
                << ", memory_min=" << memory_min
                << ", call_node_min=" << call_node_min << "\n";
    }

    if (has_seq_stmt) {
      // For body contains SeqStmt (multiple statements).
      // Use conservative strategy: take GCD of all buffers including local.
      // The special local buffer optimization only applies to simple single
      // BufferStore cases where we can be confident about the access pattern.
      vector_size_ = arith::ZeroAwareGCD(
          arith::ZeroAwareGCD(local_fragment_min, memory_min), call_node_min);
      if (verbose) {
        std::cerr << "  [Strategy] Has SeqStmt, using conservative GCD of all"
                  << " -> vector_size=" << vector_size_ << "\n";
      }
    } else if (has_global_or_shared_buffer) {
      // Has memory buffers and simple case (no SeqStmt):
      // ignore local/fragment constraints AND cast constraints.
      // Cast constraints are ignored because DecoupleTypeCast will later
      // split mixed-type operations into separate loops, allowing memory
      // copies to use wider vectors independently of cast width limits.
      vector_size_ = arith::ZeroAwareGCD(memory_min, non_cast_call_node_min);
      if (verbose) {
        std::cerr << "  [Strategy] Has memory buffers (simple case), using "
                  << "memory_min=" << memory_min
                  << ", non_cast_call_node_min=" << non_cast_call_node_min
                  << " (ignoring local/fragment_min=" << local_fragment_min
                  << ")" << "\n";
      }
      // vector_size may be greater than local/fragment buffers' vector_size.
      // In such case, we need to re-validate if the indices are vectorizable
      // at the new vector_size boundary. If not, take GCD.
      for (const auto &info : local_fragment_buffers) {
        if (vector_size_ > info.vector_size && !info.indices.empty()) {
          // Compute elem_offset from indices and strides
          Array<PrimExpr> strides = GetBufferStrides(info.buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < info.indices.size(); ++i) {
            elem_offset += info.indices[i] * strides[i];
          }
          if (!IndicesCanVectorize(elem_offset, inner_for_->loop_var,
                                   inner_for_->extent, vector_size_,
                                   analyzer_)) {
            // Not invariant at this vector_size, need to take GCD
            int old_vector_size = vector_size_;
            vector_size_ = arith::ZeroAwareGCD(vector_size_, info.vector_size);
            if (verbose) {
              std::cerr << "  [Re-validate] Local buffer '" << info.buffer->name
                        << "' not invariant at vector_size=" << old_vector_size
                        << ", GCD with " << info.vector_size
                        << " -> vector_size=" << vector_size_ << "\n";
            }
          }
        }
      }
    } else {
      // Only local/fragment buffers: use GCD of local_fragment_min and
      // call_node_min
      vector_size_ = arith::ZeroAwareGCD(local_fragment_min, call_node_min);
      if (verbose) {
        std::cerr << "  [Strategy] Only local/fragment buffers, using "
                     "GCD(local_fragment_min, call_node_min)="
                  << vector_size_ << "\n";
      }
    }

    // GCD with loop extent to ensure vector_size divides the loop extent
    vector_size_ = arith::ZeroAwareGCD(loop_extent_vector_size_, vector_size_);

    if (verbose) {
      std::cerr << "=== Final vector_size: " << vector_size_ << " ===" << "\n";
    }
    return vector_size_;
  }

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    bool contains_nested_for = false;
    // Must analysis vectorization on the innermost loop
    PostOrderVisit(Downcast<Stmt>(node->body), [&](const ObjectRef &obj) {
      if (obj.as<ForNode>()) {
        contains_nested_for = true;
      }
    });

    if (!contains_nested_for) {
      auto extent_ptr = as_const_int(analyzer_->Simplify(node->extent));
      // Here I disable dynamic shape completely,
      //   In order to do it, the Planner should accept an analyzer with
      //   arithmetic info outside to prove the dividiblity of vector size
      // Note(lei): This is somehow make sense because we should assume the
      // tiling size is always static.
      if (!extent_ptr) {
        loop_extent_vector_size_ = 1;
        return ffi::GetRef<Stmt>(node);
      }
      loop_extent_vector_size_ =
          arith::ZeroAwareGCD(initial_vector_size_, *extent_ptr);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *node) final {
    if (IsSharedBuffer(node->buffer) || IsGlobalBuffer(node->buffer))
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      auto boundary_check = node->buffer->shape[0].as<IntImmNode>();
      if (boundary_check && boundary_check->value == 1) {
        return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
      }
    }
    UpdateVectorSize(node->indices, node->buffer, false);
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  Stmt VisitStmt_(const BufferStoreNode *node) final {
    if (IsSharedBuffer(node->buffer) || IsGlobalBuffer(node->buffer))
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer, true);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  Stmt VisitStmt_(const IfThenElseNode *node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  static std::optional<int> GetAccessPtrElementBits(const PrimExpr &expr) {
    const auto *ptr_call = expr.as<CallNode>();
    if (ptr_call == nullptr) {
      return std::nullopt;
    }
    if (ptr_call->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK(!ptr_call->args.empty());
      DataType dtype = ptr_call->args[0].dtype();
      return dtype.bits() * dtype.lanes();
    }
    if (ptr_call->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(ptr_call->args.size(), 3U)
          << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
      const auto *buffer_load = ptr_call->args[0].as<BufferLoadNode>();
      ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
      DataType dtype = buffer_load->buffer->dtype;
      return dtype.bits() * dtype.lanes();
    }
    return std::nullopt;
  }

  static std::optional<int> GetCPAsyncBitsPerCall(const CallNode *node) {
    ICHECK_GE(node->args.size(), 3U)
        << "cp.async expects at least 3 arguments, but got " << node->args;
    const auto *count_imm = node->args[2].as<IntImmNode>();
    ICHECK(count_imm) << "cp.async transfer count must be IntImm, but got "
                      << node->args[2];
    int count = static_cast<int>(count_imm->value);
    if (count <= 0) {
      return std::nullopt;
    }
    if (node->op.same_as(builtin::ptx_cp_async())) {
      return count * 8;
    }
    ICHECK(node->op.same_as(tl::ptx_cp_async()));
    auto dst_elem_bits = GetAccessPtrElementBits(node->args[0]);
    auto src_elem_bits = GetAccessPtrElementBits(node->args[1]);
    if (!dst_elem_bits.has_value() || !src_elem_bits.has_value()) {
      return std::nullopt;
    }
    int dst_total_bits = count * dst_elem_bits.value();
    int src_total_bits = count * src_elem_bits.value();
    ICHECK_EQ(dst_total_bits, src_total_bits)
        << "tl.ptx_cp_async requires src/dst transfer widths to match, but got "
        << dst_total_bits << " vs " << src_total_bits << " bits";
    return dst_total_bits;
  }

  static int GetMaxCPAsyncVectorizeLength(int per_call_bits) {
    if (per_call_bits <= 0) {
      return 1;
    }
    int vectorize_length = 1;
    for (int target_bytes : {16, 8, 4}) {
      int target_bits = target_bytes * 8;
      if (target_bits % per_call_bits == 0) {
        vectorize_length =
            std::max(vectorize_length, target_bits / per_call_bits);
      }
    }
    return vectorize_length;
  }

  PrimExpr VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(builtin::tvm_access_ptr())) {
      HandleTvmAccessPtr(node);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == tl::atomic_add_elem_op()) {
      // Assert at least 2 args (dst_ptr and src)
      ICHECK(node->args.size() >= 2)
          << "atomic_add_elem_op requires at least 2 args (dst and src)";

      // Get dst dtype from args[0] (tvm_access_ptr or address_of(BufferLoad))
      const CallNode *dst_ptr_call = node->args[0].as<CallNode>();
      ICHECK(dst_ptr_call) << "atomic_add_elem_op first arg must be a call";

      DataType dtype;
      if (dst_ptr_call->op.same_as(builtin::address_of())) {
        auto buffer_load = dst_ptr_call->args[0].as<BufferLoadNode>();
        ICHECK(buffer_load) << "address_of arg must be BufferLoad";
        dtype = buffer_load->buffer->dtype;
      } else if (dst_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK(!dst_ptr_call->args.empty());
        dtype = dst_ptr_call->args[0].dtype();
      } else if (dst_ptr_call->op.same_as(tl::access_ptr())) {
        ICHECK_EQ(dst_ptr_call->args.size(), 3U)
            << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
        auto buffer_load = dst_ptr_call->args[0].as<BufferLoadNode>();
        ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
        dtype = buffer_load->buffer->dtype;
      } else {
        LOG(FATAL) << "atomic_add_elem_op first arg must be tvm_access_ptr, "
                      "tl.access_ptr, or address_of call, but got "
                   << node->args[0];
      }
      int vectorize_length = 1;
      if (dtype.is_float16() || dtype.is_bfloat16()) {
        vectorize_length = 2;
      } else if (dtype.is_float() && dtype.bits() == 32 &&
                 TargetHasSMVersionGE(Target::Current(false), 90)) {
        vectorize_length = 4;
      }

      buffer_vector_infos_.push_back({Buffer(), vectorize_length, false, {}});
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(builtin::ptx_cp_async()) ||
               node->op.same_as(tl::ptx_cp_async())) {
      // builtin::ptx_cp_async stores bytes, while tl::ptx_cp_async stores
      // logical element counts. In both cases we pick the largest vector width
      // whose eventual PTX payload is one of {4, 8, 16} bytes.
      int vectorize_length =
          GetMaxCPAsyncVectorizeLength(GetCPAsyncBitsPerCall(node).value_or(0));
      buffer_vector_infos_.push_back({Buffer(), vectorize_length, false, {}});
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == builtin::address_of() ||
               node->op == tl::access_ptr()) {
      // address_of and tl.access_ptr have buffer load value so we should
      // analysis the buffer load node to update vector_size_.
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }

    // vectorizable property
    OpAttrMap<TVectorizable> op_vectorizable_ =
        Op::GetAttrMap<TVectorizable>("TVectorizable");

    auto optional_op = node->op.as<Op>();
    bool vectorizable = op_vectorizable_.get(optional_op.value(), false) &&
                        !node->dtype.is_scalable_vector();
    if (vectorizable) {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }

    // For other call nodes, use PostOrderVisit to check buffer accesses
    // and determine if the given vector size is invariant
    auto check_buffer_access_invariant = [&](int target_vec_size) -> bool {
      if (!inner_for_)
        return true;
      bool all_invariant = true;
      PostOrderVisit(ffi::GetRef<PrimExpr>(node), [&](const ObjectRef &obj) {
        if (!all_invariant)
          return;
        if (auto *load = obj.as<BufferLoadNode>()) {
          auto transformed_indices =
              TransformIndices(load->indices, load->buffer);
          Array<PrimExpr> strides = GetBufferStrides(load->buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < transformed_indices.size(); ++i) {
            elem_offset += transformed_indices[i] * strides[i];
          }
          if (!IsExprInvariantInVectorBoundary(elem_offset,
                                               inner_for_->loop_var,
                                               target_vec_size, analyzer_)) {
            all_invariant = false;
          }
        } else if (auto *store = obj.as<BufferStoreNode>()) {
          auto transformed_indices =
              TransformIndices(store->indices, store->buffer);
          Array<PrimExpr> strides = GetBufferStrides(store->buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < transformed_indices.size(); ++i) {
            elem_offset += transformed_indices[i] * strides[i];
          }
          if (!IsExprInvariantInVectorBoundary(elem_offset,
                                               inner_for_->loop_var,
                                               target_vec_size, analyzer_)) {
            all_invariant = false;
          }
        } else if (auto *call = obj.as<CallNode>()) {
          // tvm_access_ptr(dtype_annotation, data, offset, extent, rw_mask)
          // The offset (args[2]) is the element offset into the buffer.
          if (call->op.same_as(builtin::tvm_access_ptr()) &&
              call->args.size() >= 3) {
            PrimExpr offset = call->args[2];
            if (!IsExprInvariantInVectorBoundary(offset, inner_for_->loop_var,
                                                 target_vec_size, analyzer_)) {
              all_invariant = false;
            }
          }
        }
      });
      return all_invariant;
    };
    // Find the largest vector size where all buffer accesses are invariant
    int call_node_vector_size = loop_extent_vector_size_;
    while (call_node_vector_size > 1) {
      if (check_buffer_access_invariant(call_node_vector_size)) {
        break;
      }
      call_node_vector_size /= 2;
    }
    buffer_vector_infos_.push_back(
        {Buffer(), call_node_vector_size, false, {}});
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr &cond) {
    // TODO: perform some checks here
  }

  void HandleTvmAccessPtr(const CallNode *node) {
    // tvm_access_ptr format: (ptype, data, offset, extent, rw_mask)
    if (!inner_for_) {
      return;
    }
    ICHECK(node->args.size() >= 3U)
        << "tvm_access_ptr requires at least 3 args";

    // args[0] is TypeAnnotation(dtype[/lanes]); dtype() encodes the element
    // type. See tvm::tirx::Buffer::access_ptr implementation.
    DataType dtype = node->args[0].dtype();
    Var data_var;
    if (auto data_var_node = node->args[1].as<VarNode>()) {
      data_var = Downcast<Var>(node->args[1]);
    }
    ICHECK(data_var.defined()) << "tvm_access_ptr second arg must be a var";
    PrimExpr offset = node->args[2];

    Optional<Buffer> buffer_opt;
    Optional<Layout> layout_opt;

    // Find the Buffer whose data pointer matches data_var by searching
    // layout_map_. The layout_map_ maps Buffer -> Layout, so we iterate
    // to find the buffer whose ->data field is the same Var.
    if (layout_map_.defined()) {
      for (auto [buf, layout] : layout_map_) {
        if (buf->data.same_as(data_var)) {
          buffer_opt = buf;
          layout_opt = layout;
          break;
        }
      }
    }

    // Base vector size from loop extent.
    int access_vec_size = loop_extent_vector_size_;
    // Constrain by dtype lane capacity (128/256-bit vector load/store width).
    // This mirrors ComputeBufferVectorSize's dtype-based lower bound.
    int dtype_bits = dtype.bits() * dtype.lanes();
    if (dtype_bits > 0) {
      int dtype_lane_bound = vector_load_bits_max_ / dtype_bits;
      if (dtype_lane_bound <= 0) {
        dtype_lane_bound = 1;
      }
      access_vec_size = arith::ZeroAwareGCD(access_vec_size, dtype_lane_bound);
    }

    // If the buffer has a layout, use the last output dimension as a proxy for
    // the maximum contiguous vector length implied by the layout.
    if (layout_opt.defined()) {
      Array<PrimExpr> out_shape = layout_opt.value()->OutputShape();
      if (!out_shape.empty()) {
        PrimExpr contig = analyzer_->Simplify(out_shape.back());
        if (auto contig_int = as_const_int(contig);
            contig_int && *contig_int > 1) {
          access_vec_size = arith::ZeroAwareGCD(access_vec_size, *contig_int);
        }
      }
    }
    // tvm_access_ptr itself is not vectorizable in TLVectorizer. If its offset
    // depends on the vectorized loop var, TLVectorizer will force scalarization
    // of the whole loop body. To avoid planning a vector size that will be
    // immediately scalarized (and to keep semantics sane for side-effectful
    // calls), require the offset to be invariant within the vector boundary.
    PrimExpr offset_s = analyzer_->Simplify(offset);
    while (access_vec_size > 1 &&
           !IndicesCanVectorize(offset_s, inner_for_->loop_var,
                                inner_for_->extent, access_vec_size,
                                analyzer_)) {
      access_vec_size /= 2;
    }
    // Record as a memory-like constraint if we can resolve the buffer.
    buffer_vector_infos_.push_back(
        {buffer_opt.value_or(Buffer()), access_vec_size, false, {}});
  }

  Array<PrimExpr> TransformIndices(const Array<PrimExpr> &indices,
                                   const Buffer &buffer) {
    auto transformed_indices = indices;
    if (layout_map_.defined() && layout_map_.count(buffer)) {
      ICHECK(IsBufferContiguous(buffer, analyzer_))
          << buffer
          << " has non-contiguous strides, but layout map is provided.";
      // forward indices
      auto layout = layout_map_[buffer];
      transformed_indices = layout->Forward(indices);
      // Reshape transformed_indices to match buffer->shape dimensions if needed
      if (transformed_indices.size() != buffer->shape.size()) {
        // Step 1: Compute linear offset using layout->OutputShape()
        auto output_shape = layout->OutputShape();
        ICHECK_EQ(transformed_indices.size(), output_shape.size())
            << "Forward indices size " << transformed_indices.size()
            << " != OutputShape size " << output_shape.size();
        PrimExpr linear_offset = 0;
        PrimExpr stride = 1;
        for (int i = output_shape.size() - 1; i >= 0; --i) {
          linear_offset = linear_offset + transformed_indices[i] * stride;
          stride = stride * output_shape[i];
        }
        // Step 2: Decompose linear_offset into buffer->shape dimensions
        Array<PrimExpr> new_indices;
        for (int i = buffer->shape.size() - 1; i >= 0; --i) {
          new_indices.push_back(FloorMod(linear_offset, buffer->shape[i]));
          linear_offset = FloorDiv(linear_offset, buffer->shape[i]);
        }
        transformed_indices =
            Array<PrimExpr>{new_indices.rbegin(), new_indices.rend()};
      }
    }
    return transformed_indices;
  }

  PrimExpr VisitExpr_(const CastNode *node) final {
    // Consider both source and target types to ensure all intermediate
    // vector types can be represented. For example, casting int32 to
    // float8_e4m3fn: target allows 128/8=16 lanes but int32 only supports
    // up to 128/32=4 lanes in CUDA vector types.
    int target_lanes = vector_load_bits_max_ / node->dtype.bits();
    int source_bits = node->value.dtype().bits();
    int max_lanes = target_lanes;
    if (source_bits > 0) {
      int source_lanes = vector_load_bits_max_ / source_bits;
      max_lanes = std::min(target_lanes, source_lanes);
    }
    int cast_vector_size = arith::ZeroAwareGCD(max_lanes, initial_vector_size_);
    // Record cast constraint (use empty buffer to indicate cast)
    // Mark is_cast=true so Plan() can distinguish cast from other call nodes
    buffer_vector_infos_.push_back(
        {Buffer(), cast_vector_size, false, {}, /*is_cast=*/true});
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  int ComputeBufferVectorSize(const Array<PrimExpr> &indices,
                              const Buffer &buffer, bool is_store) {
    if (!inner_for_)
      return initial_vector_size_;

    int buffer_vec_size = loop_extent_vector_size_;

    // Transform indices using layout_map if present
    auto transformed_indices = TransformIndices(indices, buffer);

    // 1. Compute raw element offset
    Array<PrimExpr> strides = GetBufferStrides(buffer);

    PrimExpr elem_offset = 0;
    for (size_t i = 0; i < transformed_indices.size(); ++i) {
      elem_offset += transformed_indices[i] * strides[i];
    }

    // 2. Check if current buffer_vec_size works with invariant boundary check
    // In some cases, buffer_vec_size is max (e.g. 128), but
    // IsExprInvariantInVectorBoundary may only be true at a smaller size (e.g.
    // 64). Recursively halve buffer_vec_size until we find a size where
    // is_invariant is true. Fallback: minimum vector size based on buffer dtype
    int min_vec_size = arith::ZeroAwareGCD(
        buffer_vec_size,
        vector_load_bits_max_ / (buffer->dtype.bits() * buffer->dtype.lanes()));
    bool is_invariant = false;
    int try_vec_size = buffer_vec_size;
    while (try_vec_size >= min_vec_size) {
      is_invariant = IsExprInvariantInVectorBoundary(
          elem_offset, inner_for_->loop_var, try_vec_size, analyzer_);
      if (is_invariant) {
        buffer_vec_size = try_vec_size;
        break;
      }
      try_vec_size /= 2;
    }
    // If is_invariant is still false, use the fallback min_vec_size
    if (!is_invariant) {
      buffer_vec_size = min_vec_size;
    }

    // 3. If element offset is independent with loop_var, ignore it.
    bool is_independent =
        CanProveIndependent(elem_offset, inner_for_->loop_var, analyzer_);
    // For BufferStore, if indices is invariant or independent with loop_var,
    // we should not vectorize it (broadcasting store is not supported).
    if (is_store && (is_invariant || is_independent)) {
      return 1;
    }
    if (is_independent) {
      return buffer_vec_size; // only limited constraint from this buffer
    }
    // 4. Try to find max vectorize size for this buffer
    while (buffer_vec_size > 1 &&
           !IndicesCanVectorize(elem_offset, inner_for_->loop_var,
                                inner_for_->extent, buffer_vec_size,
                                analyzer_)) {
      buffer_vec_size /= 2;
    }
    return buffer_vec_size;
  }

  void UpdateVectorSize(const Array<PrimExpr> &indices, const Buffer &buffer,
                        bool is_store) {
    int buffer_vec_size = ComputeBufferVectorSize(indices, buffer, is_store);
    buffer_vector_infos_.push_back(
        {buffer, buffer_vec_size, is_store, indices});
  }

  // NOTE(wt): The base class IRMutatorWithAnalyzer::VisitStmt_(LetStmtNode*)
  // binds let variables, but this causes issues when the same variable name
  // appears multiple times with different values (e.g., in pipelined loops
  // where the body is duplicated). For this case, we allow the analyzer to
  // override the binding.
  //
  // apache/tvm StmtFunctor vtable does not dispatch to the vendored
  // `tilelang::tl_tir::LetStmtNode`. Override the top-level
  // `VisitStmt(const Stmt&)` to intercept the vendored type before the
  // built-in vtable dispatch.
  Stmt VisitStmt(const Stmt &stmt) override {
    if (const auto *op = stmt.as<LetStmtNode>()) {
      PrimExpr value = this->VisitExpr(op->value);
      if (SideEffect(value) <= CallEffectKind::kPure) {
        // Allow override to handle duplicated loop bodies in pipelined loops
        analyzer_->Bind(op->var, value, /*allow_override=*/true);
      }
      // Continue visiting the body to collect vectorization info
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      } else {
        return LetStmt(op->var, std::move(value), std::move(body), op->span);
      }
    }
    if (auto out = ::tilelang::tl_tir::TryVisitAllocateMutator(this, stmt)) {
      return *out;
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt(stmt);
  }

  int vector_load_bits_max_;
  int initial_vector_size_ = 128;
  int loop_extent_vector_size_ = 128;

  const ForNode *inner_for_{};
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
  std::vector<BufferVectorInfo> buffer_vector_infos_;
  LayoutMap layout_map_;
};

// CPPMEGA: Z3 idea #12 — alignment proof companion to #1 (contiguity).
//
// After vectorization decides to mark a For as kVectorized with a given
// `vector_size`, optionally try to prove that the buffer base address
// (in BYTES) of every memory access in the loop body is aligned to
// `vector_size * dtype_bytes`. If proven for at least one global/shared
// access, attach a `tl.vec_aligned` annotation on the inner For so
// downstream codegen (MSL `vec.load_aligned`/CUDA `ld.global.v4.b32`)
// can elide the unaligned-load fallback.
//
// Conservative on UNKNOWN/timeout: no annotation. The annotation is
// purely additive — no semantic change unless the codegen explicitly
// reads the attr. PassConfig: `tl.vectorize_alignment_proof` (default OFF).
//
// The proof query, in BV/integer form:
//
//     forall free_vars in BV32:
//       FloorMod(elem_offset_bytes, vector_size * dtype_bytes) == 0
//
// where `elem_offset_bytes = (buffer.elem_offset + index_offset) * dtype_bytes`.
// `index_offset` is the same `elem_offset` expression that the contiguity
// path computes (sum of `indices[i] * strides[i]`), but evaluated at the
// loop variable's *base value* (so that `var % vector_size == 0` is implicit).
//
// Bit-bound free vars to BV32 emulation via EnterConstraint.
static bool Z3CanProveAlignedAccess(const Buffer &buffer,
                                    const Array<PrimExpr> &indices,
                                    const Var &loop_var, int vector_size,
                                    arith::Analyzer *analyzer) {
  if (!buffer.defined() || indices.empty()) {
    return false;
  }
  int dtype_bytes = buffer->dtype.bytes();
  if (dtype_bytes <= 0) {
    return false;
  }
  int64_t alignment = static_cast<int64_t>(vector_size) * dtype_bytes;
  if (alignment <= 0) {
    return false;
  }

  // Compute element offset (in elements). Use the buffer's strides if
  // present, otherwise derive a row-major stride layout.
  Array<PrimExpr> strides = GetBufferStrides(buffer);
  if (strides.size() != indices.size()) {
    return false;
  }
  PrimExpr elem_offset = make_const(DataType::Int(32), 0);
  for (size_t i = 0; i < indices.size(); ++i) {
    elem_offset = elem_offset + indices[i] * strides[i];
  }
  // Add the buffer's own elem_offset (in elements) — this captures the
  // base offset for sub-buffers / views.
  if (buffer->elem_offset.defined()) {
    elem_offset = elem_offset + cast(elem_offset.dtype(), buffer->elem_offset);
  }
  // Convert to bytes.
  PrimExpr base_addr_bytes =
      elem_offset * make_const(elem_offset.dtype(), dtype_bytes);

  try {
    auto &z3 = arith::Z3Prover(analyzer);
    z3.SetTimeoutMs(50);

    // Bit-bound: every free Var (including the loop var itself) is
    // assumed to be in [0, 2^31). For the loop var specifically, also
    // assume it's a multiple of `vector_size` — because the
    // VectorizeRewriter has already split the loop so the inner var
    // ranges over [0, vector_size) and the *outer* var ranges over
    // [0, extent/vector_size) — but the per-iteration alignment goal is
    // about the START of each vector lane, i.e. when var == 0 modulo
    // vector_size. We collect free vars from the elem_offset.
    std::unordered_set<const VarNode *> free_vars;
    PostOrderVisit(elem_offset, [&](const ObjectRef &obj) {
      if (const auto *v = obj.as<VarNode>()) {
        free_vars.insert(v);
      }
    });
    // CPPMEGA fix-B2 (idea712): use ConstraintScope RAII so EnterConstraint
    // push/pop is balanced even on early-return / exception. The previous
    // manual recoverers vector leaked solver scope frames if any
    // EnterConstraint or CanProve below threw.
    std::vector<::tilelang::tlz3::ConstraintScope> scopes;
    for (const VarNode *v : free_vars) {
      Var var = ffi::GetRef<Var>(v);
      DataType dt = var.dtype();
      if (!dt.is_int() && !dt.is_uint()) {
        // Non-int free var — cannot bit-bound. Bail. RAII unwinds scopes.
        return false;
      }
      // CPPMEGA fix-B4 (idea712): dtype-aware BV bounds. The flat
      // [0, 2^31) bound was unsound for any signed-int var that may
      // hold a negative offset (e.g. sub-buffer biases or
      // `LegalizeNegativeIndex`-rewritten loads). Use the dtype's
      // proper signed/unsigned range instead.
      auto [lo64, hi64] = ::tilelang::tlz3::BVBoundsForDtype(dt);
      PrimExpr lo = make_const(dt, lo64);
      PrimExpr hi = make_const(dt, hi64);
      PrimExpr bound = (var >= lo) && (var < hi);
      // For the loop var: pin lo to 0 (loop vars are non-negative by
      // construction) and assume it's a multiple of vector_size — the
      // VectorizeRewriter's invariant for the outer-loop iteration
      // boundary at the START of each vector chunk. Goal: prove the
      // address at lane 0 of every vector chunk is aligned.
      if (v == loop_var.get()) {
        bound = (var >= make_const(dt, 0)) && (var < hi) &&
                (FloorMod(var, make_const(dt, vector_size)) ==
                 make_const(dt, 0));
      }
      scopes.emplace_back(z3, bound);
    }

    PrimExpr goal =
        FloorMod(base_addr_bytes,
                 make_const(base_addr_bytes.dtype(), alignment)) ==
        make_const(base_addr_bytes.dtype(), 0);

    bool proved = false;
    try {
      proved = z3.CanProve(goal);
    } catch (...) {
      proved = false;
    }
    // ConstraintScope destructors run in reverse order at scope exit.
    return proved;
  } catch (...) {
    return false;
  }
}

// Walk the loop body and attempt to prove alignment for every
// global/shared BufferLoad and BufferStore. Returns true iff there is at
// least one such access AND every one is provably aligned.
static bool Z3CanProveLoopAligned(const Stmt &body, const Var &loop_var,
                                  int vector_size,
                                  arith::Analyzer *analyzer) {
  bool saw_memory = false;
  bool all_aligned = true;
  PostOrderVisit(body, [&](const ObjectRef &obj) {
    if (!all_aligned) return;
    if (const auto *ld = obj.as<BufferLoadNode>()) {
      if (IsLocalBuffer(ld->buffer, /*allow_var=*/true) ||
          IsFragmentBuffer(ld->buffer)) {
        return; // local/fragment access — alignment irrelevant.
      }
      saw_memory = true;
      if (!Z3CanProveAlignedAccess(ld->buffer, ld->indices, loop_var,
                                   vector_size, analyzer)) {
        all_aligned = false;
      }
    } else if (const auto *st = obj.as<BufferStoreNode>()) {
      if (IsLocalBuffer(st->buffer, /*allow_var=*/true) ||
          IsFragmentBuffer(st->buffer)) {
        return;
      }
      saw_memory = true;
      if (!Z3CanProveAlignedAccess(st->buffer, st->indices, loop_var,
                                   vector_size, analyzer)) {
        all_aligned = false;
      }
    }
  });
  return saw_memory && all_aligned;
}

// Build a new annotation map with `tl.vec_aligned -> True` added.
static Map<String, ffi::Any> MakeAlignedAnnotations(
    const Map<String, ffi::Any> &existing) {
  Map<String, ffi::Any> out = existing;
  out.Set("tl.vec_aligned", Bool(true));
  return out;
}

class VectorizeRewriter : public StmtExprMutator {
public:
  VectorizeRewriter(int vector_size) : vector_size_(vector_size) {}

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ == node) { // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_
          << " for loop: " << fnode;
      ICHECK(is_zero(fnode->min));

      // CPPMEGA: Z3 idea #12 — best-effort alignment proof. Default OFF.
      bool alignment_proof_enabled = false;
      try {
        alignment_proof_enabled =
            tvm::transform::PassContext::Current()
                ->GetConfig<Bool>(kVectorizeAlignmentProof, Bool(false))
                .value();
      } catch (...) {
        alignment_proof_enabled = false;
      }
      bool aligned = false;
      if (alignment_proof_enabled) {
        arith::Analyzer analyzer;
        try {
          aligned = Z3CanProveLoopAligned(fnode->body, fnode->loop_var,
                                          vector_size_, &analyzer);
        } catch (...) {
          aligned = false;
        }
      }

      if (extent == vector_size_) {
        fnode.CopyOnWrite()->kind = ForKind::kVectorized;
        if (aligned) {
          fnode.CopyOnWrite()->annotations =
              MakeAlignedAnnotations(fnode->annotations);
        }
        return fnode;
      } else {
        Var inner_var = Var("vec");
        Var outer_var = Var(old_var->name_hint);
        Map<Var, PrimExpr> vmap;
        vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
        Stmt body = Substitute(fnode->body, vmap);
        Map<String, ffi::Any> inner_annotations;
        if (aligned) {
          inner_annotations =
              MakeAlignedAnnotations(Map<String, ffi::Any>());
        }
        body = For(inner_var, 0, vector_size_, ForKind::kVectorized, body,
                   /*thread_binding=*/std::nullopt, inner_annotations);
        // TileLang uses ForKind::kParallel in frontend SIMT loops. After
        // vectorization, keep semantics equivalent but downgrade to serial so
        // subsequent passes (e.g. pragma-unroll) can run.
        ForKind outer_kind = fnode->kind;
        if (outer_kind == ForKind::kParallel) {
          outer_kind = ForKind::kSerial;
        }
        body = For(outer_var, 0, extent / vector_size_, outer_kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->step,
                   fnode->span);
        return body;
      }
    } else {
      // Keep other loops intact, except for TileLang frontend "parallel" loops
      // which should behave as serial loops after lowering.
      For loop = ret.as<For>().value();
      if (loop->kind == ForKind::kParallel) {
        loop.CopyOnWrite()->kind = ForKind::kSerial;
      }
      return loop;
    }
  }

  const ForNode *inner_for_{};
  const int vector_size_;
};

int GetVectorizeSize(const For &loop, const LayoutMap &layout_map) {
  arith::Analyzer analyzer;
  return VectorizePlanner(&analyzer, layout_map).Plan(loop);
}

int GetVectorizeSize(const For &loop, arith::Analyzer *analyzer,
                     const LayoutMap &layout_map) {
  return VectorizePlanner(analyzer, layout_map).Plan(loop);
}

bool CanProveIndependent(const PrimExpr &expr, Var var,
                         arith::Analyzer *analyzer) {
  // 1. if var doesn't exist, it is independent
  bool used_var = UsesVar(expr, [&](const VarNode *v) {
    return tvm::ffi::GetRef<Var>(v).same_as(var);
  });
  if (!used_var) {
    return true;
  }
  // 2. if \forall v_1, v_2, f(v_1) == f(v_2), f is independent with v
  Var var_1("_t", var.dtype());
  auto expr_1 = Substitute(expr, {{var, var_1}});
  if (analyzer->CanProveEqual(expr, expr_1)) {
    return true;
  }
  return false;
}

bool IsExprInvariantInVectorBoundary(const PrimExpr &expr, Var var,
                                     int target_vectorized_size,
                                     arith::Analyzer *analyzer) {
  // Check if expr is invariant within vector boundaries
  // We're trying to prove the access expression A[f(var)] depends only on
  // floor(var/vecsize), not on var%vecsize
  // Mathematically:
  // \forall var, f(floor(var/vecsize)*vecsize + var%vecsize) ==
  // f(floor(var/vecsize)*vecsize + 0)
  // Example: for i in T.vectorized(8):
  //     A[i] = B[i] * C[i//4]
  // if vecsize=4, f(i)=i//4 depends only on i//4
  // Therefore A[i] = B[i] * C[i//4] can be vectorized with vecsize=4
  PrimExpr var_aligned =
      floordiv(var, target_vectorized_size) * target_vectorized_size;
  PrimExpr expr_aligned = Substitute(expr, {{var, var_aligned}});
  if (analyzer->CanProveEqual(expr, expr_aligned)) {
    return true;
  }
  return false;
}

// CPPMEGA: Z3 idea #1 — affine-pattern guard.
//
// Audit fix (HIGH): the unit-stride proof works by substituting `var -> var+1`
// and asking Z3 whether `(expr_next - expr) == 1`. For non-affine accesses
// like `A[B[i]]` (where `B[i]` is a separate BufferLoad), the substitution
// only rewrites the outer `i`, leaving the inner BufferLoad unchanged — so
// the difference simplifies to `B[i+1] - B[i]`, which Z3 cannot disprove
// being 1 in general (especially with no constraints on `B`). Worse, a
// stub-shaped Z3 expression for an opaque BufferLoad can sometimes be
// proven equal in trivial models, leading to a false positive.
//
// Conservative fix: walk `expr` first; if it contains *any* BufferLoad or
// any non-{Add,Sub,Mul,IntImm,Var} arithmetic node, return false and skip
// the Z3 fallback entirely. This shrinks the Z3 call domain to syntactic
// affine functions of `var` (plus arbitrary loop-invariants captured by
// other Vars), where the substitution-and-subtract trick is sound.
//
// We allow Vars other than `var` to appear: those are loop-invariant and
// will cancel in the subtraction. We disallow FloorDiv/FloorMod/etc. for
// safety — accesses like `var * stride / N` have stride only when `stride
// == N` and the simplifier should have already canonicalised that.
static bool IsAffineInVar(const PrimExpr &expr, const Var &var) {
  (void)var;  // currently unused; reserved for stricter checks (e.g. var-only).
  bool ok = true;
  PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (!ok) {
      return;
    }
    if (obj.as<BufferLoadNode>()) {
      ok = false;  // any indirect index → reject
      return;
    }
    // Only inspect PrimExpr nodes; ignore Stmt / Buffer / etc.
    if (!obj->IsInstance<PrimExprNode>()) {
      return;
    }
    // Whitelist: Add, Sub, Mul, IntImm, Var (any var; loop-invariants OK).
    // Everything else (Div, FloorDiv, FloorMod, Cast, Call, Select, …) is
    // rejected to keep the substitution trick sound.
    if (obj.as<AddNode>() || obj.as<SubNode>() || obj.as<MulNode>() ||
        obj.as<IntImmNode>() || obj.as<VarNode>()) {
      return;
    }
    ok = false;
  });
  return ok;
}

// CPPMEGA: Z3 idea #1 — vectorize_loop contiguity proof.
//
// When the heuristic ramp-extraction path can't conclude stride==1 (typically
// because `iter_var_size` is symbolic and the simplifier won't fold the access
// expression into a Ramp node), prove unit-stride directly:
//
//   forall var in [0, iter_var_size - 1):  expr(var + 1) - expr(var) == 1
//
// `expr` is an element-offset (already converted from indices via strides), so
// the goal is literally `delta == 1` (no element_bytes multiplier). Until the
// parallel `z3-bv-mode` branch lands `SetBitVectorMode(width)`, we bit-bound
// `var` by pushing range constraints via `EnterConstraint`. Any
// timeout / unknown / exception → conservative false. Heuristic-first: this
// only runs after all simplifier-based paths have failed.
//
// Audit fix (HIGH): we now require `expr` to be *syntactically affine* in
// `var` before invoking Z3 (see IsAffineInVar above). Indirect indexing
// `A[B[i]] = C[i]` is conservatively rejected.
//
// TIR-vs-Z3 semantic divergences kept in mind:
//   * FloorDiv/FloorMod (TIR) round toward -inf; Z3 BV bvsdiv/bvsmod round
//     toward 0. We constrain `var >= 0` so they agree on the bounded domain.
//   * Signed overflow on `var + 1`: the unsigned-32-bit bit-bound (see the
//     `bv_hi` literal below) keeps the successor in range; an
//     iter_var_size > 2^32 would be rejected here. We model unsigned 32-bit
//     pointer arithmetic semantics — the typical case for index registers
//     held in a 64-bit GPR.
//   * Loop-carried offsets `expr = base + var*stride + offset` are handled
//     correctly: `Substitute(var->var+1)` rewrites only the `var` term, and
//     `(base + (var+1)*stride + offset) - (base + var*stride + offset)`
//     simplifies to `stride`, so unit-stride iff `stride == 1`.
static bool Z3CanProveUnitStride(const PrimExpr &expr, const Var &var,
                                 const PrimExpr &iter_var_size,
                                 arith::Analyzer *analyzer) {
  // Audit fix (HIGH): affine guard. Indirect/non-affine accesses bypass Z3.
  if (!IsAffineInVar(expr, var)) {
    return false;
  }
  try {
    auto &z3 = arith::Z3Prover(analyzer);
    z3.SetTimeoutMs(50);
    DataType vt = var.dtype();
    PrimExpr lo = make_const(vt, 0);
    // Audit fix (MEDIUM): widen the bit-bound from signed 2^31 to unsigned
    // 2^32. Pointer arithmetic on the typical 64-bit host represents indices
    // as 32-bit *unsigned* values inside a 64-bit register, so the practical
    // upper bound on a contiguous index is 2^32, not 2^31.
    //
    // Implementation note: when `vt` is int32, we cannot literally write a
    // constant of value 2^32 (it does not fit in int32). In that case the
    // signed-int32 representation already caps `var` at 2^31 - 1, which is
    // strictly tighter than the unsigned-32 bound, so the constraint is
    // already satisfied implicitly and we simply omit the redundant `var <
    // bv_hi` clause. When `vt` is int64 (or wider), we use the full 2^32
    // bound. This preserves the documented assumption (unsigned 32-bit
    // emulation) without violating dtype constraints.
    bool vt_is_int32 = vt.is_int() && vt.bits() <= 32;
    // Loop-validity bound: var must allow `var+1` to remain a legal index,
    // i.e. var < iter_var_size - 1. This also leaves room for the negative
    // direction probe: var >= 1 is enforced inside that block. This rules
    // out the wrap-around case where iter_var_size could be 0 or 1 (no
    // contiguity to prove).
    PrimExpr iter_hi = analyzer->Simplify(iter_var_size - 1);
    PrimExpr range_constraint =
        (var >= lo) && (var < iter_hi) && (iter_var_size > 0);
    if (!vt_is_int32) {
      // Wider dtype: enforce explicit unsigned-32-bit emulation.
      PrimExpr bv_hi = make_const(vt, int64_t(1) << 32);
      range_constraint =
          range_constraint && (var < bv_hi) && (iter_var_size <= bv_hi);
    } else {
      // int32: signed range already caps at 2^31 - 1 < 2^32, but we still
      // bound iter_var_size by the int32 max.
      PrimExpr bv_hi = make_const(vt, int64_t(0x7fffffff));
      range_constraint = range_constraint && (iter_var_size <= bv_hi);
    }
    auto recover = z3.EnterConstraint(range_constraint);

    // Audit fix (HIGH): negative strides. The TileLang For loop is
    // normalised (min == 0, extent > 0, increment +1), so the loop
    // *variable* always advances positively. But the *access expression*
    // can still decrease in `var` — e.g. `out[N-1-i] = in[N-1-i]` produces
    // an address that decreases by 1 each iteration. That is just as
    // vectorizable (in absolute-value sense) as a positive stride: the
    // hardware load/store is `vload`/`vstore` either way, with reversed
    // element order.
    //
    // We probe both directions:
    //   delta_pos = expr(var+1) - expr(var)   (positive stride: == +1)
    //   delta_neg = expr(var) - expr(var-1)   (negative stride: == -1
    //                                          ⇔  expr(var-1) == expr(var)+1)
    // Either result of magnitude 1 means "addresses are contiguous"; we
    // accept both.
    PrimExpr var_plus_1 = var + make_const(vt, 1);
    PrimExpr expr_next = Substitute(expr, {{var, var_plus_1}});
    PrimExpr stride_goal_pos =
        (expr_next - expr) == make_const(expr.dtype(), 1);
    bool proved = z3.CanProve(stride_goal_pos);

    if (!proved) {
      // Negative-direction probe. Requires var >= 1 so var-1 is in-range;
      // we push that as a temporary constraint, prove, then pop.
      PrimExpr neg_constraint = (var >= make_const(vt, 1));
      auto recover_neg = z3.EnterConstraint(neg_constraint);
      PrimExpr var_minus_1 = var - make_const(vt, 1);
      PrimExpr expr_prev = Substitute(expr, {{var, var_minus_1}});
      // expr(var) - expr(var-1) == -1   ⇔  expr(var-1) - expr(var) == +1
      PrimExpr stride_goal_neg =
          (expr - expr_prev) == make_const(expr.dtype(), -1);
      proved = z3.CanProve(stride_goal_neg);
      recover_neg();
    }

    recover();
    if (!proved) {
      // Audit fix (LOW): log silent UNKNOWN/timeout/false paths so missed
      // vectorizations can be diagnosed later. We can't distinguish
      // "proved false" from "UNKNOWN" / "timeout" through the bool return
      // alone — DLOG(INFO) covers all three in one place. Behavior is
      // unchanged: the caller still sees `false` and falls back to scalar.
      DLOG(INFO) << "Z3CanProveUnitStride: could not prove unit stride for "
                 << "expr=" << expr << " var=" << var
                 << " iter_var_size=" << iter_var_size
                 << " (timeout/unknown/false — falling back to scalar)";
    }
    return proved;
  } catch (const std::exception &e) {
    // Conservative: any Z3 error / timeout / UNKNOWN / exception leaves
    // the For un-vectorized. Surface the exception in DLOG so a debug
    // build can pinpoint which queries blow up.
    DLOG(INFO) << "Z3CanProveUnitStride: exception (" << e.what()
               << ") — conservatively returning false. expr=" << expr
               << " var=" << var;
    return false;
  } catch (...) {
    DLOG(INFO) << "Z3CanProveUnitStride: unknown exception — conservatively "
                  "returning false. expr="
               << expr << " var=" << var;
    return false;
  }
}

bool IndicesCanVectorize(const PrimExpr &expr, Var var,
                         const PrimExpr &iter_var_size,
                         int target_vectorized_size,
                         arith::Analyzer *analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1)
    return true;

  // Extent must be divisible
  PrimExpr target_size_for_iter =
      make_const(iter_var_size.dtype(), target_vectorized_size);
  PrimExpr target_size_for_expr =
      make_const(expr.dtype(), target_vectorized_size);
  PrimExpr target_size_for_var =
      make_const(var.dtype(), target_vectorized_size);
  PrimExpr zero = make_const(var.dtype(), 0);

  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_size_for_iter),
                               0))
    return false;

  if (IsExprInvariantInVectorBoundary(expr, var, target_vectorized_size,
                                      analyzer)) {
    return true;
  }

  auto simplified_expr = analyzer->Simplify(Substitute(expr, {{var, zero}}));
  // The base offset must be divisible
  if (!analyzer->CanProveEqual(FloorMod(simplified_expr, target_size_for_expr),
                               zero)) {
    return false;
  }

  // Bind thread range
  Var v0("v0", var.dtype()), v1("v1", var.dtype());
  analyzer->Bind(v0, Range(zero, target_size_for_var));
  analyzer->Bind(v1, Range(zero, analyzer->Simplify(FloorDiv(
                                     iter_var_size, target_size_for_iter))));
  PrimExpr expr_transformed = analyzer->Simplify(
      Substitute(expr, {{var, v0 + v1 * target_size_for_var}}));
  Vectorizer vectorizer(v0, target_size_for_var);
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);

  // This simplify is necessary for thread region specified
  // optimizations.
  expr_vectorized = analyzer->Simplify(expr_vectorized);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    // Broadcast value
    if (expr_vectorized.dtype().lanes() == 1)
      return true;
    // CPPMEGA: Z3 idea #1 — broadcast/non-ramp shape was reached but lanes
    // count > 1 means the access actually does depend on `var`. Try the
    // direct Z3 unit-stride proof as a last resort before giving up.
    if (Z3CanProveUnitStride(expr, var, iter_var_size, analyzer)) {
      return true;
    }
    return false;
  } else {
    // CPPMEGA fix-B1 (idea712): only accept positive unit stride.
    //
    // Background: the parallel `z3-stack` branch adds a Z3-backed
    // negative-stride probe. The `VectorizeRewriter` codegen, however,
    // emits a `Ramp(min=0, stride=+1, lanes=N)` and assumes positive
    // direction. If a negative-stride probe is ever ported here without
    // the matching codegen change (a `negative_ramp` annotation +
    // `Ramp(stride=-1)` emission in `VectorizeRewriter`), kernels like
    // `for i in range(N-1, -1, -1): out[i] = in[i]` would be marked
    // vectorizable but lowered with the wrong-order ramp.
    //
    // Decision: REJECT negative stride at the planner (option (a) in the
    // cross-checked review). This is the safe-by-default position. If a
    // future change wires the codegen, replace this with the proper
    // negative_ramp flag plumbing (option (b)) and re-enable here.
    //
    // CPPMEGA Z3 idea #1 retained: still allow Z3 to prove stride==+1 for
    // expressions that don't simplify to a literal 1 (e.g. `(k%4)+1` where
    // k is provably a multiple of 4), and use the positive-only contiguity
    // fallback. This drops the prior `stride == -1` branch.
    if (is_one(ramp_node->stride)) {
      return true;
    }
    {
      auto &z3 = arith::Z3Prover(analyzer);
      z3.SetTimeoutMs(50);
      if (z3.CanProve(ramp_node->stride ==
                      make_const(ramp_node->stride.dtype(), 1))) {
        return true;
      }
    }
    return Z3CanProveUnitStride(expr, var, iter_var_size, analyzer);
  }
}

namespace {

/*!
 * \brief Convert TIR parallel loops into serial loops.
 *
 * TileLang uses ForKind::kParallel in a few places as a frontend "SIMT loop"
 * marker. When vectorize size resolves to 1 (i.e. no vectorization is applied),
 * keeping these loops as kParallel can block later loop transforms that only
 * apply to serial loops (e.g. pragma-unroll rewriting).
 *
 * This rewriter is intentionally conservative: it only downgrades kParallel to
 * kSerial and leaves all other loop kinds untouched.
 */
class ParallelToSerialRewriter : public StmtExprMutator {
private:
  Stmt VisitStmt_(const ForNode *node) final {
    Stmt visited = StmtExprMutator::VisitStmt_(node);
    For loop = Downcast<For>(visited);
    if (loop->kind == ForKind::kParallel) {
      loop.CopyOnWrite()->kind = ForKind::kSerial;
    }
    return loop;
  }
};

For ParallelToSerial(const For &loop) {
  ParallelToSerialRewriter rewriter;
  return Downcast<For>(rewriter(loop));
}

} // namespace

For VectorizeLoop(const For &loop, const LayoutMap &layout_map,
                  int vectorize_hint) {
  if (vectorize_hint <= 0) {
    arith::Analyzer analyzer;
    VectorizePlanner planner(&analyzer, layout_map);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return ParallelToSerial(loop);
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

For VectorizeLoop(const For &loop, arith::Analyzer *analyzer,
                  const LayoutMap &layout_map, int vectorize_hint) {
  if (vectorize_hint <= 0) {
    VectorizePlanner planner(analyzer, layout_map);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return ParallelToSerial(loop);
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

} // namespace tl
} // namespace tvm
