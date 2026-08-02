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
 * \file codegen_metal.cc
 */
#include "codegen_metal.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../op/builtin.h"
// CPPMEGA: apache renamed metal codegen entry header. The new
// `MetalModuleCreateWithFallback` factory lives in `target/metal/` and
// accepts `ffi::Map<String, Bytes>` smap + `ffi::Map<String, String>` source.
#include "runtime/thread_storage_scope.h"
#include "target/build_common.h"
#include "target/metal/metal_fallback_module.h"

namespace tvm {

namespace tl {
PrimFunc PointerValueTypeRewrite(
    PrimFunc f, bool allow_untyped_pointers = false, bool rewrite_params = true,
    bool rewrite_buffer_map = true, bool rewrite_alloc_buffer_node = true,
    bool rewrite_indices = true, bool rewrite_let_node = true,
    bool rewrite_scalar_read_to_vector_shuffle = true);
} // namespace tl

namespace codegen {

namespace {

class MetalBodyBufferAliasCollector final : public StmtExprVisitor {
public:
  explicit MetalBodyBufferAliasCollector(
      std::unordered_map<std::string, const VarNode *> param_aliases)
      : param_aliases_(std::move(param_aliases)) {}

  void VisitExpr_(const BufferLoadNode *op) final {
    MaybeCollect(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    MaybeCollect(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  struct Alias {
    const VarNode *param_var{nullptr};
    DataType dtype;
  };

  const std::unordered_map<const VarNode *, Alias> &aliases() const {
    return aliases_;
  }

private:
  void MaybeCollect(const Buffer &buffer) {
    const VarNode *data = buffer->data.get();
    if (aliases_.count(data)) {
      return;
    }
    auto it = param_aliases_.find(data->name_hint);
    if (it == param_aliases_.end() || it->second == data) {
      return;
    }
    if (!data->type_annotation.as<PointerTypeNode>()) {
      return;
    }
    std::string scope = GetPtrStorageScope(buffer->data);
    if (scope == "global") {
      aliases_[data] = Alias{it->second, buffer->dtype};
    }
  }

  std::unordered_map<std::string, const VarNode *> param_aliases_;
  std::unordered_map<const VarNode *, Alias> aliases_;
};

class MetalBodyBufferWriteCollector final : public StmtExprVisitor {
public:
  void VisitStmt_(const BufferStoreNode *op) final {
    MaybeCollect(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  const std::unordered_set<const VarNode *> &written_vars() const {
    return written_vars_;
  }

  const std::unordered_set<std::string> &written_names() const {
    return written_names_;
  }

private:
  void MaybeCollect(const Buffer &buffer) {
    const VarNode *data = buffer->data.get();
    written_vars_.insert(data);
    written_names_.insert(data->name_hint);
  }

  std::unordered_set<const VarNode *> written_vars_;
  std::unordered_set<std::string> written_names_;
};

} // namespace

void CodeGenTileLangMetal::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
  // Per-function CT state reset.  `emitted_mpp_include_` is intentionally
  // sticky: decl_stream is shared across kernels in the same module, so we
  // only want to emit the MPP include once per module.
  emitted_frag_lane_vars_ = false;
  emitted_pct_op_ = false;
  cooperative_tensor_dtype_.clear();
  ct_c_inlined_.clear();
  ct_c_inlined_base_.clear();
  ct_c_inlined_next_ = 0;
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenTileLangMetal::CodeGenTileLangMetal(Target target) : target_(target) {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream << "#include <metal_simdgroup>\n";
  decl_stream << "using namespace metal;\n\n";
  // CPPMEGA: hybrid tl_pr_c granularity + stack-c switch dispatch.
  // FP8/FP4 helper prelude is emitted lazily by `EmitFPHelperPrelude` after
  // `CollectReferencedLowPrecisionDtypes` has scanned the kernel body. This
  // avoids unconditionally injecting ~9KB of dead helper code into every
  // Metal kernel (e.g. pure-float32 vecadd).
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int[2];\n"
              << "};\n\n";
  // RFC §5.4 / lower_tma_to_ptr_arith.cc:249 — the non-NV TMA fallback
  // emits ``tl::call_extern("__tl_ptr_copy_elem", dst, src, bytes)`` for
  // each per-element memcpy in the rewritten pointer-arith For-nest. We
  // emit MSL inline overloads here so that the resulting .metal source
  // compiles as-is. Address-space-qualified overloads cover the four
  // combinations the TMA decomposition can produce:
  //   - device -> threadgroup  (tma_load    : global -> shared)
  //   - threadgroup -> device  (tma_store   : shared -> global)
  //   - device -> device       (global -> global, defensive)
  //   - threadgroup -> threadgroup (shared -> shared, defensive)
  // The body is a simple byte loop; MSL's compiler will vectorize when
  // the alignment is statically known.
  decl_stream
      << "static inline void __tl_ptr_copy_elem("
         "device void* dst, device const void* src, int bytes) {\n"
      << "  device char* d = (device char*)dst;\n"
      << "  device const char* s = (device const char*)src;\n"
      << "  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }\n"
      << "}\n"
      << "static inline void __tl_ptr_copy_elem("
         "threadgroup void* dst, device const void* src, int bytes) {\n"
      << "  threadgroup char* d = (threadgroup char*)dst;\n"
      << "  device const char* s = (device const char*)src;\n"
      << "  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }\n"
      << "}\n"
      << "static inline void __tl_ptr_copy_elem("
         "device void* dst, threadgroup const void* src, int bytes) {\n"
      << "  device char* d = (device char*)dst;\n"
      << "  threadgroup const char* s = (threadgroup const char*)src;\n"
      << "  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }\n"
      << "}\n"
      << "static inline void __tl_ptr_copy_elem("
         "threadgroup void* dst, threadgroup const void* src, int bytes) {\n"
      << "  threadgroup char* d = (threadgroup char*)dst;\n"
      << "  threadgroup const char* s = (threadgroup const char*)src;\n"
      << "  for (int i = 0; i < bytes; ++i) { d[i] = s[i]; }\n"
      << "}\n\n";
  decl_stream
      << "namespace tl {\n"
      << "struct SumOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "x + y; }\n"
      << "};\n"
      << "struct MulOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "x * y; }\n"
      << "};\n"
      << "struct MaxOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "y < x ? x : y; }\n"
      << "};\n"
      << "struct MinOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "y > x ? x : y; }\n"
      << "};\n"
      << "struct BitAndOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "x & y; }\n"
      << "};\n"
      << "struct BitOrOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "x | y; }\n"
      << "};\n"
      << "struct BitXorOp {\n"
      << "  template <typename T> inline T operator()(T x, T y) const { return "
         "x ^ y; }\n"
      << "};\n"
      << "template <typename T, int rows_per_threadgroup, int cols>\n"
      << "struct RowReduceSumContiguousInnermost {\n"
      << "  static_assert(rows_per_threadgroup > 0,\n"
      << "                \"rows_per_threadgroup must be positive\");\n"
      << "  static_assert(cols > 0, \"cols must be positive\");\n"
      << "  enum { simdgroup_size = 32 };\n"
      << "  static inline void run(device const T* A, device T* B, uint "
         "block_id,\n"
      << "                         uint tid, uint rows) {\n"
      << "    const uint row_in_group = tid / uint(simdgroup_size);\n"
      << "    const uint lane = tid & uint(simdgroup_size - 1);\n"
      << "    if (row_in_group >= uint(rows_per_threadgroup)) {\n"
      << "      return;\n"
      << "    }\n"
      << "    const uint row = block_id * uint(rows_per_threadgroup) + "
         "row_in_group;\n"
      << "    if (row >= rows) {\n"
      << "      return;\n"
      << "    }\n"
      << "    T acc = T(0);\n"
      << "    for (uint col = lane; col < uint(cols); col += "
         "uint(simdgroup_size)) {\n"
      << "      acc += A[row * uint(cols) + col];\n"
      << "    }\n"
      << "    T total = simd_sum(acc);\n"
      << "    if (lane == 0) {\n"
      << "      B[row] = total;\n"
      << "    }\n"
      << "  }\n"
      << "};\n"
      << "struct SyncThreadsBarrier {\n"
      << "  template <int phase = 0> static inline void sync() {\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "  }\n"
      << "};\n"
      << "template <int all_threads> struct NamedBarrier {\n"
      << "  template <int phase = 0> static inline void sync() {\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "  }\n"
      << "};\n"
      << "template <class Reducer, int threads, int scale, int thread_offset,\n"
      << "          class Barrier, int batch_size, int workspace_stride>\n"
      << "struct AllReduce;\n"
      << "template <class Reducer>\n"
      << "struct SimdgroupIntraReduce {\n"
      << "  template <typename T>\n"
      << "  static inline T run(T x) {\n"
      << "    x = Reducer()(x, simd_shuffle_xor(x, uint(16)));\n"
      << "    x = Reducer()(x, simd_shuffle_xor(x, uint(8)));\n"
      << "    x = Reducer()(x, simd_shuffle_xor(x, uint(4)));\n"
      << "    x = Reducer()(x, simd_shuffle_xor(x, uint(2)));\n"
      << "    x = Reducer()(x, simd_shuffle_xor(x, uint(1)));\n"
      << "    return x;\n"
      << "  }\n"
      << "};\n"
      << "template <>\n"
      << "struct SimdgroupIntraReduce<SumOp> {\n"
      << "  template <typename T>\n"
      << "  static inline T run(T x) {\n"
      << "    return simd_sum(x);\n"
      << "  }\n"
      << "};\n"
      << "template <class Reducer, int threads, int thread_offset,\n"
      << "          class Barrier, int batch_size, int workspace_stride>\n"
      << "struct AllReduceSimdgroupCross {\n"
      << "  enum { simdgroup_size = 32 };\n"
      << "  enum { simdgroup_count = threads / simdgroup_size };\n"
      << "  enum { final_slot = simdgroup_count };\n"
      << "  template <typename T>\n"
      << "  static inline T reduce_simdgroup(T x) {\n"
      << "    return SimdgroupIntraReduce<Reducer>::run(x);\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline T reduce_partials(T x, uint lane) {\n"
      << "    if (lane < uint(simdgroup_count)) {\n"
      << "      if (simdgroup_count >= 32) {\n"
      << "        x = Reducer()(x, simd_shuffle_xor(x, uint(16)));\n"
      << "      }\n"
      << "      if (simdgroup_count >= 16) {\n"
      << "        x = Reducer()(x, simd_shuffle_xor(x, uint(8)));\n"
      << "      }\n"
      << "      if (simdgroup_count >= 8) {\n"
      << "        x = Reducer()(x, simd_shuffle_xor(x, uint(4)));\n"
      << "      }\n"
      << "      if (simdgroup_count >= 4) {\n"
      << "        x = Reducer()(x, simd_shuffle_xor(x, uint(2)));\n"
      << "      }\n"
      << "      if (simdgroup_count >= 2) {\n"
      << "        x = Reducer()(x, simd_shuffle_xor(x, uint(1)));\n"
      << "      }\n"
      << "    }\n"
      << "    return x;\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline T run(T x, uint tid, threadgroup T* red_buf) {\n"
      << "    const int local_tid = int(tid) - thread_offset;\n"
      << "    const uint lane = uint(local_tid & (simdgroup_size - 1));\n"
      << "    const uint simdgroup_id = uint(local_tid >> 5);\n"
      << "    x = reduce_simdgroup(x);\n"
      << "    if (lane == 0) {\n"
      << "      red_buf[simdgroup_id] = x;\n"
      << "    }\n"
      << "    Barrier::template sync<1>();\n"
      << "    T result = red_buf[0];\n"
      << "    if (simdgroup_id == 0) {\n"
      << "      result = lane < uint(simdgroup_count) ? red_buf[lane] : "
         "red_buf[0];\n"
      << "      result = reduce_partials(result, lane);\n"
      << "      if (lane == 0) {\n"
      << "        red_buf[final_slot] = result;\n"
      << "      }\n"
      << "    }\n"
      << "    Barrier::template sync<2>();\n"
      << "    return red_buf[final_slot];\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(thread T* x, uint tid,\n"
      << "                               threadgroup T* red_buf) {\n"
      << "    const int local_tid = int(tid) - thread_offset;\n"
      << "    const uint lane = uint(local_tid & (simdgroup_size - 1));\n"
      << "    const uint simdgroup_id = uint(local_tid >> 5);\n"
      << "    for (int i = 0; i < batch_size; ++i) {\n"
      << "      T partial = reduce_simdgroup(x[i]);\n"
      << "      x[i] = partial;\n"
      << "    }\n"
      << "    for (int i = 0; i < batch_size; ++i) {\n"
      << "      if (lane == 0) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        red_buf[simdgroup_id + batch_offset] = x[i];\n"
      << "      }\n"
      << "    }\n"
      << "    Barrier::template sync<1>();\n"
      << "    if (simdgroup_id == 0) {\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        T result = lane < uint(simdgroup_count)\n"
      << "                       ? red_buf[lane + batch_offset]\n"
      << "                       : red_buf[batch_offset];\n"
      << "        result = reduce_partials(result, lane);\n"
      << "        if (lane == 0) {\n"
      << "          red_buf[final_slot + batch_offset] = result;\n"
      << "        }\n"
      << "      }\n"
      << "    }\n"
      << "    Barrier::template sync<2>();\n"
      << "    for (int i = 0; i < batch_size; ++i) {\n"
      << "      const int batch_offset = i * workspace_stride;\n"
      << "      x[i] = red_buf[final_slot + batch_offset];\n"
      << "    }\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(threadgroup T* x, uint tid,\n"
      << "                               threadgroup T* red_buf) {\n"
      << "    const int local_tid = int(tid) - thread_offset;\n"
      << "    const uint lane = uint(local_tid & (simdgroup_size - 1));\n"
      << "    const uint simdgroup_id = uint(local_tid >> 5);\n"
      << "    for (int i = 0; i < batch_size; ++i) {\n"
      << "      T partial = reduce_simdgroup(x[i]);\n"
      << "      x[i] = partial;\n"
      << "    }\n"
      << "    for (int i = 0; i < batch_size; ++i) {\n"
      << "      if (lane == 0) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        red_buf[simdgroup_id + batch_offset] = x[i];\n"
      << "      }\n"
      << "    }\n"
      << "    Barrier::template sync<1>();\n"
      << "    if (simdgroup_id == 0) {\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        T result = lane < uint(simdgroup_count)\n"
      << "                       ? red_buf[lane + batch_offset]\n"
      << "                       : red_buf[batch_offset];\n"
      << "        result = reduce_partials(result, lane);\n"
      << "        if (lane == 0) {\n"
      << "          red_buf[final_slot + batch_offset] = result;\n"
      << "        }\n"
      << "      }\n"
      << "    }\n"
      << "    Barrier::template sync<2>();\n"
      << "    for (int i = 0; i < batch_size; ++i) {\n"
      << "      const int batch_offset = i * workspace_stride;\n"
      << "      x[i] = red_buf[final_slot + batch_offset];\n"
      << "    }\n"
      << "  }\n"
      << "};\n"
      << "template <class Reducer, int threads, int scale, int thread_offset,\n"
      << "          class Barrier, int batch_size, int workspace_stride,\n"
      << "          bool done>\n"
      << "struct AllReduceStep;\n"
      << "template <class Reducer, int threads, int scale, int thread_offset,\n"
      << "          class Barrier, int batch_size, int workspace_stride>\n"
      << "struct AllReduceStep<Reducer, threads, scale, thread_offset, "
         "Barrier,\n"
      << "                     batch_size, workspace_stride, true> {\n"
      << "  template <typename T>\n"
      << "  static inline T run(T x, uint tid, threadgroup T* red_buf = "
         "nullptr) {\n"
      << "    return x;\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(thread T* x, uint tid,\n"
      << "                               threadgroup T* red_buf = nullptr) {}\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(threadgroup T* x, uint tid,\n"
      << "                               threadgroup T* red_buf = nullptr) {}\n"
      << "};\n"
      << "template <class Reducer, int threads, int scale, int thread_offset,\n"
      << "          class Barrier, int batch_size, int workspace_stride>\n"
      << "struct AllReduceStep<Reducer, threads, scale, thread_offset, "
         "Barrier,\n"
      << "                     batch_size, workspace_stride, false> {\n"
      << "  enum { offset = threads / 2 };\n"
      << "  template <typename T>\n"
      << "  static inline T run(T x, uint tid, threadgroup T* red_buf = "
         "nullptr) {\n"
      << "    const int local_tid = int(tid) - thread_offset;\n"
      << "    if (offset >= 32) {\n"
      << "      Barrier::template sync<1>();\n"
      << "      red_buf[local_tid] = x;\n"
      << "      Barrier::template sync<2>();\n"
      << "      x = Reducer()(x, red_buf[local_tid ^ offset]);\n"
      << "    } else {\n"
      << "      x = Reducer()(x, simd_shuffle_xor(x, uint(offset)));\n"
      << "    }\n"
      << "    if (offset == scale) {\n"
      << "      return x;\n"
      << "    }\n"
      << "    return AllReduce<Reducer, offset, scale, thread_offset, "
         "Barrier,\n"
      << "                     batch_size, workspace_stride>::run(x, tid, "
         "red_buf);\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(thread T* x, uint tid,\n"
      << "                               threadgroup T* red_buf = nullptr) {\n"
      << "    const int local_tid = int(tid) - thread_offset;\n"
      << "    if (offset >= 32) {\n"
      << "      Barrier::template sync<1>();\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        red_buf[local_tid + batch_offset] = x[i];\n"
      << "      }\n"
      << "      Barrier::template sync<2>();\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) + "
         "batch_offset]);\n"
      << "      }\n"
      << "    } else {\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        x[i] = Reducer()(x[i], simd_shuffle_xor(x[i], "
         "uint(offset)));\n"
      << "      }\n"
      << "    }\n"
      << "    if (offset != scale) {\n"
      << "      AllReduce<Reducer, offset, scale, thread_offset, Barrier,\n"
      << "                batch_size, workspace_stride>::run_batch(x, tid, "
         "red_buf);\n"
      << "    }\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(threadgroup T* x, uint tid,\n"
      << "                               threadgroup T* red_buf = nullptr) {\n"
      << "    const int local_tid = int(tid) - thread_offset;\n"
      << "    if (offset >= 32) {\n"
      << "      Barrier::template sync<1>();\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        red_buf[local_tid + batch_offset] = x[i];\n"
      << "      }\n"
      << "      Barrier::template sync<2>();\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        const int batch_offset = i * workspace_stride;\n"
      << "        x[i] = Reducer()(x[i], red_buf[(local_tid ^ offset) + "
         "batch_offset]);\n"
      << "      }\n"
      << "    } else {\n"
      << "      for (int i = 0; i < batch_size; ++i) {\n"
      << "        x[i] = Reducer()(x[i], simd_shuffle_xor(x[i], "
         "uint(offset)));\n"
      << "      }\n"
      << "    }\n"
      << "    if (offset != scale) {\n"
      << "      AllReduce<Reducer, offset, scale, thread_offset, Barrier,\n"
      << "                batch_size, workspace_stride>::run_batch(x, tid, "
         "red_buf);\n"
      << "    }\n"
      << "  }\n"
      << "};\n"
      << "template <class Reducer, int threads, int scale, int thread_offset = "
         "0,\n"
      << "          class Barrier = SyncThreadsBarrier, int batch_size = 1,\n"
      << "          int workspace_stride = 0>\n"
      << "struct AllReduce {\n"
      << "  static_assert(threads % scale == 0,\n"
      << "                \"tl::AllReduce<>: threads must be divisible by "
         "scale\");\n"
      << "  static_assert((threads & (threads - 1)) == 0,\n"
      << "                \"tl::AllReduce<>: threads must be a power of "
         "two\");\n"
      << "  template <typename T>\n"
      << "  static inline T run(T x, uint tid, threadgroup T* red_buf = "
         "nullptr) {\n"
      << "    if (threads > 32 && scale == 1 && (thread_offset % 32) == 0 &&\n"
      << "        workspace_stride >= threads) {\n"
      << "      return AllReduceSimdgroupCross<Reducer, threads, "
         "thread_offset,\n"
      << "          Barrier, batch_size, workspace_stride>::run(x, tid, "
         "red_buf);\n"
      << "    }\n"
      << "    return AllReduceStep<Reducer, threads, scale, thread_offset, "
         "Barrier,\n"
      << "                         batch_size, workspace_stride,\n"
      << "                         (threads == scale)>::run(x, tid, red_buf);\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(thread T* x, uint tid,\n"
      << "                               threadgroup T* red_buf = nullptr) {\n"
      << "    if (threads > 32 && scale == 1 && (thread_offset % 32) == 0 &&\n"
      << "        workspace_stride >= threads) {\n"
      << "      AllReduceSimdgroupCross<Reducer, threads, thread_offset,\n"
      << "          Barrier, batch_size, workspace_stride>::run_batch(x, tid, "
         "red_buf);\n"
      << "      return;\n"
      << "    }\n"
      << "    AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,\n"
      << "                  batch_size, workspace_stride,\n"
      << "                  (threads == scale)>::run_batch(x, tid, red_buf);\n"
      << "  }\n"
      << "  template <typename T>\n"
      << "  static inline void run_batch(threadgroup T* x, uint tid,\n"
      << "                               threadgroup T* red_buf = nullptr) {\n"
      << "    if (threads > 32 && scale == 1 && (thread_offset % 32) == 0 &&\n"
      << "        workspace_stride >= threads) {\n"
      << "      AllReduceSimdgroupCross<Reducer, threads, thread_offset,\n"
      << "          Barrier, batch_size, workspace_stride>::run_batch(x, tid, "
         "red_buf);\n"
      << "      return;\n"
      << "    }\n"
      << "    AllReduceStep<Reducer, threads, scale, thread_offset, Barrier,\n"
      << "                  batch_size, workspace_stride,\n"
      << "                  (threads == scale)>::run_batch(x, tid, red_buf);\n"
      << "  }\n"
      << "};\n"
      << "} /* namespace tl */\n\n";
}

// CPPMEGA: hybrid tl_pr_c granularity + stack-c switch dispatch.
// Per-dtype FP8 helper emitters. Each writes its decl into decl_stream so
// that Finish() will prepend it ahead of the kernel body. The bodies are
// raw text — no logic, only structure — so they are easy to extend.
void CodeGenTileLangMetal::EmitFp8E3M4Helper() {
  decl_stream << "static inline half __tvm_fp8_e3m4_to_half(uchar x) {\n"
              << "  uint raw = uint(x);\n"
              << "  uint abs = raw & 0x7fu;\n"
              << "  if (abs == 0u) return half(0.0f);\n"
              << "  uint exp = (raw >> 4) & 0x07u;\n"
              << "  uint mant = raw & 0x0fu;\n"
              << "  if (exp == 0x07u && mant == 0x0fu) return "
                 "half(as_type<float>(0x7fc00000u));\n"
              << "  float mag = (exp == 0u)\n"
              << "      ? (float(mant) * 0.0625f * exp2(-2.0f))\n"
              << "      : ((1.0f + float(mant) * 0.0625f) * "
                 "exp2(float(int(exp) - 3)));\n"
              << "  return half((raw & 0x80u) != 0u ? -mag : mag);\n"
              << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E4M3Helper() {
  decl_stream << "static inline half __tvm_fp8_e4m3_to_half(uchar x) {\n"
              << "  uint raw = uint(x);\n"
              << "  uint abs = raw & 0x7fu;\n"
              << "  if (abs == 0u) return half(0.0f);\n"
              << "  uint exp = (raw >> 3) & 0x0fu;\n"
              << "  uint mant = raw & 0x07u;\n"
              << "  if (exp == 0x0fu && mant == 0x07u) return "
                 "half(as_type<float>(0x7fc00000u));\n"
              << "  float mag = (exp == 0u)\n"
              << "      ? (float(mant) * 0.125f * exp2(-6.0f))\n"
              << "      : ((1.0f + float(mant) * 0.125f) * exp2(float(int(exp) "
                 "- 7)));\n"
              << "  return half((raw & 0x80u) != 0u ? -mag : mag);\n"
              << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E4M3FnAliasHelper() {
  // Depends on EmitFp8E4M3Helper having been emitted already.
  decl_stream << "static inline half __tvm_fp8_e4m3fn_to_half(uchar x) {\n"
              << "  return __tvm_fp8_e4m3_to_half(x);\n"
              << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E4M3FnuzHelper() {
  decl_stream << "static inline half __tvm_fp8_e4m3fnuz_to_half(uchar x) {\n"
              << "  uint raw = uint(x) & 0x7fu;\n"
              << "  uint exp = (raw >> 3) & 0x0fu;\n"
              << "  uint mant = raw & 0x07u;\n"
              << "  if (raw == 0u) return half(0.0f);\n"
              << "  if (exp == 0x0fu && mant == 0x07u) return "
                 "half(as_type<float>(0x7fc00000u));\n"
              << "  float mag = (exp == 0u)\n"
              << "      ? (float(mant) * 0.125f * exp2(-6.0f))\n"
              << "      : ((1.0f + float(mant) * 0.125f) * exp2(float(int(exp) "
                 "- 7)));\n"
              << "  return half(mag);\n"
              << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E4M3B11FnuzHelper() {
  decl_stream
      << "static inline half __tvm_fp8_e4m3b11fnuz_to_half(uchar x) {\n"
      << "  uint original = uint(x);\n"
      << "  uint raw = original & 0x7fu;\n"
      << "  uint exp = (raw >> 3) & 0x0fu;\n"
      << "  uint mant = raw & 0x07u;\n"
      << "  if (original == 0x80u) return half(as_type<float>(0x7fc00000u));\n"
      << "  if (raw == 0u) return half(0.0f);\n"
      << "  float mag = (exp == 0u)\n"
      << "      ? (float(mant) * 0.125f * exp2(-10.0f))\n"
      << "      : ((1.0f + float(mant) * 0.125f) * exp2(float(int(exp) - "
         "11)));\n"
      << "  return half(mag);\n"
      << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E5M2Helper() {
  decl_stream << "static inline half __tvm_fp8_e5m2_to_half(uchar x) {\n"
              << "  uint raw = uint(x);\n"
              << "  uint abs = raw & 0x7fu;\n"
              << "  if (abs == 0u) return half(0.0f);\n"
              << "  uint exp = (raw >> 2) & 0x1fu;\n"
              << "  uint mant = raw & 0x03u;\n"
              << "  if (exp == 0x1fu) {\n"
              << "    float inf = as_type<float>((raw & 0x80u) != 0u ? "
                 "0xff800000u : 0x7f800000u);\n"
              << "    float nan = as_type<float>(0x7fc00000u);\n"
              << "    return half(mant == 0u ? inf : nan);\n"
              << "  }\n"
              << "  float mag = (exp == 0u)\n"
              << "      ? (float(mant) * 0.25f * exp2(-14.0f))\n"
              << "      : ((1.0f + float(mant) * 0.25f) * exp2(float(int(exp) "
                 "- 15)));\n"
              << "  return half((raw & 0x80u) != 0u ? -mag : mag);\n"
              << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E5M2FnuzHelper() {
  decl_stream
      << "static inline half __tvm_fp8_e5m2fnuz_to_half(uchar x) {\n"
      << "  uint raw = uint(x) & 0x7fu;\n"
      << "  uint exp = (raw >> 2) & 0x1fu;\n"
      << "  uint mant = raw & 0x03u;\n"
      << "  if (raw == 0u) return half(0.0f);\n"
      << "  if (exp == 0x1fu) return half(as_type<float>(0x7fc00000u));\n"
      << "  float mag = (exp == 0u)\n"
      << "      ? (float(mant) * 0.25f * exp2(-14.0f))\n"
      << "      : ((1.0f + float(mant) * 0.25f) * exp2(float(int(exp) - "
         "15)));\n"
      << "  return half(mag);\n"
      << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8E8M0FnuHelper() {
  decl_stream << "static inline float __tvm_fp8_e8m0fnu_to_float(uchar x) {\n"
              << "  uint raw = uint(x);\n"
              << "  if (raw == 0xffu) return as_type<float>(0x7fc00000u);\n"
              << "  return exp2(float(int(raw) - 127));\n"
              << "}\n\n";
}

void CodeGenTileLangMetal::EmitFp8Dot4Helpers() {
  decl_stream
      << "constant float __tvm_fp8_e4m3fn_lut[256] = {\n"
      << "  0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, "
         "0.009765625f, 0.01171875f, 0.013671875f,\n"
      << "  0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, "
         "0.025390625f, 0.02734375f, 0.029296875f,\n"
      << "  0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, "
         "0.05078125f, 0.0546875f, 0.05859375f,\n"
      << "  0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, "
         "0.109375f, 0.1171875f,\n"
      << "  0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, "
         "0.21875f, 0.234375f,\n"
      << "  0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, "
         "0.46875f,\n"
      << "  0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,\n"
      << "  1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,\n"
      << "  2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,\n"
      << "  4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,\n"
      << "  8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,\n"
      << "  16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,\n"
      << "  32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,\n"
      << "  64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,\n"
      << "  128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,\n"
      << "  256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,\n"
      << "  0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, "
         "-0.009765625f, -0.01171875f, -0.013671875f,\n"
      << "  -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, "
         "-0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,\n"
      << "  -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, "
         "-0.05078125f, -0.0546875f, -0.05859375f,\n"
      << "  -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, "
         "-0.1015625f, -0.109375f, -0.1171875f,\n"
      << "  -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, "
         "-0.21875f, -0.234375f,\n"
      << "  -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, "
         "-0.4375f, -0.46875f,\n"
      << "  -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, "
         "-0.9375f,\n"
      << "  -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,\n"
      << "  -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,\n"
      << "  -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,\n"
      << "  -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,\n"
      << "  -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,\n"
      << "  -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,\n"
      << "  -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, "
         "-120.0f,\n"
      << "  -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, "
         "-240.0f,\n"
      << "  -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, "
         "NAN,\n"
      << "};\n\n";
  decl_stream
      << "static inline float __tvm_fp8_e4m3_dot4_words(uint pa, uint pb) {\n"
      << "  return __tvm_fp8_e4m3fn_lut[pa & 0xFFu] * __tvm_fp8_e4m3fn_lut[pb "
         "& 0xFFu]\n"
      << "       + __tvm_fp8_e4m3fn_lut[(pa >> 8) & 0xFFu] * "
         "__tvm_fp8_e4m3fn_lut[(pb >> 8) & 0xFFu]\n"
      << "       + __tvm_fp8_e4m3fn_lut[(pa >> 16) & 0xFFu] * "
         "__tvm_fp8_e4m3fn_lut[(pb >> 16) & 0xFFu]\n"
      << "       + __tvm_fp8_e4m3fn_lut[(pa >> 24) & 0xFFu] * "
         "__tvm_fp8_e4m3fn_lut[(pb >> 24) & 0xFFu];\n"
      << "}\n\n";
  decl_stream
      << "static inline uint __tvm_fp8_load_u32(device const uchar* p, uint "
         "word_idx) {\n"
      << "  return reinterpret_cast<device const uint*>(p)[word_idx];\n"
      << "}\n"
      << "static inline uint __tvm_fp8_load_u32(threadgroup const uchar* p, "
         "uint word_idx) {\n"
      << "  return reinterpret_cast<threadgroup const uint*>(p)[word_idx];\n"
      << "}\n"
      << "static inline uint __tvm_fp8_load_u32(constant const uchar* p, uint "
         "word_idx) {\n"
      << "  return reinterpret_cast<constant const uint*>(p)[word_idx];\n"
      << "}\n"
      << "static inline uint __tvm_fp8_load_u32(device const uint* p, uint "
         "word_idx) {\n"
      << "  return p[word_idx];\n"
      << "}\n"
      << "static inline uint __tvm_fp8_load_u32(threadgroup const uint* p, "
         "uint word_idx) {\n"
      << "  return p[word_idx];\n"
      << "}\n"
      << "static inline uint __tvm_fp8_load_u32(constant const uint* p, uint "
         "word_idx) {\n"
      << "  return p[word_idx];\n"
      << "}\n\n";
  decl_stream
      << "static inline float __tvm_fp8_e4m3_dot4_packed(device const uchar* "
         "a, device const uchar* b, uint a_word_idx, uint b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(device const uchar* "
         "a, threadgroup const uchar* b, uint a_word_idx, uint b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(device const uchar* "
         "a, constant const uchar* b, uint a_word_idx, uint b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(threadgroup const "
         "uchar* a, device const uchar* b, uint a_word_idx, uint b_word_idx) "
         "{\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(threadgroup const "
         "uchar* a, threadgroup const uchar* b, uint a_word_idx, uint "
         "b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(threadgroup const "
         "uchar* a, constant const uchar* b, uint a_word_idx, uint b_word_idx) "
         "{\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(constant const uchar* "
         "a, device const uchar* b, uint a_word_idx, uint b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(constant const uchar* "
         "a, threadgroup const uchar* b, uint a_word_idx, uint b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n"
      << "static inline float __tvm_fp8_e4m3_dot4_packed(constant const uchar* "
         "a, constant const uchar* b, uint a_word_idx, uint b_word_idx) {\n"
      << "  return __tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32(a, "
         "a_word_idx), __tvm_fp8_load_u32(b, b_word_idx));\n"
      << "}\n\n";
}

void CodeGenTileLangMetal::EmitBFloat16Helper() {
  if (emitted_bfloat16_helper_) {
    return;
  }
  emitted_bfloat16_helper_ = true;
  decl_stream << "struct tvm_bfloat16 {\n"
              << "  ushort bits;\n"
              << "  tvm_bfloat16() = default;\n"
              << "  tvm_bfloat16(float value) {\n"
              << "    uint raw = as_type<uint>(value);\n"
              << "    uint lsb = (raw >> 16) & 1u;\n"
              << "    bits = ushort((raw + 0x7fffu + lsb) >> 16);\n"
              << "  }\n"
              << "  operator float() const {\n"
              << "    return as_type<float>(uint(bits) << 16);\n"
              << "  }\n"
              << "};\n"
              << "static inline float __tvm_bfloat16_to_float(thread const "
                 "tvm_bfloat16& value) {\n"
              << "  return as_type<float>(uint(value.bits) << 16);\n"
              << "}\n"
              << "static inline float __tvm_bfloat16_to_float(device const "
                 "tvm_bfloat16& value) {\n"
              << "  return as_type<float>(uint(value.bits) << 16);\n"
              << "}\n"
              << "static inline float __tvm_bfloat16_to_float(threadgroup "
                 "const tvm_bfloat16& value) {\n"
              << "  return as_type<float>(uint(value.bits) << 16);\n"
              << "}\n"
              << "static inline float __tvm_bfloat16_to_float(constant const "
                 "tvm_bfloat16& value) {\n"
              << "  return as_type<float>(uint(value.bits) << 16);\n"
              << "}\n\n";
}

// CPPMEGA: hybrid tl_pr_c granularity + stack-c switch dispatch.
// Pre-walker that scans a PrimFunc for any FP8 (and forward-compat FP4) dtype
// use. The walker visits BufferLoad/BufferStore/Cast/Call/Allocate/Broadcast/
// Var/Let/Ramp expressions and stores the unique TypeCodes in
// `referenced_fp8_codes_`, which `EmitFPHelperPrelude` consumes via switch
// dispatch into the per-dtype emitters above.
namespace {

class MetalFp8DTypeCollector final : public StmtExprVisitor {
public:
  std::set<int> referenced_codes;
  bool uses_dot4{false};
  bool uses_atomic_add{false};
  bool uses_atomic_cas{false};
  // CPPMEGA / Path C: track usage of the two MSL kernel-attribute intrinsics
  // emitted by tilelang/language/fp8_op.py. These do not have a matching
  // ``[[thread_position_in_grid]]`` / ``[[thread_index_in_simdgroup]]``
  // declaration in the default Metal kernel signature, so we have to inject
  // them on demand when the body actually references the intrinsic.
  bool uses_grid_tid_x{false};
  bool uses_simd_lane_id{false};

  // Names accepted as the FP8 packed dot4 intrinsic. We accept BOTH the
  // legacy ``tir.metal.*`` namespace (still emitted by the tilelang macro at
  // tilelang/language/fp8_op.py) AND the post-rename ``tirx.metal.*``
  // namespace (which is what cppmega_mlx/nn/_tilelang/_msl_transform.py
  // registers as a TVM Op so ``Op.get(...)`` succeeds via the python compat
  // shim in 3rdparty/tvm/python/tvm/tir/__init__.py). The shim translates
  // ``Op.get("tir.metal.X")`` -> ``Op.get("tirx.metal.X")`` on lookup
  // failure; the resolved Op carries the registered (``tirx.*``) name into
  // the C++ CallNode, but older PrimFuncs cached pre-shim may still arrive
  // carrying the original ``tir.*`` literal. Recognising both keeps codegen
  // working regardless of which side resolved the lookup. DO NOT delete the
  // ``tir.metal.*`` branches even if a future canonicalisation makes them
  // unreachable in practice — per repo policy "Never silently delete dead
  // code" they document the namespace ambiguity for the next bisection.
  static bool IsFp8Dot4Intrin(const std::string &name) {
    return name == "tir.metal.fp8_e4m3_dot4" ||
           name == "tirx.metal.fp8_e4m3_dot4";
  }
  static bool IsFp8LoadU32Intrin(const std::string &name) {
    return name == "tir.metal.fp8_load_u32" ||
           name == "tirx.metal.fp8_load_u32";
  }
  static bool IsFp8Dot4WordsIntrin(const std::string &name) {
    return name == "tir.metal.fp8_e4m3_dot4_words" ||
           name == "tirx.metal.fp8_e4m3_dot4_words";
  }
  static bool UsesFp8Dot4HelperIntrin(const std::string &name) {
    return IsFp8Dot4Intrin(name) || IsFp8LoadU32Intrin(name) ||
           IsFp8Dot4WordsIntrin(name);
  }
  static bool IsGridTidXIntrin(const std::string &name) {
    return name == "tir.metal.thread_position_in_grid_x" ||
           name == "tirx.metal.thread_position_in_grid_x";
  }
  static bool IsThreadgroupTidXIntrin(const std::string &name) {
    return name == "tir.metal.thread_position_in_threadgroup_x" ||
           name == "tirx.metal.thread_position_in_threadgroup_x";
  }
  static bool IsSimdLaneIdIntrin(const std::string &name) {
    return name == "tir.metal.thread_index_in_simdgroup" ||
           name == "tirx.metal.thread_index_in_simdgroup";
  }
  static bool IsSimdSumIntrin(const std::string &name) {
    return name == "tir.metal.simd_sum" || name == "tirx.metal.simd_sum";
  }
  static bool IsGridTidXVarName(const std::string &name) {
    return name == "grid_tid" || name.rfind("grid_tid_", 0) == 0;
  }
  static bool IsSimdLaneIdVarName(const std::string &name) {
    return name == "simd_lane" || name.rfind("simd_lane_", 0) == 0;
  }

  void Note(const DataType &t) {
    if (t.is_float8() || t.is_float4()) {
      referenced_codes.insert(static_cast<int>(t.code()));
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Note(op->dtype);
    if (op->buffer.defined())
      Note(op->buffer->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    Note(op->value.dtype());
    if (op->buffer.defined())
      Note(op->buffer->dtype);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CastNode *op) final {
    Note(op->dtype);
    Note(op->value.dtype());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    Note(op->dtype);
    for (const auto &arg : op->args) {
      Note(arg.dtype());
    }
    if (op->op.same_as(tl::atomic_add_elem_op()) ||
        op->op.same_as(tl::atomic_add_ret_elem_op())) {
      uses_atomic_add = true;
    }
    if (op->op.same_as(builtin::atomic_cas())) {
      uses_atomic_cas = true;
    }
    if (auto *opn = op->op.as<OpNode>()) {
      // CPPMEGA: the python ``T.call_intrin("tir.metal.fp8_e4m3_dot4", ...)``
      // wrapper rewrites legacy ``tir.*`` names to ``tirx.*`` (see
      // 3rdparty/tvm/python/tvm/tirx/expr.py:Call.__init__), so the registered
      // op name is ``tirx.metal.fp8_e4m3_dot4``. ``IsFp8Dot4Intrin`` matches
      // both spellings to avoid a silent miss on the helper-prelude emission.
      if (UsesFp8Dot4HelperIntrin(opn->name)) {
        uses_dot4 = true;
      } else if (IsGridTidXIntrin(opn->name)) {
        uses_grid_tid_x = true;
      } else if (IsSimdLaneIdIntrin(opn->name)) {
        uses_simd_lane_id = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  // CPPMEGA: in the swap branch `AllocateNode` resolves to the vendored
  // `tilelang::tl_tir::AllocateNode`, which is not part of StmtExprVisitor's
  // virtual hierarchy — overloading it here would hide base virtuals. Buffer
  // allocations carry their dtype through the buffer_map walk in
  // CollectReferencedLowPrecisionDtypes; nothing is missed by skipping it.
  void VisitExpr_(const BroadcastNode *op) final {
    Note(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const RampNode *op) final {
    Note(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const VarNode *op) final {
    Note(op->dtype);
    if (IsGridTidXVarName(op->name_hint)) {
      uses_grid_tid_x = true;
    } else if (IsSimdLaneIdVarName(op->name_hint)) {
      uses_simd_lane_id = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const LetNode *op) final {
    Note(op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }
};

// CPPMEGA / Path C: side-table mapping each CodeGenTileLangMetal instance to
// the freshly-supplied MSL identifiers it chose for the
// ``[[thread_position_in_grid]]`` and ``[[thread_index_in_simdgroup]]``
// kernel arguments (when used). This lives in a free helper rather than as a
// member because the Path-C task scope is restricted to codegen_metal.cc and
// the header may not be modified for this fix; see the explanatory comment in
// AddFunction below. Keyed by ``this`` because emission is single-threaded
// per codegen and entries are overwritten when the same instance starts a
// new function. Empty strings mean "not used in this kernel".
struct MetalScalarIntrinIds {
  std::string grid_tid_x;
  std::string simd_lane_id;
};
inline std::unordered_map<const void *, MetalScalarIntrinIds> &
GetMetalScalarIntrinIdMap() {
  static std::unordered_map<const void *, MetalScalarIntrinIds> kMap;
  return kMap;
}

} // namespace

void CodeGenTileLangMetal::CollectReferencedLowPrecisionDtypes(
    const PrimFunc &f) {
  referenced_fp8_codes_.clear();
  uses_fp8_dot4_ = false;
  uses_atomic_add_ = false;
  uses_atomic_cas_ = false;
  MetalFp8DTypeCollector collector;
  // Inspect parameter dtypes (handle pointers carry their pointee dtype via
  // the buffer_map; non-handle parameters carry it directly).
  for (const Var &v : f->params) {
    if (v.dtype().is_float8() || v.dtype().is_float4()) {
      referenced_fp8_codes_.insert(static_cast<int>(v.dtype().code()));
    }
    if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        if (prim->dtype.is_float8() || prim->dtype.is_float4()) {
          referenced_fp8_codes_.insert(static_cast<int>(prim->dtype.code()));
        }
      }
    }
  }
  for (const auto &kv : f->buffer_map) {
    DataType bt = kv.second->dtype;
    if (bt.is_float8() || bt.is_float4()) {
      referenced_fp8_codes_.insert(static_cast<int>(bt.code()));
    }
  }
  collector(f->body);
  referenced_fp8_codes_.insert(collector.referenced_codes.begin(),
                               collector.referenced_codes.end());
  if (collector.uses_dot4)
    uses_fp8_dot4_ = true;
  if (collector.uses_atomic_add)
    uses_atomic_add_ = true;
  if (collector.uses_atomic_cas)
    uses_atomic_cas_ = true;
}

void CodeGenTileLangMetal::EmitAtomicAddHelperPrelude() {
  if (!uses_atomic_add_ || emitted_atomic_add_helper_)
    return;
  emitted_atomic_add_helper_ = true;
  // The bf16 AtomicAdd overloads below reference the `tvm_bfloat16` storage
  // struct. PrintType emits it lazily, but the kernel body that triggers that
  // is generated AFTER this prelude — so force-emit the struct now (idempotent)
  // to guarantee it precedes the overload that uses it.
  EmitBFloat16Helper();
  decl_stream
      << "namespace tl {\n"
      << "static inline float AtomicAdd(device float* address, float val,\n"
      << "                              int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  device atomic_uint* bits = reinterpret_cast<device "
         "atomic_uint*>(address);\n"
      << "  uint old_bits = atomic_load_explicit(bits, memory_order_relaxed);\n"
      << "  while (true) {\n"
      << "    float old_val = as_type<float>(old_bits);\n"
      << "    uint new_bits = as_type<uint>(old_val + val);\n"
      << "    uint expected = old_bits;\n"
      << "    if (atomic_compare_exchange_weak_explicit(\n"
      << "            bits, &expected, new_bits, memory_order_relaxed,\n"
      << "            memory_order_relaxed)) {\n"
      << "      return old_val;\n"
      << "    }\n"
      << "    old_bits = expected;\n"
      << "  }\n"
      << "}\n"
      << "static inline int AtomicAdd(device int* address, int val,\n"
      << "                            int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  return atomic_fetch_add_explicit(\n"
      << "      reinterpret_cast<device atomic_int*>(address), val,\n"
      << "      memory_order_relaxed);\n"
      << "}\n"
      << "static inline uint AtomicAdd(device uint* address, uint val,\n"
      << "                             int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  return atomic_fetch_add_explicit(\n"
      << "      reinterpret_cast<device atomic_uint*>(address), val,\n"
      << "      memory_order_relaxed);\n"
      << "}\n"
      // Threadgroup (shared-memory) overloads: recurrent backward (e.g. Mamba3)
      // accumulates gradients into dynamic threadgroup scratch (buf_dyn_shmem),
      // which is `threadgroup float*` — the device-only overloads above do not
      // match. These mirror the device variants in threadgroup address space.
      << "static inline float AtomicAdd(threadgroup float* address, float "
         "val,\n"
      << "                              int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  threadgroup atomic_uint* bits =\n"
      << "      reinterpret_cast<threadgroup atomic_uint*>(address);\n"
      << "  uint old_bits = atomic_load_explicit(bits, memory_order_relaxed);\n"
      << "  while (true) {\n"
      << "    float old_val = as_type<float>(old_bits);\n"
      << "    uint new_bits = as_type<uint>(old_val + val);\n"
      << "    uint expected = old_bits;\n"
      << "    if (atomic_compare_exchange_weak_explicit(\n"
      << "            bits, &expected, new_bits, memory_order_relaxed,\n"
      << "            memory_order_relaxed)) {\n"
      << "      return old_val;\n"
      << "    }\n"
      << "    old_bits = expected;\n"
      << "  }\n"
      << "}\n"
      << "static inline int AtomicAdd(threadgroup int* address, int val,\n"
      << "                            int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  return atomic_fetch_add_explicit(\n"
      << "      reinterpret_cast<threadgroup atomic_int*>(address), val,\n"
      << "      memory_order_relaxed);\n"
      << "}\n"
      << "static inline uint AtomicAdd(threadgroup uint* address, uint val,\n"
      << "                             int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  return atomic_fetch_add_explicit(\n"
      << "      reinterpret_cast<threadgroup atomic_uint*>(address), val,\n"
      << "      memory_order_relaxed);\n"
      << "}\n"
      // bfloat16 overloads. Apple GPUs have no native bf16 atomic, so we run a
      // 32-bit-word CAS loop on the containing aligned word: the bf16 element
      // is 16-bit (tvm_bfloat16::bits is a ushort), so we align the byte
      // address down to a 4-byte boundary, pick which 16-bit half holds our
      // element, decode old bf16 -> float, add val (float), re-encode to bf16
      // with the SAME round-to-nearest-even used by tvm_bfloat16(float), splice
      // it back into the correct half, and CAS the whole word. Returns the
      // pre-add float value. This is numerically correct for scatter/gradient
      // accumulation (each addend lands at bf16 precision, like a serial bf16
      // sum). Mirrors the float CAS loop above for both address spaces.
      << "static inline ushort __tl_bf16_round(float value) {\n"
      << "  uint raw = as_type<uint>(value);\n"
      << "  uint lsb = (raw >> 16) & 1u;\n"
      << "  return ushort((raw + 0x7fffu + lsb) >> 16);\n"
      << "}\n"
      << "static inline float AtomicAdd(device tvm_bfloat16* address, float "
         "val,\n"
      << "                              int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  uintptr_t addr = reinterpret_cast<uintptr_t>(address);\n"
      << "  uint shift = uint(addr & 2u) * 8u;  // 0 for low half, 16 for "
         "high\n"
      << "  device atomic_uint* word = reinterpret_cast<device atomic_uint*>(\n"
      << "      addr & ~uintptr_t(3u));\n"
      << "  uint old_word = atomic_load_explicit(word, memory_order_relaxed);\n"
      << "  while (true) {\n"
      << "    ushort old_bits = ushort((old_word >> shift) & 0xffffu);\n"
      << "    float old_val = as_type<float>(uint(old_bits) << 16);\n"
      << "    ushort new_bits = __tl_bf16_round(old_val + val);\n"
      << "    uint new_word = (old_word & ~(0xffffu << shift)) |\n"
      << "                    (uint(new_bits) << shift);\n"
      << "    uint expected = old_word;\n"
      << "    if (atomic_compare_exchange_weak_explicit(\n"
      << "            word, &expected, new_word, memory_order_relaxed,\n"
      << "            memory_order_relaxed)) {\n"
      << "      return old_val;\n"
      << "    }\n"
      << "    old_word = expected;\n"
      << "  }\n"
      << "}\n"
      << "static inline float AtomicAdd(threadgroup tvm_bfloat16* address,\n"
      << "                              float val, int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  uintptr_t addr = reinterpret_cast<uintptr_t>(address);\n"
      << "  uint shift = uint(addr & 2u) * 8u;\n"
      << "  threadgroup atomic_uint* word =\n"
      << "      reinterpret_cast<threadgroup atomic_uint*>(addr & "
         "~uintptr_t(3u));\n"
      << "  uint old_word = atomic_load_explicit(word, memory_order_relaxed);\n"
      << "  while (true) {\n"
      << "    ushort old_bits = ushort((old_word >> shift) & 0xffffu);\n"
      << "    float old_val = as_type<float>(uint(old_bits) << 16);\n"
      << "    ushort new_bits = __tl_bf16_round(old_val + val);\n"
      << "    uint new_word = (old_word & ~(0xffffu << shift)) |\n"
      << "                    (uint(new_bits) << shift);\n"
      << "    uint expected = old_word;\n"
      << "    if (atomic_compare_exchange_weak_explicit(\n"
      << "            word, &expected, new_word, memory_order_relaxed,\n"
      << "            memory_order_relaxed)) {\n"
      << "      return old_val;\n"
      << "    }\n"
      << "    old_word = expected;\n"
      << "  }\n"
      << "}\n"
      << "} /* namespace tl */\n\n";
}

void CodeGenTileLangMetal::EmitAtomicCASHelperPrelude() {
  if (!uses_atomic_cas_ || emitted_atomic_cas_helper_)
    return;
  emitted_atomic_cas_helper_ = true;
  decl_stream
      << "namespace tl {\n"
      << "static inline int AtomicCAS(device int* address, int expected,\n"
      << "                            int desired, int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  int observed = expected;\n"
      << "  while (!atomic_compare_exchange_weak_explicit(\n"
      << "             reinterpret_cast<device atomic_int*>(address), "
         "&observed,\n"
      << "             desired, memory_order_relaxed, memory_order_relaxed)) "
         "{\n"
      << "    if (observed != expected) break;\n"
      << "  }\n"
      << "  return observed;\n"
      << "}\n"
      << "static inline uint AtomicCAS(device uint* address, uint expected,\n"
      << "                             uint desired, int memory_order = 0) {\n"
      << "  (void)memory_order;\n"
      << "  uint observed = expected;\n"
      << "  while (!atomic_compare_exchange_weak_explicit(\n"
      << "             reinterpret_cast<device atomic_uint*>(address), "
         "&observed,\n"
      << "             desired, memory_order_relaxed, memory_order_relaxed)) "
         "{\n"
      << "    if (observed != expected) break;\n"
      << "  }\n"
      << "  return observed;\n"
      << "}\n"
      << "} /* namespace tl */\n\n";
}

void CodeGenTileLangMetal::EmitFPHelperPrelude() {
  // CPPMEGA: hybrid tl_pr_c granularity + stack-c switch dispatch.
  // Track if e4m3 base helper already emitted (e4m3fn alias depends on it).
  bool e4m3_emitted = false;
  for (int code : referenced_fp8_codes_) {
    switch (static_cast<DataType::TypeCode>(code)) {
    case DataType::kFloat8_e3m4:
      EmitFp8E3M4Helper();
      break;
    case DataType::kFloat8_e4m3:
      if (!e4m3_emitted) {
        EmitFp8E4M3Helper();
        e4m3_emitted = true;
      }
      break;
    case DataType::kFloat8_e4m3fn:
      if (!e4m3_emitted) {
        EmitFp8E4M3Helper();
        e4m3_emitted = true;
      }
      EmitFp8E4M3FnAliasHelper();
      break;
    case DataType::kFloat8_e4m3fnuz:
      EmitFp8E4M3FnuzHelper();
      break;
    case DataType::kFloat8_e4m3b11fnuz:
      EmitFp8E4M3B11FnuzHelper();
      break;
    case DataType::kFloat8_e5m2:
      EmitFp8E5M2Helper();
      break;
    case DataType::kFloat8_e5m2fnuz:
      EmitFp8E5M2FnuzHelper();
      break;
    case DataType::kFloat8_e8m0fnu:
      EmitFp8E8M0FnuHelper();
      break;
    default:
      // Other FP4/FP8 codes are fail-closed in PrintType/CastFromTo today.
      break;
    }
  }
  // The dot4-packed path stores its inputs as e4m3fn but uses the LUT path,
  // so it does not require __tvm_fp8_e4m3_to_half — emit only the LUT helpers.
  if (uses_fp8_dot4_)
    EmitFp8Dot4Helpers();
}

void CodeGenTileLangMetal::AddFunction(const GlobalVar &gvar,
                                       const PrimFunc &func) {
  // NOTE: There is no inter-function calls among Metal kernels.
  // For now we keep the metal codegen without inter-function call
  // process.
  // We can switch to follow the flow with inter-function call process
  // after the Metal function declaration is properly printed.
  // In Metal, for PrimFuncs with signature
  //    def func(A: Buffer, B: Buffer, x: int, y: float) -> None
  // where there are trailing pod parameters, the codegen emits a struct
  //    struct func_params{ x: int; y: float; }
  // for the function. In the flow of inter-function call process,
  // the struct will be emitted for every time a function is declared.
  // So consequently there are duplicate appearances of a same struct,
  // which makes the Metal compiler unable to recognize.

  // clear previous generated state.
  this->InitFuncState(func);
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->FreshName("v_");

  // CPPMEGA: hybrid tl_pr_c granularity + stack-c switch dispatch.
  // Scan the PrimFunc body for FP8/FP4 dtype use, then emit only the
  // matching `__tvm_fp8_*_to_half` helpers (and dot4 LUT/dot4_packed
  // overloads, when used) into decl_stream. This must run before
  // PrintStmt(func->body) so the helper symbols are visible to the kernel
  // body generated later in this method. Pure-fp32 kernels emit zero
  // helpers.
  this->CollectReferencedLowPrecisionDtypes(func);
  this->EmitAtomicAddHelperPrelude();
  this->EmitAtomicCASHelperPrelude();
  this->EmitFPHelperPrelude();

  // add to alloc buffer type.
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "kernel void "
               << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  size_t limit =
      target_->GetAttr<Integer>("max_function_args").value().IntValue();
  // RULE #1 (cppmega Path-C auto-split, design §6.5): an over-budget kernel
  // must NEVER silently reach newComputePipelineState. Count only handle
  // (buffer- binding) params -- the ABI-correct metric the Python auto-split
  // planner uses
  // (_kernel_parameter_count_for_target counts buffer_map entries only); scalar
  // by-value params consume no buffer-argument slot. If the buffer-binding
  // count exceeds the device buffer-argument limit, FAIL LOUD here instead of
  // emitting a kernel the Metal driver will crash on. The Python planner is
  // guaranteed to pre-empt this (it breaks/raises at the accept/reject point),
  // so reaching this throw means the pre-empt was bypassed -- surface it, do
  // not degrade.
  size_t num_buffer_binding_params = 0;
  for (size_t i = 0; i < func->params.size(); ++i) {
    if (func->params[i].dtype().is_handle()) {
      ++num_buffer_binding_params;
    }
  }
  ICHECK_LE(num_buffer_binding_params, limit)
      << "Metal kernel binds " << num_buffer_binding_params
      << " buffer arguments, exceeding the device max_function_args limit of "
      << limit
      << ". This over-budget kernel would crash newComputePipelineState; the "
         "Path-C auto-split planner must split it before codegen (RULE #1: "
         "fail "
         "loud, never emit a kernel that silently fails on the device).";
  std::unordered_map<std::string, const VarNode *> external_buffer_aliases;
  MetalBodyBufferWriteCollector write_collector;
  write_collector(func->body);
  const auto &written_buffer_vars = write_collector.written_vars();
  const auto &written_buffer_names = write_collector.written_names();
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v.dtype().is_handle())
      break;
    this->stream << "  ";
    std::string vid = AllocVarID(v.get());
    external_buffer_aliases.emplace(v->name_hint, v.get());
    auto it_buf = func->buffer_map.find(v);
    if (it_buf != func->buffer_map.end()) {
      const Buffer &buf = (*it_buf).second;
      external_buffer_aliases[buf->data->name_hint] = v.get();
      if (!buf->data.same_as(v)) {
        // Apache codegen prints BufferLoad/Store through Buffer.data, while
        // the Metal ABI only exposes the handle param.  Keep the alias local
        // to external buffer_map params; scratch buffers are still declared by
        // AllocBuffer/Allocate nodes.
        var_idmap_[buf->data.get()] = vid;
        alloc_storage_scope_[buf->data.get()] = "global";
        RegisterHandleType(buf->data.get(), buf->dtype);
      }
    }
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, this->stream);
    }
    bool readonly_buffer = false;
    if (uses_fp8_dot4_) {
      readonly_buffer = !written_buffer_vars.count(v.get()) &&
                        !written_buffer_names.count(v->name_hint);
      auto it_buf = func->buffer_map.find(v);
      if (readonly_buffer && it_buf != func->buffer_map.end()) {
        const Buffer &buf = (*it_buf).second;
        readonly_buffer = !written_buffer_vars.count(buf->data.get()) &&
                          !written_buffer_names.count(buf->data->name_hint);
      }
    }
    if (readonly_buffer) {
      this->stream << "const ";
    }
    PrintType(GetType(v), this->stream);
    // Register handle data type
    // TODO(tvm-team): consider simply keep type info in the
    // type annotation(via a normalizing rewriting).
    if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(v.get(), prim->dtype);
      }
    }
    this->stream << ' ' << vid << " [[ buffer(" << i << ") ]],\n";
  }
  // Setup normal arguments.
  size_t nargs = func->params.size() - num_buffer;
  std::string varg = name_supply_->FreshName("arg");
  if (nargs != 0) {
    std::string arg_buf_type =
        static_cast<std::string>(global_symbol.value()) + "_args_t";
    this->stream << "  constant " << arg_buf_type << "& " << varg
                 << " [[ buffer(" << num_buffer << ") ]],\n";
    // declare the struct
    decl_stream << "struct " << arg_buf_type << " {\n";
    for (size_t i = num_buffer; i < func->params.size(); ++i) {
      Var v = func->params[i];
      ICHECK(!v.dtype().is_handle());
      std::string vid = AllocVarID(v.get());
      std::ostringstream vref;
      if (v.dtype().bits() == 32) {
        decl_stream << "  ";
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vid << "[2];\n";
        vref << varg << "." << vid << "[0]";
      } else if (v.dtype().bits() == 64) {
        decl_stream << "  ";
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vid << ";\n";
        vref << varg << "." << vid;
      } else {
        // For non 32bit type, ref through arg union.
        decl_stream << "  __TVMArgUnion " << vid << ";\n";
        vref << varg << "." << vid << ".v_";
        PrintType(v.dtype(), vref);
      }
      var_idmap_[v.get()] = vref.str();
    }
    decl_stream << "};\n\n";
  }
  // Setup the thread group info.
  ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");
  int work_dim = 0;
  auto launch_params =
      func->GetAttr<ffi::Array<ffi::String>>(tirx::attr::kKernelLaunchParams)
          .value();
  for (const auto &tag : launch_params) {
    if (tag != runtime::launch_param::kUseDynamicSharedMemoryTag) {
      runtime::ThreadScope scope = runtime::ThreadScope::Create(tag);
      work_dim = std::max(work_dim, scope.dim_index + 1);
    }
  }

  // CPPMEGA / Path C: scan the body for the two MSL kernel-attribute
  // intrinsics emitted by tilelang/language/fp8_op.py
  // (``tir[x].metal.thread_position_in_grid_x``,
  // ``tir[x].metal.thread_index_in_simdgroup``). Each one we see has to
  // become an extra kernel argument bearing the matching MSL attribute,
  // because Metal exposes those builtins exclusively as kernel-signature
  // attributes — there is no free function spelling for them.
  bool needs_grid_tid_x = false;
  bool needs_simd_lane_id = false;
  {
    MetalFp8DTypeCollector kernel_intrin_collector;
    kernel_intrin_collector(func->body);
    needs_grid_tid_x = kernel_intrin_collector.uses_grid_tid_x;
    needs_simd_lane_id = kernel_intrin_collector.uses_simd_lane_id;
  }
  std::string grid_tid_x_id;
  std::string simd_lane_id_id;
  std::string grid_tid_x_expr;
  if (needs_grid_tid_x) {
    // Identifier name is observable from cppmega.mlx tests
    // (tests/test_tilelang_fp8_vecmat_path_c.py asserts the substring
    // ``gridThreadIdx`` in the generated MSL); keep the prefix stable.
    // FreshName still suffixes "_<n>" if it collides with a user var.
    grid_tid_x_id = name_supply_->FreshName("gridThreadIdx");
  }
  if (needs_simd_lane_id) {
    simd_lane_id_id = name_supply_->FreshName("simdLaneId");
  }

  if (work_dim != 0) {
    // use ushort by default for now
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    stream << " blockIdx [[threadgroup_position_in_grid]],\n";
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    // Trailing-comma handling: if extra MSL-attribute args follow we need a
    // comma here, otherwise none. Done explicitly so the no-extras path
    // remains byte-identical with the pre-Path-C output.
    if (needs_grid_tid_x || needs_simd_lane_id) {
      stream << " threadIdx [[thread_position_in_threadgroup]],\n";
    } else {
      stream << " threadIdx [[thread_position_in_threadgroup]]\n";
    }
  }
  if (needs_grid_tid_x) {
    stream << "  ";
    if (work_dim != 0) {
      // Metal requires all thread-position attributes in one kernel
      // signature to use the same scalar/vector width. The ordinary
      // TileLang launch parameters above use `work_dim`, so mirror that
      // width here and have the intrinsic expression read `.x`.
      PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
      grid_tid_x_expr = grid_tid_x_id + ".x";
    } else {
      stream << "uint";
      grid_tid_x_expr = grid_tid_x_id;
    }
    stream << " " << grid_tid_x_id << " [[thread_position_in_grid]]";
    if (needs_simd_lane_id) {
      stream << ",\n";
    } else {
      stream << "\n";
    }
  }
  if (needs_simd_lane_id) {
    stream << "  uint " << simd_lane_id_id
           << " [[thread_index_in_simdgroup]]\n";
  }
  // Stash the resolved MSL identifiers on the codegen instance via a
  // function-static map keyed by ``this``; the call-site visitor reads them
  // back when lowering the corresponding intrinsic. We use a function-static
  // (not a member) because per the Path-C task contract the .h header is not
  // part of this fix.
  GetMetalScalarIntrinIdMap()[this] = MetalScalarIntrinIds{
      needs_grid_tid_x ? grid_tid_x_expr : std::string(),
      needs_simd_lane_id ? simd_lane_id_id : std::string(),
  };
  thread_work_dim_ = work_dim;

  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  MetalBodyBufferAliasCollector alias_collector(
      std::move(external_buffer_aliases));
  alias_collector(func->body);
  for (const auto &kv : alias_collector.aliases()) {
    auto param_it = var_idmap_.find(kv.second.param_var);
    ICHECK(param_it != var_idmap_.end())
        << "Metal body buffer alias matched param without a codegen id: "
        << kv.first->name_hint;
    if (!var_idmap_.count(kv.first)) {
      // FlattenBuffer/PointerValueTypeRewrite may create a body-local
      // Buffer.data Var that is pointer-distinct from both the ABI param and
      // buffer_map entry.  Source codegen is pointer-identity based, so
      // register that external alias to the already emitted Metal parameter
      // before printing the body.  Scratch/local buffers are excluded by the
      // global-scope check.
      var_idmap_[kv.first] = param_it->second;
      alloc_storage_scope_[kv.first] = "global";
      RegisterHandleType(kv.first, kv.second.dtype);
    }
  }
  this->PrintStmt(func->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangMetal::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  // if we only have threadIdx.x
  // metal will directly print as threadIdx
  std::string vname = iv->thread_tag;
  if (thread_work_dim_ <= 1) {
    vname = vname.substr(0, iv->thread_tag.length() - 2);
  }
  var_idmap_[iv->var.get()] =
      CastFromTo(vname, DataType::UInt(thread_index_bits_), iv->var.dtype());
}

void CodeGenTileLangMetal::PrintType(DataType t,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  // CPPMEGA: explicit fail-closed branches for native FP8 (e4m3/e5m2/...) and
  // FP4 (e2m1fn) Metal storage. Without these, an arbitrary downstream pass
  // may be the first to choke on the unmapped dtype, hiding the underlying
  // "Metal codegen does not support native sub-byte/8-bit float storage"
  // contract from users. Mirrors stack-c's structural shape (PR #2144 / #2145
  // / #2147): only the 8-bit unsigned-byte storage emulation + LUT decode is
  // safe on Metal today, and only when explicitly opted into via the
  // uint8-boundary helpers in tilelang.intrinsics.metal_quant. Native dtypes
  // routed through the codegen surface are intentionally rejected here so the
  // caller sees a clear "Cannot convert type ... to Metal type" diagnostic
  // rather than a cryptic IR-level ICHECK from storage_rewrite or simdgroup
  // lowering. See docs/mlx_port_master_plan.md (Metal FP8/FP4 fail-closed).
  if (t.is_float8()) {
    if (t.is_float8_e4m3() || t.is_float8_e5m2()) {
      if (lanes == 8 || lanes == 16) {
        os << "uint" << lanes / 4;
      } else {
        os << "uchar";
      }
      if (lanes >= 2 && lanes <= 4) {
        os << lanes;
      } else if (!(lanes == 8 || lanes == 16)) {
        ICHECK_EQ(lanes, 1);
      }
      return;
    }
    LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
  }
  if (t.is_float4()) {
    LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
  }
  if (t.is_float() && t.bits() == 16 && lanes > 4 && lanes <= 8 &&
      lanes % 2 == 0) {
    os << "uint" << lanes / 2;
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    if (lanes == 3) {
      os << "packed_";
    }
    switch (t.bits()) {
    case 16:
      os << "half";
      break;
    case 32:
      os << "float";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
    case 8:
      os << "char";
      break;
    case 16:
      os << "short";
      break;
    case 32:
      os << "int";
      break;
    case 64:
      os << "long";
      break;
    case 1:
      os << "bool";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    ICHECK_EQ(lanes, 1) << "only scalar bfloat16 is supported in Metal codegen";
    EmitBFloat16Helper();
    os << "tvm_bfloat16";
    return;
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
}

void CodeGenTileLangMetal::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier not supported";
  }
}

void CodeGenTileLangMetal::PrintVecElemLoad(const std::string &vec, DataType t,
                                            int i,
                                            std::ostream &os) { // NOLINT(*)
  if (t.is_float16() && t.lanes() > 4) {
    os << "((thread half*)(&" << vec << "))[" << i << "]";
  } else if (t.is_float8() && t.lanes() > 4) {
    os << "((thread uchar*)(&" << vec << "))[" << i << "]";
  } else {
    os << vec << "[" << i << "]";
  }
}

void CodeGenTileLangMetal::PrintVecElemStore(const std::string &vec, DataType t,
                                             int i, const std::string &value) {
  this->PrintIndent();
  if (t.is_float16() && t.lanes() > 4) {
    stream << "((thread half*)(&" << vec << "))[" << i << "] = " << value
           << ";\n";
  } else if (t.is_float8() && t.lanes() > 4) {
    stream << "((thread uchar*)(&" << vec << "))[" << i << "] = " << value
           << ";\n";
  } else {
    stream << vec << "[" << i << "]"
           << " = " << value << ";\n";
  }
}

void CodeGenTileLangMetal::PrintStorageScope(const std::string &scope,
                                             std::ostream &os) { // NOLINT(*)
  if (scope == "global") {
    os << "device ";
  } else if (scope == "shared" || scope == "shared.dyn") {
    os << "threadgroup ";
  } else if (scope == "local" || scope == "local.fragment") {
    os << "thread ";
  } else if (scope == "metal.simdgroup") {
    // The actual simdgroup matrix declaration is emitted by the allocation
    // visitor; this branch only keeps incidental scope printing from aborting.
  } else {
    LOG(FATAL) << "Unknown storage scope `" << scope << "`";
  }
}

void CodeGenTileLangMetal::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  if (scope == "metal.simdgroup") {
    DataType matrix_dtype = op->dtype.element_of();
    size_t scalar_elements = constant_size * op->dtype.lanes();
    ICHECK(matrix_dtype == DataType::Float(16) ||
           matrix_dtype == DataType::Float(32) ||
           matrix_dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported, but got "
        << op->dtype;
    ICHECK(scalar_elements % 64 == 0)
        << "Only 8x8 matrix is supported, but got " << scalar_elements
        << " scalar elements\n";

    std::ostringstream dtype_os;
    PrintType(matrix_dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[op->buffer_var.get()] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << scalar_elements / 64 << "];\n";
  } else if (scope == "local.var") {
    ICHECK(op->dtype.is_scalar())
        << "Vector local.var allocation is not supported.";
    ICHECK_EQ(constant_size, 1)
        << "Only scalar local.var allocation is supported.";
    PrimExpr init = tirx::make_const(op->dtype, 0);
    auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
    if (init_it != op->annotations.end()) {
      PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
      if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
        user_init = tirx::Cast(op->dtype, user_init);
      }
      init = user_init;
    }
    std::string init_value = PrintExpr(init);
    PrintType(op->dtype, stream);
    stream << ' ' << vid << " = " << init_value << ";\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

// === Metal M5 cooperative tensor helpers (PR tile-ai/tilelang#2252) ========
// These helpers are no-ops unless the function actually allocates a buffer in
// the `metal.cooperative_tensor` scope, so M1-M4 kernel codegen is unchanged.

void CodeGenTileLangMetal::EmitCooperativeTensorLanePreambleIfNeeded() {
  if (!emitted_mpp_include_) {
    // Inject the MPP header at the top of decl_stream so it precedes the
    // first kernel body.  Note: this forces `-std=metal4.0` at xcrun time;
    // the build callback is responsible for selecting that flag when the
    // CT path is active.
    decl_stream
        << "#include "
           "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
    emitted_mpp_include_ = true;
  }
  if (!emitted_frag_lane_vars_) {
    // The MPP coop_tensor load/store layout depends on a 32-lane SIMD-group
    // fragment-lane mapping.  Emit it once per kernel body.
    this->PrintIndent();
    stream << "const ushort __lane = ((uint)threadIdx.x) % 32;\n";
    this->PrintIndent();
    stream << "const ushort __qid = __lane >> 2;\n";
    this->PrintIndent();
    stream << "const ushort __base_row = (__qid & 4) | ((__lane >> 1) & 3);\n";
    this->PrintIndent();
    stream << "const ushort __base_col = ((__qid & 2) | (__lane & 1)) * 4;\n";
    emitted_frag_lane_vars_ = true;
  }
}

std::string
CodeGenTileLangMetal::GetAddrSpaceOf(const PrimExpr &ptr_expr) const {
  if (auto *call = ptr_expr.as<CallNode>()) {
    if (call->op.same_as(builtin::address_of())) {
      if (auto *load = call->args[0].as<BufferLoadNode>()) {
        auto it = alloc_storage_scope_.find(load->buffer->data.get());
        if (it != alloc_storage_scope_.end()) {
          const std::string &scope = it->second;
          if (scope == "shared" || scope == "shared.dyn") {
            return "threadgroup";
          }
          if (scope == "local" || scope == "metal.cooperative_tensor") {
            return "thread";
          }
          if (scope == "global") {
            return "device";
          }
        }
      }
    }
    for (const auto &arg : call->args) {
      std::string result = GetAddrSpaceOf(arg);
      if (!result.empty() && result != "thread") {
        return result;
      }
    }
  }
  if (auto *var = ptr_expr.as<VarNode>()) {
    auto it = alloc_storage_scope_.find(var);
    if (it != alloc_storage_scope_.end()) {
      const std::string &scope = it->second;
      if (scope == "shared" || scope == "shared.dyn") {
        return "threadgroup";
      }
      if (scope == "local" || scope == "metal.cooperative_tensor") {
        return "thread";
      }
      if (scope == "global") {
        return "device";
      }
    }
  }
  return "thread";
}

void CodeGenTileLangMetal::EnsureCooperativeTensorBuffer(const Var &var) {
  if (cooperative_tensor_dtype_.count(var.get()) != 0) {
    return;
  }
  auto type_it = handle_data_type_.find(var.get());
  ICHECK(type_it != handle_data_type_.end())
      << "Cannot find variable allocation for cooperative_tensor: " << var;
  std::ostringstream dtype_os;
  PrintType(type_it->second, dtype_os);
  cooperative_tensor_dtype_[var.get()] = dtype_os.str();
}

void CodeGenTileLangMetal::VisitStmt_(const AllocBufferNode *op) {
  ICHECK(op->buffer.defined());
  std::string vid = AllocVarID(op->buffer->data.get());

  this->PrintIndent();
  size_t constant_size = 1;
  for (const auto &dim : op->buffer->shape) {
    const IntImmNode *dim_imm = dim.as<IntImmNode>();
    ICHECK(dim_imm) << "Can only handle constant size stack allocation for now";
    constant_size *= dim_imm->value;
  }
  ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer->data);
  alloc_storage_scope_[op->buffer->data.get()] = scope;
  DataType dtype = op->buffer->dtype;
  if (scope == "metal.cooperative_tensor") {
    // Metal M5 cooperative tensor scope (PR tile-ai/tilelang#2252).
    // Hardware: requires Apple M5+ silicon and MSL 4 (Xcode 16+).  Compiling
    // this code path on M1-M4 toolchains will fail at xcrun metal time; the
    // backend instruction selector in src/backend/metal/op/gemm.cc avoids
    // emitting this scope on non-M5 hardware via its shape check.
    DataType matrix_dtype = dtype.element_of();
    size_t scalar_elements = constant_size * dtype.lanes();
    ICHECK(matrix_dtype == DataType::Float(16) ||
           matrix_dtype == DataType::Float(32) ||
           matrix_dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported for "
           "cooperative_tensor, but got "
        << dtype;
    ICHECK(scalar_elements % 64 == 0)
        << "cooperative_tensor buffer size must be multiple of 64, got "
        << scalar_elements;

    EmitCooperativeTensorLanePreambleIfNeeded();

    std::ostringstream dtype_os;
    PrintType(matrix_dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    cooperative_tensor_dtype_[op->buffer->data.get()] = dtype_str;
    int elems_per_thread = static_cast<int>(scalar_elements) / 32;
    stream << "thread " << dtype_str << " " << vid << '[' << elems_per_thread
           << "];\n";
    if (dtype_str == "float" && elems_per_thread >= 16 &&
        elems_per_thread % 16 == 0) {
      int num_c_tiles = elems_per_thread / 16;
      ct_c_inlined_.insert(op->buffer->data.get());
      // `__pct_desc`/`__pct_op` are identical for every inlined accumulator, so
      // emit them at most once per kernel; re-emitting per buffer produces an
      // MSL "redefinition" error when a kernel has >1 cooperative_tensor.
      if (!emitted_pct_op_) {
        this->PrintIndent();
        stream << "constexpr auto __pct_desc = "
                  "mpp::tensor_ops::matmul2d_descriptor("
               << "16, 32, 16, false, false, true, "
               << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_"
                  "accumulate);\n";
        this->PrintIndent();
        stream << "mpp::tensor_ops::matmul2d<__pct_desc, "
                  "metal::execution_simdgroup> __pct_op;\n";
        emitted_pct_op_ = true;
      }
      // Give each buffer a unique base into the global `__pct_cN` namespace so
      // names never collide across multiple accumulators in one kernel.
      int base = ct_c_inlined_next_;
      ct_c_inlined_base_[op->buffer->data.get()] = base;
      ct_c_inlined_next_ += num_c_tiles;
      for (int t = 0; t < num_c_tiles; t++) {
        this->PrintIndent();
        stream << "auto __pct_c" << (base + t)
               << " = __pct_op.get_destination_cooperative_tensor<"
               << "decltype(__pct_op.get_left_input_cooperative_tensor<half, "
                  "half, float>()), "
               << "decltype(__pct_op.get_right_input_cooperative_tensor<half, "
                  "half, float>()), float>(); "
               << "for (ushort __i = 0; __i < 16; __i++) __pct_c" << (base + t)
               << "[__i] = 0.0f;\n";
      }
    }
  } else if (scope == "metal.simdgroup") {
    DataType matrix_dtype = dtype.element_of();
    size_t scalar_elements = constant_size * dtype.lanes();
    ICHECK(matrix_dtype == DataType::Float(16) ||
           matrix_dtype == DataType::Float(32) ||
           matrix_dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported, but got "
        << dtype;
    ICHECK(scalar_elements % 64 == 0)
        << "Only 8x8 matrix is supported, but got " << scalar_elements
        << " scalar elements\n";

    std::ostringstream dtype_os;
    PrintType(matrix_dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[op->buffer->data.get()] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << scalar_elements / 64 << "];\n";
  } else if (scope == "local.var") {
    ICHECK(dtype.is_scalar())
        << "Vector local.var allocation is not supported.";
    ICHECK_EQ(constant_size, 1)
        << "Only scalar local.var allocation is supported.";
    PrimExpr init = tirx::make_const(dtype, 0);
    auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
    if (init_it != op->annotations.end()) {
      PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
      if (!user_init.dtype().is_void() && user_init.dtype() != dtype) {
        user_init = tirx::Cast(dtype, user_init);
      }
      init = user_init;
    }
    std::string init_value = PrintExpr(init);
    PrintType(dtype, stream);
    stream << ' ' << vid << " = " << init_value << ";\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer->data.get(), dtype);
  if (op->annotations.count(tirx::attr::kVolatile)) {
    MarkVolatile(op->buffer->data.get());
  }
}

void CodeGenTileLangMetal::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "pragma_unroll_factor") {
    const auto *factor = op->value.as<IntImmNode>();
    ICHECK(factor) << "pragma_unroll_factor expects an IntImm value";
    const auto *loop_var = op->node.as<VarNode>();
    ICHECK(loop_var) << "pragma_unroll_factor expects a loop var node";
    unroll_factor_[loop_var] = Downcast<IntImm>(op->value);
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitStmt_(const BindNode *op) {
  const auto *call = op->value.as<CallNode>();
  if (uses_fp8_dot4_ && op->var.dtype().is_handle() && call != nullptr &&
      call->op.same_as(builtin::address_of()) && call->args.size() == 1) {
    if (const auto *load = call->args[0].as<BufferLoadNode>();
        load != nullptr && load->buffer->dtype.is_float8()) {
      std::string value = PrintExpr(op->value);
      std::string scope;
      auto it = alloc_storage_scope_.find(load->buffer->data.get());
      if (it != alloc_storage_scope_.end()) {
        scope = it->second;
      }
      if (scope.empty()) {
        scope = GetPtrStorageScope(load->buffer->data);
      }
      PrintIndent();
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, stream);
      }
      stream << "const uint* " << AllocVarID(op->var.get())
             << " = reinterpret_cast<";
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, stream);
      }
      stream << "const uint*>(" << value << ");\n";
      return;
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitStmt_(const ForNode *op) {
  if (op->kind == ForKind::kUnrolled) {
    PrintIndent();
    auto ann = op->annotations.find("pragma_unroll_factor");
    auto it = unroll_factor_.find(op->loop_var.get());
    if (ann != op->annotations.end()) {
      const auto *factor = (*ann).second.as<IntImmNode>();
      ICHECK(factor) << "pragma_unroll_factor expects an IntImm value";
      stream << "#pragma unroll " << PrintExpr(Downcast<IntImm>((*ann).second))
             << "\n";
    } else if (it != unroll_factor_.end()) {
      stream << "#pragma unroll " << PrintExpr(it->second) << "\n";
    } else {
      stream << "#pragma unroll\n";
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitExpr_(const BufferLoadNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::string scope;
  auto it = alloc_storage_scope_.find(op->buffer->data.get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(op->buffer->data);
  }
  if (scope == "local.var") {
    ICHECK_EQ(op->indices.size(), 1)
        << "Load from non-flat local.var memory not supported.";
    ICHECK(op->dtype.is_scalar()) << "Vector local.var load is not supported.";
    auto index = op->indices[0].as<IntImmNode>();
    ICHECK(index && index->value == 0)
        << "local.var load requires scalar index 0.";
    os << GetVarID(op->buffer->data.get());
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangMetal::VisitExpr_(const VarNode *op,
                                      std::ostream &os) { // NOLINT(*)
  if (MetalFp8DTypeCollector::IsGridTidXVarName(op->name_hint)) {
    auto it = GetMetalScalarIntrinIdMap().find(this);
    ICHECK(it != GetMetalScalarIntrinIdMap().end() &&
           !it->second.grid_tid_x.empty())
        << "canonical Metal grid-tid scalar var referenced from a kernel "
        << "whose signature was not augmented with [[thread_position_in_grid]]";
    os << it->second.grid_tid_x;
    return;
  }
  if (MetalFp8DTypeCollector::IsSimdLaneIdVarName(op->name_hint)) {
    auto it = GetMetalScalarIntrinIdMap().find(this);
    ICHECK(it != GetMetalScalarIntrinIdMap().end() &&
           !it->second.simd_lane_id.empty())
        << "canonical Metal SIMD-lane scalar var referenced from a kernel "
        << "whose signature was not augmented with "
           "[[thread_index_in_simdgroup]]";
    os << it->second.simd_lane_id;
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangMetal::VisitStmt_(const BufferStoreNode *op) {
  std::string scope;
  auto it = alloc_storage_scope_.find(op->buffer->data.get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(op->buffer->data);
  }
  if (scope == "local.var") {
    ICHECK_EQ(op->indices.size(), 1)
        << "Store to non-flat local.var memory not supported.";
    ICHECK(op->value.dtype().is_scalar())
        << "Vector local.var store is not supported.";
    auto index = op->indices[0].as<IntImmNode>();
    ICHECK(index && index->value == 0)
        << "local.var store requires scalar index 0.";
    std::string value = PrintExpr(op->value);
    this->PrintIndent();
    stream << GetVarID(op->buffer->data.get()) << " = " << value << ";\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitExpr_(const SelectNode *op,
                                      std::ostream &os) { // NOLINT(*)
  os << "select(" << PrintExpr(op->false_value) << ", "
     << PrintExpr(op->true_value) << ", " << PrintExpr(op->condition) << ")";
}

void CodeGenTileLangMetal::VisitExpr_(const BroadcastNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::string v = PrintExpr(op->value);
  int lanes = op->dtype.lanes();
  if (op->dtype.is_float16() && lanes > 4 && lanes % 2 == 0) {
    os << "uint" << lanes / 2 << "(";
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      os << "as_type<uint>(half2(" << v << ", " << v << "))";
    }
    os << ')';
  } else {
    PrintType(op->dtype, os);
    os << "(";
    for (int i = 0; i < lanes; ++i) {
      if (i != 0)
        os << ", ";
      os << v;
    }
    os << ')';
  }
}

void CodeGenTileLangMetal::VisitExpr_(const CallNode *op,
                                      std::ostream &os) { // NOLINT(*)
  CHECK(!op->op.as<GlobalVarNode>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls PrimFunc "
      << op->op;
  auto f_check_simdgroup_shape = [](PrimExpr col, PrimExpr row) {
    ICHECK(col->IsInstance<IntImmNode>() && row->IsInstance<IntImmNode>())
        << "Only constant shape is supported for simdgroup matrix, but got "
        << col << "x" << row;
    int col_val = col.as<IntImmNode>()->value;
    int row_val = row.as<IntImmNode>()->value;
    ICHECK(col_val == 8 && row_val == 8)
        << "Only 8x8 matrix is supported, but got " << col_val << "x"
        << row_val;
  };
  if (op->op.same_as(builtin::make_filled_simdgroup_matrix())) {
    ICHECK_EQ(op->args.size(), 5);
    Var var = Downcast<Var>(op->args[0]);
    // Get the data type of the simdgroup matrix
    auto it = simdgroup_dtype_.find(var.get());
    ICHECK(it != simdgroup_dtype_.end())
        << "Cannot find variable allocation for simdgroup: " << var;
    const std::string &dtype_str = it->second;
    f_check_simdgroup_shape(op->args[3], op->args[4]);
    os << PrintExpr(var) << "[" << PrintExpr(op->args[1])
       << "] = make_filled_simdgroup_matrix<" << dtype_str << ", "
       << PrintExpr(op->args[3]) << ", " << PrintExpr(op->args[4]) << ">("
       << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_load())) {
    ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_load(" << PrintExpr(op->args[0]) << "["
       << PrintExpr(op->args[1]) << "], " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_store())) {
    ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_store(" << PrintExpr(op->args[0]) << "["
       << PrintExpr(op->args[1]) << "], " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_multiply_accumulate())) {
    ICHECK_EQ(op->args.size(), 8);
    os << "simdgroup_multiply_accumulate("                                 //
       << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "], " //
       << PrintExpr(op->args[2]) << "[" << PrintExpr(op->args[3]) << "], " //
       << PrintExpr(op->args[4]) << "[" << PrintExpr(op->args[5]) << "], " //
       << PrintExpr(op->args[6]) << "[" << PrintExpr(op->args[7]) << "])";
  } else if (op->op.same_as(tl::cooperative_tensor_fill())) {
    // Metal M5 cooperative tensor fill (PR tile-ai/tilelang#2252).  Fills
    // one 16x32 tile with the given value.  When the destination buffer is
    // float and tile-aligned, also re-zero the pre-staged __pct_cN
    // destination cooperative_tensor (constant-idx fast path).
    ICHECK_EQ(op->args.size(), 5);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string val = PrintExpr(op->args[2]);
    int rows = op->args[3].as<IntImmNode>()->value;
    int cols = op->args[4].as<IntImmNode>()->value;
    int elems_per_tile = rows * cols / 32;
    Var fill_v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(fill_v);
    bool is_inlined = ct_c_inlined_.count(fill_v.get()) > 0;
    auto *fill_idx_imm = op->args[1].as<IntImmNode>();
    os << "for (ushort __i = 0; __i < " << elems_per_tile << "; __i++) " << var
       << "[" << idx << " * " << elems_per_tile << " + __i] = " << val;
    if (is_inlined && fill_idx_imm) {
      int fill_base = ct_c_inlined_base_.count(fill_v.get())
                          ? ct_c_inlined_base_[fill_v.get()]
                          : 0;
      os << "; for (ushort __i = 0; __i < " << elems_per_tile << "; __i++) "
         << "__pct_c" << (fill_base + fill_idx_imm->value) << "[__i] = " << val;
    }
  } else if (op->op.same_as(tl::cooperative_tensor_load())) {
    // Load `rows`x`cols` from device/threadgroup pointer into a per-thread
    // tile slice of the cooperative_tensor at index `idx`.
    //
    // LAYOUT-CORRECTNESS FIX: for the A (LEFT) / B (RIGHT) matmul operands we
    // do NOT stage into a thread-private array via the 8x8 simdgroup
    // fragment-lane map (__base_row/__base_col) anymore.  The mpp cooperative
    // input tensor has an opaque, implementation-defined per-lane element
    // distribution, so a later linear copy from such a staging array scrambles
    // the operands (observed maxdiff ~22 on M4).  Instead we RECORD the
    // threadgroup/device source view here and let the matmul2d emission call
    // mpp's native `cooperative_tensor.load(...)`, which fills the input tensor
    // in mpp's own internal order.  The DEST (C) operand keeps the legacy
    // fragment-map path because the C accumulator round-trips through SMEM.
    ICHECK_GE(op->args.size(), 11);
    int operand_role = op->args[10].as<IntImmNode>()->value;
    if (operand_role == 0 || operand_role == 1) {
      // A (LEFT) / B (RIGHT): emit nothing.  The matmul2d emission fills the
      // mpp input cooperative tensors from the threadgroup/device source views
      // passed directly into cooperative_tensor_multiply_accumulate, using
      // mpp's native cooperative_tensor.load (correct element distribution).
      // The thread-private A_local/B_local staging arrays are now unused by the
      // coop path and get DCE'd by the Metal compiler.
      return;
    }
    // operand_role == 2 (DEST/C): load the existing C accumulator from the
    // (row-major, row-stride) SMEM/device tile into the mpp destination
    // cooperative tensor.  For the inlined __pct_cN accumulator (constant tile
    // index) this MUST use mpp's native cooperative_tensor.load so the element
    // distribution matches what __op.run / .store use; the legacy fragment-map
    // scatter below corrupted the C accumulator on the non-clear path.
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string src_ptr = PrintExpr(op->args[2]);
    std::string stride = PrintExpr(op->args[3]);
    int rows = op->args[4].as<IntImmNode>()->value;
    int cols = op->args[5].as<IntImmNode>()->value;
    Var v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v.get());
    ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(op->args[2]);
    bool dest_inlined = ct_c_inlined_.count(v.get()) > 0;
    auto *load_idx_imm = op->args[1].as<IntImmNode>();
    if (dest_inlined && load_idx_imm) {
      int nfrag_r_i = rows / 16;
      int nfrag_c_i = cols / 16;
      int total_elems_i = nfrag_r_i * nfrag_c_i * 8;
      int mma_tiles_per_load = total_elems_i / 16;
      ICHECK(mma_tiles_per_load == 1)
          << "inlined cooperative_tensor C load expects a single 16x32 "
             "destination micro-tile per call, got "
          << mma_tiles_per_load;
      int buf_base =
          ct_c_inlined_base_.count(v.get()) ? ct_c_inlined_base_[v.get()] : 0;
      int base_pct = buf_base + load_idx_imm->value * mma_tiles_per_load;
      // mpp C tile: row-major rows x cols (M x N) -> mpp extents (N, M),
      // strides {1, stride}.
      os << "{ metal::tensor<" << addr_space << " " << dtype
         << ", metal::dextents<int32_t, 2>, metal::tensor_inline> __ts_c(("
         << addr_space << " " << dtype << "*)" << src_ptr
         << ", metal::dextents<int32_t, 2>(" << cols << ", " << rows
         << "), metal::array<int32_t, 2>{1, (int32_t)(" << stride << ")}); "
         << "__pct_c" << base_pct << ".load(__ts_c); }";
      return;
    }
    // Non-inlined fallback: legacy fragment-map gather into the thread-private
    // C_ct staging array.
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    os << "{ " << addr_space << " " << dtype << "* __src = (" << addr_space
       << " " << dtype << "*)" << src_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << (nfrag_r * nfrag_c * 8) << " + " << elem_offset << "]) = "
           << "*(" << addr_space << " " << dtype << "4*)(&__src[__r0 * "
           << stride << " + __c0]); "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << (nfrag_r * nfrag_c * 8) << " + " << (elem_offset + 4) << "]) = "
           << "*(" << addr_space << " " << dtype << "4*)(&__src[__r1 * "
           << stride << " + __c0]); } ";
        elem_offset += 8;
      }
    }
    os << "}";
  } else if (op->op.same_as(tl::cooperative_tensor_store())) {
    // Inverse of cooperative_tensor_load.  Also when the source C buffer is
    // an inlined `__pct_cN` accumulator, scatter that back into the thread-
    // private array before the final store.
    ICHECK_GE(op->args.size(), 11);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string dst_ptr = PrintExpr(op->args[2]);
    std::string stride = PrintExpr(op->args[3]);
    int rows = op->args[4].as<IntImmNode>()->value;
    int cols = op->args[5].as<IntImmNode>()->value;
    Var v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v.get());
    ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(op->args[2]);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    int total_elems = nfrag_r * nfrag_c * 8;
    bool is_inlined = ct_c_inlined_.count(v.get()) > 0;
    auto *store_idx_imm = op->args[1].as<IntImmNode>();
    // LAYOUT-CORRECTNESS FIX: for the inlined float accumulator the result
    // lives in the mpp destination cooperative tensor `__pct_cN`.  Store it
    // back to the (row-major, row-stride) C SMEM/device tile via mpp's native
    // `cooperative_tensor.store(metal::tensor<...>)`, which uses mpp's internal
    // element distribution -- the same one `__op.run` wrote.  The previous
    // fragment-map scatter (__base_row/__base_col) did NOT match that
    // distribution and corrupted the C tile.  Each inlined `__pct_cN` is one
    // `micro_size_x` x `micro_size_y` (16x32) destination tensor, i.e. one
    // store micro-tile, so total_elems==16 and mma_tiles_per_store==1 here.
    if (is_inlined && store_idx_imm) {
      int mma_tiles_per_store = total_elems / 16;
      ICHECK(mma_tiles_per_store == 1)
          << "inlined cooperative_tensor C store expects a single 16x32 "
             "destination micro-tile per call, got "
          << mma_tiles_per_store;
      int buf_base =
          ct_c_inlined_base_.count(v.get()) ? ct_c_inlined_base_[v.get()] : 0;
      int base_pct = buf_base + store_idx_imm->value * mma_tiles_per_store;
      // mpp C tile: row-major rows x cols (M x N), row stride `stride`.
      // mpp-order extents are (cols, rows) = (N, M), strides {1, stride}.
      os << "{ metal::tensor<" << addr_space << " " << dtype
         << ", metal::dextents<int32_t, 2>, metal::tensor_inline> __ts_c(("
         << addr_space << " " << dtype << "*)" << dst_ptr
         << ", metal::dextents<int32_t, 2>(" << cols << ", " << rows
         << "), metal::array<int32_t, 2>{1, (int32_t)(" << stride << ")}); "
         << "__pct_c" << base_pct << ".store(__ts_c); }";
      return;
    }
    // Non-inlined fallback: legacy fragment-map scatter from the thread-private
    // C_ct staging array.  (Only reached when the accumulator was not promoted
    // to an inlined __pct_cN, e.g. non-float dest.)
    os << "{ " << addr_space << " " << dtype << "* __dst = (" << addr_space
       << " " << dtype << "*)" << dst_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(" << addr_space << " " << dtype << "4*)(&__dst[__r0 * "
           << stride << " + __c0]) = "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << total_elems << " + " << elem_offset << "]); "
           << "*(" << addr_space << " " << dtype << "4*)(&__dst[__r1 * "
           << stride << " + __c0]) = "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << total_elems << " + " << (elem_offset + 4) << "]); } ";
        elem_offset += 8;
      }
    }
    os << "}";
  } else if (op->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
    // Emit `mpp::tensor_ops::matmul2d`.  When the accumulator is the
    // pre-staged inlined float accumulator (constant idx), reuse the
    // existing __pct_cN destination; otherwise materialize a fresh
    // destination CT on the stack.
    ICHECK_GE(op->args.size(), 13);
    int M = op->args[8].as<IntImmNode>()->value;
    int N = op->args[9].as<IntImmNode>()->value;
    int K = op->args[10].as<IntImmNode>()->value;
    bool trans_a = op->args[11].as<IntImmNode>()->value != 0;
    bool trans_b = op->args[12].as<IntImmNode>()->value != 0;

    std::string c_var = PrintExpr(op->args[0]);
    std::string c_idx = PrintExpr(op->args[1]);

    Var a_v = Downcast<Var>(op->args[2]);
    Var c_v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(a_v);
    EnsureCooperativeTensorBuffer(Downcast<Var>(op->args[4]));
    EnsureCooperativeTensorBuffer(c_v);
    auto a_it = cooperative_tensor_dtype_.find(a_v.get());
    auto c_it = cooperative_tensor_dtype_.find(c_v.get());
    ICHECK(a_it != cooperative_tensor_dtype_.end());
    ICHECK(c_it != cooperative_tensor_dtype_.end());
    std::string a_dtype = a_it->second;
    std::string c_dtype = c_it->second;

    int c_elems = M * N / 32;

    ICHECK(M == 32 || N == 32 || K == 32)
        << "MPP matmul2d requires at least one of M, N, K to be 32, got " << M
        << "x" << N << "x" << K;

    bool c_inlined = ct_c_inlined_.count(c_v.get()) > 0;
    auto *c_idx_imm = op->args[1].as<IntImmNode>();
    bool c_idx_const = c_inlined && c_idx_imm != nullptr;
    int c_buf_base =
        ct_c_inlined_base_.count(c_v.get()) ? ct_c_inlined_base_[c_v.get()] : 0;

    // LAYOUT-CORRECTNESS FIX: fill the mpp input cooperative tensors via mpp's
    // native `cooperative_tensor.load(metal::tensor<...>)`.  This replaces the
    // previous linear copy from a fragment-map staging array, which scrambled
    // the operands because the cooperative-tensor per-lane element order is
    // implementation-defined and unrelated to the 8x8 simdgroup fragment map.
    //
    // The A/B threadgroup (or device) source views are passed in directly as
    // trailing args [13..16] = (a_ptr, a_stride, b_ptr, b_stride), recomputed
    // by the Python mma macro from the same access pattern the ldmatrix step
    // used.  This is self-contained -- no fragile cross-call recording across
    // renamed loop variables.
    ICHECK_GE(op->args.size(), 17)
        << "cooperative_tensor_multiply_accumulate requires the trailing "
           "a_ptr/a_stride/b_ptr/b_stride source-view args for the native-load "
           "layout fix";
    std::string a_ptr = PrintExpr(op->args[13]);
    std::string a_stride = PrintExpr(op->args[14]);
    std::string b_ptr = PrintExpr(op->args[15]);
    std::string b_stride = PrintExpr(op->args[16]);
    std::string a_addr = GetAddrSpaceOf(op->args[13]);
    std::string b_addr = GetAddrSpaceOf(op->args[15]);

    // Build a rank-2 mpp source tensor view over the (row-major, row-stride)
    // SMEM/device tile.  Per the MPP matmul2d contract (NN): for A,
    // extent(0)=K, extent(1)=M; for B, extent(0)=N, extent(1)=K.  The physical
    // tile is row-major (rows x cols, row stride `stride`), so the mpp-order
    // extents are (cols, rows) and strides {1, stride}.  A tile is M(rows) x
    // K(cols); B tile is K(rows) x N(cols).  The descriptor's transpose flags
    // tell mpp how to interpret it.
    auto emit_src_tensor = [&](const std::string &name, const std::string &addr,
                               const std::string &dt, const std::string &ptr,
                               const std::string &strd, int extent0,
                               int extent1) {
      os << "metal::tensor<" << addr << " " << dt
         << ", metal::dextents<int32_t, 2>, metal::tensor_inline> " << name
         << "((" << addr << " " << dt << "*)" << ptr
         << ", metal::dextents<int32_t, 2>(" << extent0 << ", " << extent1
         << "), metal::array<int32_t, 2>{1, (int32_t)(" << strd << ")}); ";
    };

    os << "{ "
       << "constexpr auto __desc = mpp::tensor_ops::matmul2d_descriptor(" << M
       << ", " << N << ", " << K << ", " << (trans_a ? "true" : "false") << ", "
       << (trans_b ? "true" : "false") << ", true, "
       << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate); "
       << "mpp::tensor_ops::matmul2d<__desc, metal::execution_simdgroup> __op; "
       << "auto __ct_a = __op.get_left_input_cooperative_tensor<" << a_dtype
       << ", " << a_dtype << ", " << c_dtype << ">(); "
       << "auto __ct_b = __op.get_right_input_cooperative_tensor<" << a_dtype
       << ", " << a_dtype << ", " << c_dtype << ">(); ";
    // A: extent0=K, extent1=M ; B: extent0=N, extent1=K.
    emit_src_tensor("__ts_a", a_addr, a_dtype, a_ptr, a_stride, K, M);
    emit_src_tensor("__ts_b", b_addr, a_dtype, b_ptr, b_stride, N, K);
    os << "__ct_a.load(__ts_a); __ct_b.load(__ts_b); ";
    if (c_idx_const) {
      os << "__op.run(__ct_a, __ct_b, __pct_c"
         << (c_buf_base + c_idx_imm->value) << "); }";
    } else {
      os << "auto __ct_c = __op.get_destination_cooperative_tensor<"
         << "decltype(__ct_a), decltype(__ct_b), " << c_dtype << ">(); "
         << "for (ushort __i = 0; __i < " << c_elems << "; __i++) "
         << "__ct_c[__i] = " << c_var << "[" << c_idx << " * " << c_elems
         << " + __i]; "
         << "__op.run(__ct_a, __ct_b, __ct_c); "
         << "for (ushort __i = 0; __i < " << c_elems << "; __i++) " << c_var
         << "[" << c_idx << " * " << c_elems << " + __i] = __ct_c[__i]; }";
    }
  } else if (op->op.same_as(builtin::reinterpret())) {
    // generate as_type<TYPE>(ARG)
    os << "(as_type<";
    this->PrintType(op->dtype, os);
    os << ">(";
    this->PrintExpr(op->args[0], os);
    os << "))";
  } else if (op->op.same_as(tl::shfl_xor_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "simd_shuffle_xor(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(tl::sync_threads_partial())) {
    // Apple SIMD groups are always convergent at the simd-group level, so
    // partial-lane sync collapses to a simdgroup_barrier. mask + n_threads
    // are accepted for source compatibility but ignored at codegen time.
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.sync_threads_partial expects <mask, n_threads>.";
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (op->op.same_as(tl::atomic_add_elem_op())) {
    ICHECK_GE(op->args.size(), 2U)
        << "tl.atomic_add_elem_op expects dst_ptr and src_value.";
    this->PrintIndent();
    this->stream << "tl::AtomicAdd(" << PrintExpr(op->args[0]) << ", "
                 << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_add_ret_elem_op())) {
    ICHECK_GE(op->args.size(), 2U)
        << "tl.atomic_add_ret_elem_op expects dst_ptr and src_value.";
    os << "tl::AtomicAdd(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(builtin::atomic_cas())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tir.atomic_cas expects dst_ptr, expected, and desired.";
    ICHECK(op->dtype.is_int() || op->dtype.is_uint())
        << "Metal tir.atomic_cas supports int/uint dtypes; got " << op->dtype;
    ICHECK_EQ(op->dtype.bits(), 32)
        << "Metal tir.atomic_cas supports 32-bit atomics; got " << op->dtype;
    os << "tl::AtomicCAS(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(tl::atomic_xchg_elem_op()) ||
             op->op.same_as(tl::atomic_xchg_ret_elem_op()) ||
             op->op.same_as(tl::atomic_and_elem_op()) ||
             op->op.same_as(tl::atomic_and_ret_elem_op()) ||
             op->op.same_as(tl::atomic_or_elem_op()) ||
             op->op.same_as(tl::atomic_or_ret_elem_op()) ||
             op->op.same_as(tl::atomic_xor_elem_op()) ||
             op->op.same_as(tl::atomic_xor_ret_elem_op())) {
    // Metal atomic_xchg/and/or/xor: emit MSL ``atomic_*_explicit`` from the
    // ``<metal_atomic>`` header. Metal restricts these primitives to
    // ``atomic_int`` / ``atomic_uint`` storage; for fp dtypes the runtime
    // would need a CAS loop, so we error out cleanly here. The Op enum is
    // statically registered in src/op/builtin.cc.
    bool is_int_atomic =
        op->args.size() >= 2 &&
        (op->args[1].dtype().is_int() || op->args[1].dtype().is_uint());
    ICHECK(is_int_atomic)
        << "Metal atomic xchg/and/or/xor only supports atomic_int / "
        << "atomic_uint dtypes; got value dtype " << op->args[1].dtype()
        << ". TODO: implement <op> for fp dtype on Metal via CAS";
    std::string fn;
    if (op->op.same_as(tl::atomic_xchg_elem_op()) ||
        op->op.same_as(tl::atomic_xchg_ret_elem_op())) {
      fn = "atomic_exchange_explicit";
    } else if (op->op.same_as(tl::atomic_and_elem_op()) ||
               op->op.same_as(tl::atomic_and_ret_elem_op())) {
      fn = "atomic_fetch_and_explicit";
    } else if (op->op.same_as(tl::atomic_or_elem_op()) ||
               op->op.same_as(tl::atomic_or_ret_elem_op())) {
      fn = "atomic_fetch_or_explicit";
    } else {
      fn = "atomic_fetch_xor_explicit";
    }
    os << fn << "(" << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[1])
       << ", memory_order_relaxed)";
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsFp8Dot4Intrin(opn->name)) {
    // CPPMEGA / Path C: lower packed e4m3 dot4 to word loads plus LUT decode.
    // FP8 buffer binds above cast the hot-loop aliases to uint pointers, so
    // the load helper becomes a direct word load for the vecmat fast path.
    ICHECK_EQ(op->args.size(), 4)
        << "tir[x].metal.fp8_e4m3_dot4 expects 4 args (a_ptr, b_ptr, "
        << "a_word_idx, b_word_idx), got " << op->args.size();
    os << "__tvm_fp8_e4m3_dot4_words(__tvm_fp8_load_u32("
       << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[2]) << "), "
       << "__tvm_fp8_load_u32(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[3]) << "))";
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsFp8LoadU32Intrin(opn->name)) {
    ICHECK_EQ(op->args.size(), 2)
        << "tir[x].metal.fp8_load_u32 expects 2 args (ptr, word_idx), got "
        << op->args.size();
    os << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "]";
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsFp8Dot4WordsIntrin(opn->name)) {
    ICHECK_EQ(op->args.size(), 2)
        << "tir[x].metal.fp8_e4m3_dot4_words expects 2 args (a_word, "
        << "b_word), got " << op->args.size();
    std::string pa = PrintExpr(op->args[0]);
    std::string pb = PrintExpr(op->args[1]);
    os << "(__tvm_fp8_e4m3fn_lut[" << pa << " & 0xFFu] * __tvm_fp8_e4m3fn_lut["
       << pb << " & 0xFFu]"
       << " + __tvm_fp8_e4m3fn_lut[(" << pa
       << " >> 8) & 0xFFu] * __tvm_fp8_e4m3fn_lut[(" << pb << " >> 8) & 0xFFu]"
       << " + __tvm_fp8_e4m3fn_lut[(" << pa
       << " >> 16) & 0xFFu] * __tvm_fp8_e4m3fn_lut[(" << pb
       << " >> 16) & 0xFFu]"
       << " + __tvm_fp8_e4m3fn_lut[(" << pa
       << " >> 24) & 0xFFu] * __tvm_fp8_e4m3fn_lut[(" << pb
       << " >> 24) & 0xFFu])";
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsGridTidXIntrin(opn->name)) {
    // CPPMEGA / Path C: ``tir[x].metal.thread_position_in_grid_x`` lowers to
    // the kernel argument we declared with the
    // ``[[thread_position_in_grid]]`` MSL attribute in
    // PrintFuncDecl. Look up the freshly-supplied identifier on the
    // side-table populated during signature emission.
    auto it = GetMetalScalarIntrinIdMap().find(this);
    ICHECK(it != GetMetalScalarIntrinIdMap().end() &&
           !it->second.grid_tid_x.empty())
        << "tir[x].metal.thread_position_in_grid_x referenced from a kernel "
        << "whose signature was not augmented with [[thread_position_in_grid]]"
        << " — body collector missed the call site, file a bug.";
    os << it->second.grid_tid_x;
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsSimdLaneIdIntrin(opn->name)) {
    // CPPMEGA / Path C: ``tir[x].metal.thread_index_in_simdgroup`` lowers to
    // the kernel argument we declared with the
    // ``[[thread_index_in_simdgroup]]`` MSL attribute. Same lookup pattern
    // as above.
    auto it = GetMetalScalarIntrinIdMap().find(this);
    ICHECK(it != GetMetalScalarIntrinIdMap().end() &&
           !it->second.simd_lane_id.empty())
        << "tir[x].metal.thread_index_in_simdgroup referenced from a kernel "
        << "whose signature was not augmented with "
           "[[thread_index_in_simdgroup]]"
        << " — body collector missed the call site, file a bug.";
    os << it->second.simd_lane_id;
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsThreadgroupTidXIntrin(opn->name)) {
    os << "((int)threadIdx.x)";
  } else if (auto *opn = op->op.as<OpNode>();
             opn != nullptr &&
             MetalFp8DTypeCollector::IsSimdSumIntrin(opn->name)) {
    ICHECK_EQ(op->args.size(), 1)
        << "tir[x].metal.simd_sum expects 1 arg, got " << op->args.size();
    os << "simd_sum(" << PrintExpr(op->args[0]) << ")";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangMetal::VisitExpr_(const FloatImmNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::ostringstream temp;
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      temp << "-";
    }
    temp << "INFINITY";
  } else if (std::isnan(op->value)) {
    temp << "NAN";
  } else {
    temp << std::scientific << op->value;
    if (op->dtype.bits() == 32)
      temp << 'f';
    else if (op->dtype.bits() == 16)
      temp << 'h';
  }
  MarkConst(temp.str());
  os << temp.str();
}

std::string CodeGenTileLangMetal::CastFromTo(std::string value, DataType from,
                                             DataType target) {
  if (from == target) {
    return value;
  }
  if (from.is_scalar() && from.is_float8() &&
      (target.is_float16() || target == DataType::Float(32))) {
    const char *helper = nullptr;
    if (from.is_float8_e4m3()) {
      helper = "__tvm_fp8_e4m3_to_half";
    } else if (from.is_float8_e5m2()) {
      helper = "__tvm_fp8_e5m2_to_half";
    }
    if (helper != nullptr) {
      std::string decoded = std::string(helper) + "(" + value + ")";
      if (target == DataType::Float(32)) {
        return "((float)" + decoded + ")";
      }
      return decoded;
    }
  }
  if (from.is_bfloat16() && target.is_float()) {
    std::string decoded = "__tvm_bfloat16_to_float(" + value + ")";
    if (target.bits() == 32) {
      return "((float)" + decoded + ")";
    }
    if (target.bits() == 16) {
      return "((half)" + decoded + ")";
    }
  }
  if (from.is_float() && target.is_bfloat16()) {
    if (from.bits() == 32) {
      return "tvm_bfloat16(" + value + ")";
    }
    if (from.bits() == 16) {
      return "tvm_bfloat16((float)" + value + ")";
    }
  }
  return CodeGenC::CastFromTo(std::move(value), from, target);
}

ffi::Module BuildTileLangMetal(IRModule mod, Target target) {
  bool output_ssa = false;
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tirx::transform::PassContext &ctx) -> PrimFunc {
    // CPPMEGA: TileLang lowering may leave free-standing T.Buffer aliases
    // without a lexical DeclBuffer/AllocBuffer.  The TileLang-aware helper
    // keeps Apache's vector-type rewrite but permits those aliases in the same
    // way `tl.StorageRewrite` does, scoped only to the TileLang Metal builder.
    return tl::PointerValueTypeRewrite(std::move(f), true);
  };
  mod = tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.TileLangMetalPointerValueTypeRewrite",
      {})(std::move(mod));

  std::ostringstream source_maker;
  // CPPMEGA: apache's new MetalModuleCreateWithFallback expects
  // `ffi::Map<String, Bytes>` for smap and `ffi::Map<String, String>` for
  // source.
  ffi::Map<ffi::String, ffi::Bytes> smap;
  const auto fmetal_compile =
      tvm::ffi::Function::GetGlobal("tvm_callback_metal_compile");
  std::string fmt = fmetal_compile ? "metallib" : "metal";

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto global_symbol =
        kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenTileLangMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenTileLangMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    if (fmetal_compile) {
      fsource = (*fmetal_compile)(fsource, target).cast<std::string>();
    }
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  ffi::Map<ffi::String, ffi::String> source;
  source.Set("metal", source_maker.str());
  return target::MetalModuleCreateWithFallback(
      std::move(smap), ffi::String(fmt), ExtractFuncInfo(mod),
      std::move(source));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_metal", BuildTileLangMetal);
}
} // namespace codegen
} // namespace tvm
