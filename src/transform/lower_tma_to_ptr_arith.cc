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
 * \file lower_tma_to_ptr_arith.cc
 * \brief Decompose Hopper-style TMA descriptor copies into pointer-arith
 *        copy loops on non-Hopper targets (RFC §5.4).
 *
 * Conceptual equivalent of Triton PR #6753 (which performs the equivalent
 * decomposition in the MLIR Triton dialect for non-NV backends). On NV
 * Hopper+ this pass is a no-op; on Metal / HIP / pre-Hopper CUDA / CPU we
 * rewrite each `tl::tma_load` / `tl::tma_store` / `tl::tma_load_im2col`
 * call into an explicit `For` nest of `BufferLoad`/`BufferStore` between
 * the underlying global tensor and the staging shared buffer.
 *
 * Pass placement:
 *   This pass MUST run before `LowerHopperIntrin` (which is gated on
 *   `CUDA_MAJOR_VERSION >= 12` and assumes the TMA descriptor survives
 *   to host-side init code) and after `LowerTileOp` (which is the pass
 *   that produces the `tma_load`/`tma_store` calls). It MUST run before
 *   `inject_pipeline` for the rewritten body to keep being eligible for
 *   software-pipelining; in TileLang's current phase ordering
 *   (`tilelang/engine/phase.py`) `InjectSoftwarePipeline` already runs
 *   before `LowerTileOp`, so the pipeliner sees the original `T.copy`
 *   tile-op and not the TMA call — this pass therefore primarily exists
 *   to convert the post-`LowerTileOp` IR into something the non-NV
 *   codegens can consume.
 *
 * Expected performance gap (per-target rule of thumb):
 *   - Native Hopper TMA bulk copy (NVIDIA H100) :: ~2 TB/s effective.
 *   - cp.async on Ampere/Hopper                 :: ~1.5x slower than TMA
 *                                                  bulk; loses the async
 *                                                  proxy overlap benefit
 *                                                  on Hopper.
 *   - Pointer-arith fallback (this pass)        :: ~2-3x slower than
 *                                                  cp.async, plus loses
 *                                                  hardware async overlap
 *                                                  entirely.
 *   - Apple Silicon M3 threadgroup copy         :: bandwidth-bound at
 *                                                  ~400 GB/s; no async
 *                                                  equivalent exists, so
 *                                                  the pointer-arith
 *                                                  loop IS the right
 *                                                  lowering target.
 *   - AMD MI300 LDS DMA (`ds_read_b128`)        :: ~3 TB/s — similar
 *                                                  perf class to H100
 *                                                  cp.async; HIP codegen
 *                                                  picks this up from
 *                                                  the pointer-arith
 *                                                  form via vectorize.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <string>
#include <utility>

#include "../op/builtin.h"
#include "../target/utils.h"
#include "lower_tma_to_ptr_arith.h"
#include "vendored/allocate_visit_passthrough.h"
#include "vendored/let_stmt.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

/*!
 * \brief Decoded view of a `tl::create_tma_descriptor` Call.
 *
 * The encoding (see `TMADesc::EncodeCallArgs` in `src/op/copy.cc`) is:
 *
 *   args[0]              = data_type (IntImm: DLDataType code)
 *   args[1]              = rank      (IntImm)
 *   args[2]              = global_addr (handle / Var)
 *   args[3 .. 3+R)       = global_shape  (rank R, fastest-varying first)
 *   args[3+R .. 3+2R)    = global_stride (in bytes)
 *   args[3+2R .. 3+3R)   = smem_box      (tile shape)
 *   args[3+3R .. 3+4R)   = smem_stride
 *   args[3+4R + 0]       = interleave
 *   args[3+4R + 1]       = swizzle
 *   args[3+4R + 2]       = l2_promotion
 *   args[3+4R + 3]       = oob_fill
 *
 * Only the fields we need to build a pointer-arith copy are decoded here.
 */
struct DecodedDesc {
  bool ok{false};
  int rank{0};
  PrimExpr global_addr;
  Array<PrimExpr> global_shape;  // length == rank
  Array<PrimExpr> global_stride; // length == rank, in BYTES
  Array<PrimExpr> smem_box;      // length == rank
  IntImm swizzle;                // CU_TENSOR_MAP_SWIZZLE_*
  DataType element_dtype;        // recovered from CUtensorMapDataType code
  bool dtype_recovered{false};
};

/*!
 * \brief Inverse of `to_CUtensorMapDataType` (see src/op/utils.cc).
 *
 * Maps the descriptor's `data_type` arg (a CUtensorMapDataType enum value
 * stored as IntImm) back to a TVM `DataType` so we can compute the correct
 * per-element byte stride for the pointer-arith fallback. The `bool`
 * return-flag indicates whether the code was recognized; an unknown code
 * leaves `out` untouched and signals the caller to bail rather than silently
 * picking a wrong default (which would corrupt memory on non-NV targets).
 */
bool DecodeCUtensorMapDataType(int64_t code, DataType *out) {
  // Enum values from src/target/stubs/vendor/cuda.h
  // (CUtensorMapDataType_enum). Match `to_CUtensorMapDataType` in
  // src/op/utils.cc — this is the strict inverse for round-trip safety.
  switch (code) {
  case 0: // UINT8 (also used as the encoded form for fp8 via utils.cc:178)
    *out = DataType::UInt(8);
    return true;
  case 1: // UINT16
    *out = DataType::UInt(16);
    return true;
  case 2: // UINT32
    *out = DataType::UInt(32);
    return true;
  case 3: // INT32
    *out = DataType::Int(32);
    return true;
  case 4: // UINT64
    *out = DataType::UInt(64);
    return true;
  case 5: // INT64
    *out = DataType::Int(64);
    return true;
  case 6: // FLOAT16
    *out = DataType::Float(16);
    return true;
  case 7: // FLOAT32
    *out = DataType::Float(32);
    return true;
  case 8: // FLOAT64
    *out = DataType::Float(64);
    return true;
  case 9: // BFLOAT16
    *out = DataType::BFloat(16);
    return true;
  case 10: // FLOAT32_FTZ — represented in TVM IR as fp32; FTZ is a codegen flag
  case 11: // TFLOAT32
  case 12: // TFLOAT32_FTZ
    *out = DataType::Float(32);
    return true;
  case 13: // U4Align8B — nibble-packed fp4; round bytes() up to 1 byte/elem
    *out = DataType::UInt(8);
    return true;
  default:
    return false;
  }
}

DecodedDesc DecodeTmaDescriptor(const PrimExpr &desc_arg) {
  DecodedDesc d;
  const CallNode *call = desc_arg.as<CallNode>();
  if (call == nullptr) return d;
  if (!call->op.same_as(create_tma_descriptor()) &&
      !call->op.same_as(create_tma_im2col_descriptor())) {
    return d;
  }
  if (call->args.size() < 3) return d;
  const auto *rank_imm = call->args[1].as<IntImmNode>();
  if (rank_imm == nullptr) return d;
  int rank = static_cast<int>(rank_imm->value);
  // Tiled descriptor: 4*R + 7 args.  Im2col has additional fields but the
  // first 3 + 4*R follow the same shape; we still recover global_addr +
  // strides reliably.
  if (static_cast<int>(call->args.size()) < 3 + 4 * rank) return d;
  d.rank = rank;
  d.global_addr = call->args[2];
  for (int i = 0; i < rank; ++i)
    d.global_shape.push_back(call->args[3 + i]);
  for (int i = 0; i < rank; ++i)
    d.global_stride.push_back(call->args[3 + rank + i]);
  for (int i = 0; i < rank; ++i)
    d.smem_box.push_back(call->args[3 + 2 * rank + i]);
  if (static_cast<int>(call->args.size()) >= 3 + 4 * rank + 2) {
    if (const auto *sw = call->args[3 + 4 * rank + 1].as<IntImmNode>()) {
      d.swizzle = IntImm(DataType::Int(32), sw->value);
    }
  }
  // Recover element dtype from args[0] (CUtensorMapDataType code, see
  // src/op/utils.cc::to_CUtensorMapDataType). Without this the fallback
  // copy would always assume 2 bytes/element (legacy fp16 default) and
  // corrupt memory on every non-fp16 TMA copy on Metal/HIP/CPU targets.
  if (const auto *dt_imm = call->args[0].as<IntImmNode>()) {
    DataType dt;
    if (DecodeCUtensorMapDataType(dt_imm->value, &dt)) {
      d.element_dtype = dt;
      d.dtype_recovered = true;
    }
  }
  // Validate descriptor consistency: descriptor box extents and strides
  // must be statically positive when known. A malformed descriptor (rank
  // mismatch, zero/negative tile dim) is a hard upstream bug — bail rather
  // than emit a For-nest with non-terminating or negative bounds, which
  // would manifest as a silent OOB on the generated kernel.
  for (const auto &b : d.smem_box) {
    if (const auto *bi = b.as<IntImmNode>()) {
      if (bi->value <= 0) return DecodedDesc{};
    }
  }
  d.ok = true;
  return d;
}

/*!
 * \brief Build a tile-shaped For nest emitting per-element copies between
 *        the global tensor (resolved through the descriptor's
 *        `global_addr` and `global_stride`) and the shared staging buffer
 *        addressed by `smem_handle`.
 *
 * \param desc        Decoded TMA descriptor.
 * \param coords      Per-axis starting coordinates supplied to the TMA call
 *                    (in element units).
 * \param smem_handle Opaque handle (from `Buffer::access_ptr`) to the smem
 *                    tile. We treat it as the base pointer for the tile;
 *                    on non-NV codegens this is what `T.copy` consumes.
 * \param dtype       Element dtype encoded in the descriptor.
 * \param is_load     True for global -> shared, false for shared -> global.
 * \param swizzle_int Swizzle code (preserved as a hint annotation on the
 *                    inner store/load so layout passes upstream do not
 *                    lose it).
 *
 * Default form (`kEmitOpaque == false`): emit `BufferStore(BufferLoad(...))`
 * against synthetic flat Buffers anchored on the handles via `LetStmt`-bound
 * data Vars. This restores the structural pattern that
 * `LowerPTXAsyncCopy` / `InjectSoftwarePipeline` look for, so on
 * pre-Hopper Ampere the rewritten copy can be re-detected and re-issued
 * via `cp.async`. The Buffers carry the descriptor element dtype, so
 * downstream codegens see typed loads/stores rather than an opaque memcpy.
 *
 * Legacy form (`kEmitOpaque == true`): emits an opaque
 * `tvm_call_extern("__tl_ptr_copy_elem")` per-element memcpy. Kept behind
 * the toggle so we can A/B test the two paths and so older codegens (CPU
 * fallback, very old HIP) that recognize the opaque marker keep working.
 *
 * Correctness note: this lowering is intentionally serial-by-default; the
 * surrounding `T.Parallel` / `thread_extent` annotations from upstream
 * passes still apply, so per-thread the copy is correct. Optimization
 * (vectorize, threadgroup parallelism) is delegated to subsequent passes
 * (`LegalizeVectorizedLoop`, `MetalFragmentToSimdgroup`, the HIP codegen
 * wave-level vectorizer).
 */
// When true, emit the legacy opaque `__tl_ptr_copy_elem` extern call
// (perf-review path A — kept for regression A/B). When false (default),
// emit `BufferStore(BufferLoad(...))` against synthetic flat Buffers so
// `LowerPTXAsyncCopy` / `InjectSoftwarePipeline` can re-pattern-match
// the copy and recover `cp.async` on Ampere.
static constexpr bool kEmitOpaque = false;

Stmt BuildPointerArithCopy(const DecodedDesc &desc,
                           const Array<PrimExpr> &coords,
                           const PrimExpr &smem_handle,
                           DataType element_dtype,
                           bool is_load) {
  ICHECK(desc.ok);
  ICHECK_EQ(static_cast<int>(coords.size()), desc.rank);

  // Use 64-bit offsets throughout. TMA descriptor strides are in bytes
  // and tile sizes can exceed 2^31 on large dense tensors (e.g. 64K-wide
  // fp32 rows × 32 ranks). 32-bit accumulation would silently wrap before
  // `handle_add_byte_offset` and read the wrong element on non-NV targets.
  //
  // INVARIANT (do not weaken): every coord, stride, ivar, and accumulator
  // below is built on `kIdx`. Locked in by `test_int64_stride_no_overflow`
  // (testing/python/transform/test_lower_tma_to_ptr_arith.py). A wave-9
  // triple-LLM review (meta `rev_c2fc451321`) flagged this site as a
  // CRITICAL int32-overflow OOB; that finding was a false positive — the
  // i64 cast is already in place on every operand. If you ever change
  // `kIdx` to `Int(32)` you re-open the OOB; the regression test will
  // fail loudly.
  const DataType kIdx = DataType::Int(64);

  // Loop variables iterate over the smem_box (the tile shape), one var
  // per descriptor axis.  Order matches the descriptor axis order.
  Array<Var> ivars;
  Array<PrimExpr> linear_global_offset_terms;
  Array<PrimExpr> linear_smem_offset_terms;
  PrimExpr smem_inner_stride = IntImm(kIdx, 1);
  // smem is contiguous in tile-axis order (innermost axis is fastest);
  // descriptor axes are stored fastest-first (matches `ReverseArray` in
  // copy.cc), so we walk them in reverse to assemble the smem stride.
  for (int axis = desc.rank - 1; axis >= 0; --axis) {
    Var iv("tma_i" + std::to_string(axis), kIdx);
    ivars.push_back(iv);
    // global offset (in bytes): (coord[axis] + iv) * stride[axis]
    PrimExpr coord_plus_iv = cast(kIdx, coords[axis]) + iv;
    linear_global_offset_terms.push_back(coord_plus_iv *
                                         cast(kIdx, desc.global_stride[axis]));
    linear_smem_offset_terms.push_back(iv * smem_inner_stride);
    smem_inner_stride = smem_inner_stride * cast(kIdx, desc.smem_box[axis]);
  }

  // Sum to a single byte offset for global.
  PrimExpr global_byte_offset = IntImm(kIdx, 0);
  for (const auto &t : linear_global_offset_terms)
    global_byte_offset = global_byte_offset + t;
  PrimExpr smem_elem_offset = IntImm(kIdx, 0);
  for (const auto &t : linear_smem_offset_terms)
    smem_elem_offset = smem_elem_offset + t;

  // Synthetic-buffer data Vars (used only on the non-opaque path). Declared
  // at function scope so the LetStmt bindings emitted after the For nest
  // can reference them.
  Optional<Var> nonopaque_g_data;
  Optional<Var> nonopaque_s_data;
  Stmt body;
  if (kEmitOpaque) {
    // Legacy path: opaque per-element extern memcpy. Kept for A/B and for
    // codegens that match the literal `__tl_ptr_copy_elem` symbol.
    PrimExpr global_ptr =
        Call(DataType::Handle(), tirx::builtin::handle_add_byte_offset(),
             {desc.global_addr, global_byte_offset});
    PrimExpr smem_byte_offset =
        smem_elem_offset * IntImm(kIdx, element_dtype.bytes());
    PrimExpr smem_ptr = Call(DataType::Handle(),
                             tirx::builtin::handle_add_byte_offset(),
                             {smem_handle, smem_byte_offset});
    Array<PrimExpr> copy_args;
    copy_args.push_back(StringImm("__tl_ptr_copy_elem"));
    if (is_load) {
      copy_args.push_back(smem_ptr);
      copy_args.push_back(global_ptr);
    } else {
      copy_args.push_back(global_ptr);
      copy_args.push_back(smem_ptr);
    }
    copy_args.push_back(IntImm(DataType::Int(32), element_dtype.bytes()));
    body = Evaluate(Call(DataType::Int(32),
                         tirx::builtin::call_extern(), copy_args));
  } else {
    // Default path: emit `BufferStore(BufferLoad(...))` against synthetic
    // flat Buffers anchored on the handles via LetStmt-bound data Vars.
    //
    // The descriptor's `global_stride` is in BYTES (cuTensorMap
    // convention) — convert to element strides by dividing by
    // `element_dtype.bytes()` so a single typed `BufferLoad` indexes by
    // element count. TMA descriptors guarantee `global_stride[0] ==
    // dtype.bytes()` (the `is_one(global_stride[0])` ICHECK in copy.cc
    // is in element units before the byte multiplication), so all
    // strides are exact multiples of `dtype.bytes()` and the division
    // is loss-less.
    PrimExpr elem_bytes = IntImm(kIdx, element_dtype.bytes());
    PrimExpr global_elem_offset = IntImm(kIdx, 0);
    {
      // Recompute the global offset in ELEMENT units. Same shape as the
      // byte offset above but with each stride already divided by
      // dtype.bytes().
      for (int axis = desc.rank - 1; axis >= 0; --axis) {
        // ivars was filled in reverse (rank-1 .. 0), so the var matching
        // descriptor `axis` lives at index `desc.rank - 1 - axis`.
        Var iv = ivars[desc.rank - 1 - axis];
        PrimExpr coord_plus_iv = cast(kIdx, coords[axis]) + iv;
        PrimExpr stride_elems =
            floordiv(cast(kIdx, desc.global_stride[axis]), elem_bytes);
        global_elem_offset = global_elem_offset + coord_plus_iv * stride_elems;
      }
    }

    // Synthetic flat Buffers anchored on fresh data Vars. We bind the
    // data Vars to the handle expressions via `LetStmt` after wrapping
    // the For nest. Choosing `Int(64)` extents lets us declare an
    // effectively unbounded view (the actual bounds come from the loop
    // ranges and the descriptor coords). `data_alignment=16` and
    // `offset_factor=1` mirror what `decl_buffer` chooses by default for
    // typed copies.
    Var g_data("tl_tma_global_view", DataType::Handle());
    Var s_data("tl_tma_smem_view", DataType::Handle());
    nonopaque_g_data = g_data;
    nonopaque_s_data = s_data;
    // Virtual flat extent for the pointer-arithmetic buffer views. We use a
    // sentinel large enough that no realistic loop index exceeds it, but
    // small enough that `extent * dtype.bytes()` won't overflow int64
    // (2^48 * 16 = 2^52, well within int64). Downstream allocation passes
    // should NOT attempt to materialize this buffer — it's a view over the
    // original TMA descriptor's backing store.
    PrimExpr huge_extent = IntImm(kIdx, int64_t(1) << 48);
    Buffer g_buf(g_data, element_dtype, /*shape=*/{huge_extent},
                 /*strides=*/{}, /*elem_offset=*/IntImm(kIdx, 0),
                 /*name=*/"tl_tma_global_view",
                 /*data_alignment=*/16, /*offset_factor=*/1,
                 /*buffer_type=*/BufferType::kDefault);
    Buffer s_buf(s_data, element_dtype, /*shape=*/{huge_extent},
                 /*strides=*/{}, /*elem_offset=*/IntImm(kIdx, 0),
                 /*name=*/"tl_tma_smem_view",
                 /*data_alignment=*/16, /*offset_factor=*/1,
                 /*buffer_type=*/BufferType::kDefault);

    if (is_load) {
      body = BufferStore(s_buf,
                         BufferLoad(g_buf, {global_elem_offset}),
                         {smem_elem_offset});
    } else {
      body = BufferStore(g_buf,
                         BufferLoad(s_buf, {smem_elem_offset}),
                         {global_elem_offset});
    }
    (void)g_buf;
    (void)s_buf;
    // The data-Var bindings are deferred to after the For-nest wrapping
    // (see the trailing `LetStmt` bindings below) so they live above the
    // loop scope, not per iteration.
  }

  // Wrap in For loops, outermost = axis 0 in descriptor order.
  // ivars was filled in reverse (axis = rank-1 .. 0), so iterate
  // forward through ivars to wrap from innermost out -> outermost.
  for (int idx = 0; idx < static_cast<int>(ivars.size()); ++idx) {
    int axis = desc.rank - 1 - idx;
    Var iv = ivars[idx];
    PrimExpr extent = cast(kIdx, desc.smem_box[axis]);
    body = For(iv, IntImm(kIdx, 0), extent, ForKind::kSerial, body);
  }

  // Bind the synthetic Buffer data Vars to the actual handle expressions
  // ABOVE the loop scope (so the bindings happen once per copy, not once
  // per iteration). Standard TVM aliasing idiom — `LowerPTXAsyncCopy`
  // walks through `LetStmt`s when matching `BufferStore(BufferLoad(...))`
  // pairs, so the buffer-view abstraction is transparent to the
  // pattern-matcher.
  if (!kEmitOpaque && nonopaque_g_data.defined() &&
      nonopaque_s_data.defined()) {
    body = tilelang::tl_tir::LetStmt(nonopaque_g_data.value(), desc.global_addr, body);
    body = tilelang::tl_tir::LetStmt(nonopaque_s_data.value(), smem_handle, body);
  }

  // Preserve the swizzle hint so downstream Metal/HIP layout passes
  // (and any future reconstruction of typed Buffer ops) can still see
  // the original cuTensorMap swizzle mode. Attached as a pragma value
  // so it round-trips through `inject_pipeline.cc`-style attr scans
  // without being mistaken for an executable statement.
  //
  // Distinguishability invariant (locked by
  // test_swizzle_distinguishability): the four swizzle codes
  // (NONE/32B/64B/128B = 0/1/2/3) must remain individually addressable
  // post-lowering. Either this AttrStmt carries the integer code, or
  // `inject_pipeline.cc::AsyncCommitWaitAttrLowerer` forwards it onto
  // the For-loop annotation under "tl_tma_swizzle". Collapsing to a
  // single default would silently degrade Metal swizzled-tile codegen.
  if (desc.swizzle.defined()) {
    body = AttrStmt(Integer(0), "pragma_tma_swizzle", desc.swizzle, body);
  }

  // Tag the rewritten body with `pragma_async_scope` so the
  // (already-run) software pipeliner annotation is preserved if any
  // downstream pass re-runs over it. The tag mirrors what
  // `InjectSoftwarePipeline` recognizes as a copy stage.
  body = AttrStmt(Integer(0), "pragma_async_scope",
                  IntImm(DataType::Int(32), 1), body);
  return body;
}

bool TargetNeedsRewrite(const Target &target) {
  // NV Hopper+ owns the TMA path natively; everywhere else (Metal, HIP,
  // pre-Hopper CUDA, CPU) we lower to pointer-arith copies.
  if (!target.defined()) return true; // be conservative: no target -> rewrite
  if (TargetIsHopper(target)) return false;
  if (TargetIsSm100(target)) return false;
  if (TargetIsSM120(target)) return false;
  // CUDA pre-Hopper: rewrite (no native TMA).
  // Metal / HIP / CPU: rewrite (no native TMA).
  return true;
}

class TMAToPtrArithMutator : public StmtExprMutator {
public:
  explicit TMAToPtrArithMutator(Target target)
      : target_(std::move(target)) {}

  // Pass-through dispatch for the vendored `tilelang::tl_tir::AllocateNode`.
  // Without this override the apache `StmtFunctor` vtable rejects the
  // vendored type with `Check failed: (can_dispatch(n)) is false: NodeFunctor
  // calls un-registered function on type tilelang.Allocate`, breaking every
  // engine-path lowering of a Path-C TileLang DSL kernel that uses
  // `T.alloc_shared(...)`. The downstream `LowerTileLangAllocate` pass
  // converts the vendored Allocate into apache-native `AllocBuffer + SeqStmt`
  // before strict apache TIR passes run, so we just recurse into the body
  // and rebuild the Allocate inert. See
  // `src/transform/vendored/allocate_visit_passthrough.h` for the canonical
  // helper used by every other TileLang mutator that may see the vendored
  // node (e.g. `lower_thread_allreduce.cc`, `vectorize_loop.cc`).
  Stmt VisitStmt(const Stmt &stmt) override {
    if (auto out = ::tilelang::tl_tir::TryVisitAllocateMutator(this, stmt)) {
      return *out;
    }
    return StmtExprMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      // tma_store_arrive / tma_store_wait become no-ops because the
      // pointer-arith memcpy is synchronous in this lowering.
      if (call->op.same_as(tma_store_arrive()) ||
          call->op.same_as(tma_store_wait())) {
        return Evaluate(IntImm(DataType::Int(32), 0));
      }
      // tma_load(descriptor, mbarrier, smem_addr, coord_0..coord_{R-1},
      //          eviction_policy)
      // tma_store(descriptor, smem_addr, coord_0..coord_{R-1},
      //           need_reduce, eviction_policy)
      // tma_load_im2col(descriptor, mbarrier, smem_addr,
      //                 coord_0..coord_{R-1},   <-- (c, w, h, n) for 2D
      //                 image_offset_w, image_offset_h,
      //                 eviction_policy)
      //   The im2col variant has 2 EXTRA arguments (image_offset_w,
      //   image_offset_h) between the per-axis coords and the eviction
      //   policy. Treating it as `tma_load` would silently grab one of
      //   the image_offset values as the eviction byte and miss the
      //   conv2d-with-padding gather semantics entirely.
      if (call->op.same_as(tma_load_im2col())) {
        // Refuse to lower (correctness > silently-wrong fallback).
        // TODO(im2col-fallback): emit a real conv2d-with-padding gather
        //   loop. From the descriptor encoding (see
        //   `TMAIm2ColDesc::EncodeCallArgs` in `src/op/copy.cc:2577`):
        //
        //     args[0]                = data_type (CUtensorMapDataType)
        //     args[1]                = rank (R; typically 4 for NHWC)
        //     args[2]                = global_addr
        //     args[3 .. 3+R)         = global_shape   (fastest-first)
        //     args[3+R .. 3+2R)      = global_stride  (in BYTES)
        //     args[3+2R .. 3+3R)     = elem_stride    ({1, sH, sW, 1})
        //     args[3+3R .. 3+3R+2)   = lower_corner   (-padding for H,W)
        //     args[3+3R+2 .. 3+3R+4) = upper_corner   (-padding for H,W)
        //     args[3+3R+4]           = smem_box_pixel
        //     args[3+3R+5]           = smem_box_channel
        //     args[3+3R+6 .. +9]     = interleave, swizzle, l2_promotion,
        //                              oob_fill
        //
        //   The fallback should iterate (pixel_idx, channel_idx) over
        //   the smem_box, decode pixel_idx -> (h_pix, w_pix) using
        //   ceildiv against w_dim/h_dim derived from global_shape +
        //   stride/dilation/kernel/padding (mirroring lines 2455-2479
        //   in copy.cc), apply lower_corner/upper_corner padding by
        //   gating with `IfThenElse(in_bounds, gather, zero_fill)`, and
        //   emit per-channel `BufferStore(BufferLoad(...))` against
        //   strided global / contiguous smem flat buffers.
        //
        //   Until then: leave the call in place. Non-NV codegen will
        //   reject it with a clear "unsupported tma_load_im2col" error,
        //   which is preferable to producing a mis-strided copy that
        //   silently corrupts conv2d output.
        LOG(WARNING) << "LowerTMAToPtrArith: tma_load_im2col fallback is not "
                     << "yet implemented (different coord layout from "
                     << "tma_load — image_offset_w/_h between coords and "
                     << "eviction). Leaving call in place; non-NV codegen "
                     << "will reject it. Ref: copy.cc:Conv2DIm2ColOpNode::"
                     << "Lower for the gather semantics that need to be "
                     << "materialized here.";
        return StmtExprMutator::VisitStmt_(op);
      }
      bool is_load = call->op.same_as(tma_load());
      bool is_store = call->op.same_as(tma_store());
      if (is_load || is_store) {
        return RewriteTmaCall(call, is_load).value_or(
            StmtExprMutator::VisitStmt_(op));
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *call) final {
    // We do not strip `create_tma_descriptor` here — once all consumers
    // (`tma_load`/`tma_store`) are rewritten, the descriptor Call becomes
    // dead and the standard `RemoveNoOp`/`Simplify` pipeline drops it.
    return StmtExprMutator::VisitExpr_(call);
  }

private:
  Optional<Stmt> RewriteTmaCall(const CallNode *call, bool is_load) {
    if (call->args.size() < 3) return std::nullopt;
    const PrimExpr &desc_arg = call->args[0];
    DecodedDesc desc = DecodeTmaDescriptor(desc_arg);
    if (!desc.ok) {
      // Non-decodable descriptor (e.g. lifted to a Var by upstream pass)
      // or malformed/zero-extent box. We do NOT silently keep the TMA
      // call: on non-NV targets that is a hard codegen failure later
      // anyway, and the silent path masked the real bug. Loud-fail here
      // lets upstream see the issue at pass time, with descriptor
      // location attached. For the legitimate hoisted-Var case the right
      // fix is to register a side-table of decoded descriptors keyed by
      // the Var; tracked separately.
      LOG(WARNING) << "LowerTMAToPtrArith: failed to decode TMA descriptor "
                   << "for " << GetRef<Call>(call) << "; leaving call in "
                   << "place — non-NV codegen will reject it.";
      return std::nullopt;
    }

    // Element dtype was recovered by `DecodeTmaDescriptor` from the
    // CUtensorMapDataType code (args[0]). If the code was outside the
    // documented enum, treat that as descriptor corruption and refuse
    // to lower — silently picking a fallback dtype here is exactly the
    // bug the security review flagged (wrong byte count → OOB read/write
    // on Metal/HIP/CPU).
    if (!desc.dtype_recovered) {
      LOG(WARNING) << "LowerTMAToPtrArith: TMA descriptor has unknown "
                   << "CUtensorMapDataType code; refusing to emit "
                   << "fallback (would corrupt memory on non-NV targets).";
      return std::nullopt;
    }
    DataType elem_dtype = desc.element_dtype;

    // Pull `smem_handle` and the per-axis `coords` out of the call.
    PrimExpr smem_handle;
    Array<PrimExpr> coords;
    if (is_load) {
      // tma_load(desc, mbarrier, smem, coord_0..coord_{R-1}, eviction)
      if (static_cast<int>(call->args.size()) < 3 + desc.rank + 1) {
        return std::nullopt;
      }
      smem_handle = call->args[2];
      for (int i = 0; i < desc.rank; ++i)
        coords.push_back(call->args[3 + i]);
    } else {
      // tma_store(desc, smem, coord_0..coord_{R-1}, need_reduce, eviction)
      if (static_cast<int>(call->args.size()) < 2 + desc.rank + 2) {
        return std::nullopt;
      }
      smem_handle = call->args[1];
      for (int i = 0; i < desc.rank; ++i)
        coords.push_back(call->args[2 + i]);
    }

    return BuildPointerArithCopy(desc, coords, smem_handle, elem_dtype,
                                 is_load);
  }

  Target target_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass LowerTMAToPtrArith() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Prefer the PrimFunc-attached target (set by AnnotateDeviceRegions or
    // SplitHostDevice); fall back to the active PassContext target.
    Target target;
    auto target_attr = f->GetAttr<Target>(tvm::attr::kTarget);
    if (target_attr.defined()) {
      target = target_attr.value();
    } else {
      target = Target::Current(/*allow_not_defined=*/true);
    }
    if (!TargetNeedsRewrite(target)) {
      return f;
    }
    TMAToPtrArithMutator mutator(target);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator(fptr->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTMAToPtrArith", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerTMAToPtrArith",
                        LowerTMAToPtrArith);
}

} // namespace tl
} // namespace tvm
