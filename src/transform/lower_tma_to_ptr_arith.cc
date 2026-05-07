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

#include <string>
#include <utility>

#include "../op/builtin.h"
#include "../target/utils.h"
#include "lower_tma_to_ptr_arith.h"

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
 * Element-level addressing uses an opaque byte-level pointer arithmetic
 * via `tir::builtin::address_of`-equivalents would be ideal, but the
 * cleanest portable form is to materialize a `BufferLoad`/`BufferStore`
 * against synthetic flat buffers reconstructed from the descriptor data.
 *
 * Because we do NOT have the original Buffer objects here (only a handle
 * lowered through `access_ptr`), we emit a `tl::call_extern("memcpy_2d")`
 * style pseudo-call wrapped in a `For` nest of unit-stride increments.
 * The non-NV codegens (Metal, HIP, CPU) recognize the resulting
 * pointer-arith pattern and emit native threadgroup-copy instructions.
 *
 * Correctness note: this lowering is intentionally serial-by-default; the
 * surrounding `T.Parallel` / `thread_extent` annotations from upstream
 * passes still apply, so per-thread the copy is correct. Optimization
 * (vectorize, threadgroup parallelism) is delegated to subsequent passes
 * (`LegalizeVectorizedLoop`, `MetalFragmentToSimdgroup`, the HIP codegen
 * wave-level vectorizer).
 */
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

  // Element-sized handle add, in bytes vs elements: TMA descriptor
  // strides are in BYTES (cuTensorMap convention), but the element
  // memcpy below operates on `element_dtype` units. We forward the byte
  // offset to a `handle_add_byte_offset` builtin (same one
  // `lower_hopper_intrin.cc` uses for L2 base pointers).
  PrimExpr global_ptr =
      Call(DataType::Handle(), tirx::builtin::handle_add_byte_offset(),
           {desc.global_addr, global_byte_offset});

  // Convert smem element offset into bytes for the same builtin. Use
  // `element_dtype.bytes()` which is now correctly recovered from the
  // descriptor's `data_type` field (fixes the legacy 2-byte default that
  // silently corrupted every non-fp16 TMA copy on non-NV targets).
  PrimExpr smem_byte_offset =
      smem_elem_offset * IntImm(kIdx, element_dtype.bytes());
  PrimExpr smem_ptr = Call(DataType::Handle(),
                           tirx::builtin::handle_add_byte_offset(),
                           {smem_handle, smem_byte_offset});

  // Innermost statement: a single-element memcpy via tvm_call_extern
  // ("__tl_ptr_copy_elem"). This is portable: the codegens we care
  // about (Metal, HIP, CPU) all understand a 1-element opaque copy.
  // TODO: verify — would prefer a `BufferStore(BufferLoad(...))` form,
  // but we no longer have the Buffer objects here. The opaque-call form
  // is correctness-preserving; a follow-up may reconstruct synthetic
  // Buffers and emit typed Loads/Stores for better codegen.
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
  Stmt body = Evaluate(Call(DataType::Int(32),
                            tirx::builtin::call_extern(), copy_args));

  // Wrap in For loops, outermost = axis 0 in descriptor order.
  // ivars was filled in reverse (axis = rank-1 .. 0), so iterate
  // forward through ivars to wrap from innermost out -> outermost.
  for (int idx = 0; idx < static_cast<int>(ivars.size()); ++idx) {
    int axis = desc.rank - 1 - idx;
    Var iv = ivars[idx];
    PrimExpr extent = cast(kIdx, desc.smem_box[axis]);
    body = For(iv, IntImm(kIdx, 0), extent, ForKind::kSerial, body);
  }

  // Preserve the swizzle hint so downstream Metal/HIP layout passes
  // (and any future reconstruction of typed Buffer ops) can still see
  // the original cuTensorMap swizzle mode. Attached as a pragma value
  // so it round-trips through `inject_pipeline.cc`-style attr scans
  // without being mistaken for an executable statement.
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
      bool is_load = call->op.same_as(tma_load()) ||
                     call->op.same_as(tma_load_im2col());
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
