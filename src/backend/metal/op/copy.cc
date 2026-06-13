/*!
 * \file tl/backend/metal/op/copy.cc
 * \brief Metal implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

bool CheckCooperativeTensorCopy(const CopyNode &op, Target target) {
  if (!TargetIsMetal(target) || !IsCooperativeTensorBuffer(op.src)) {
    return false;
  }
  return IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst);
}

Stmt LowerCooperativeTensorCopy(const CopyNode &op, const LowerArgs &T,
                                arith::Analyzer *analyzer) {
  // Lower a copy from a metal.cooperative_tensor source into a 2D shared/
  // global destination by emitting tl::cooperative_tensor_store calls.  See
  // PR tile-ai/tilelang#2252 for the algorithm details.
  (void)analyzer;
  ICHECK(IsCooperativeTensorBuffer(op.src));
  int total_elements = 1;
  for (auto s : op.src->shape) {
    auto imm = s.as<IntImmNode>();
    ICHECK(imm) << "cooperative_tensor buffer must have constant shape";
    total_elements *= imm->value;
  }

  constexpr int kTileSize = 16;
  constexpr int kTileElems = kTileSize * kTileSize;
  ICHECK(total_elements % kTileElems == 0)
      << "cooperative_tensor buffer size must be multiple of " << kTileElems
      << ", got " << total_elements;

  ICHECK(op.dst_range.size() == 2)
      << "Expected 2D destination for cooperative_tensor store";
  PrimExpr dst_row_base = op.dst_range[0]->min;
  PrimExpr dst_col_base = op.dst_range[1]->min;
  PrimExpr dst_stride = op.dst->shape[op.dst->shape.size() - 1];

  int warp_size = TargetGetWarpSize(T.target);
  const auto *block_size_imm = T.thread_bounds->extent.as<IntImmNode>();
  ICHECK(block_size_imm)
      << "cooperative_tensor copy requires constant thread bounds";
  int block_size = block_size_imm->value;
  int num_warps = block_size / warp_size;
  PrimExpr warp_id = FloorDiv(T.thread_var, warp_size);

  const auto *m_imm = op.src_range[0]->extent.as<IntImmNode>();
  const auto *n_imm = op.src_range[1]->extent.as<IntImmNode>();
  ICHECK(m_imm && n_imm)
      << "cooperative_tensor copy requires constant extents";
  int M = m_imm->value;
  int N = n_imm->value;

  int kMPerWarp = kTileSize;
  int kNPerWarp = kTileSize * 2;
  int m_warp = 1, n_warp = num_warps;
  int max_m = M / kMPerWarp;
  int max_n = N / kNPerWarp;

  bool is_gmem_kernel = false;
  if (IsGlobalBuffer(op.dst)) {
    is_gmem_kernel = true;
    for (auto &kv : T.layout_map) {
      if (IsSharedBuffer(kv.first)) {
        is_gmem_kernel = false;
        break;
      }
    }
  }
  if (is_gmem_kernel) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      int max_n_warps = N / kNPerWarp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0) {
        m_warp = 1;
      }
    }
  } else {
    float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
    float best_score = std::numeric_limits<float>::max();
    for (int m = 1; m <= std::min(num_warps, max_m); ++m) {
      if (num_warps % m != 0) {
        continue;
      }
      int n = num_warps / m;
      if (n > max_n) {
        continue;
      }
      float m_per = static_cast<float>(M) / (m * kMPerWarp);
      float n_per = static_cast<float>(N) / (n * kNPerWarp);
      float score = std::abs(m_per / n_per - ideal);
      if (score < best_score) {
        best_score = score;
        m_warp = m;
        n_warp = n;
      }
    }
  }

  int elems_per_thread = total_elements / (num_warps * warp_size);
  int warp_M = M / m_warp;
  int warp_N = N / n_warp;
  int warp_tiles = elems_per_thread / (kTileSize * kTileSize / warp_size);

  int kTileN = warp_N;
  int kTileM = kTileSize;
  if (warp_tiles > 0 && warp_M > kTileSize) {
    kTileN = warp_N;
    kTileM = kTileSize;
  }
  if (kTileN > warp_N) {
    kTileN = warp_N;
  }

  int warp_row_tiles = warp_M / kTileM;
  int warp_col_tiles = warp_N / kTileN;

  ICHECK(warp_row_tiles > 0 && warp_col_tiles > 0)
      << "Cannot partition " << M << "x" << N << " matrix across " << m_warp
      << "x" << n_warp << " warps";

  int tile_elems_per_thread = kTileM * kTileN / warp_size;
  ICHECK(warp_row_tiles * warp_col_tiles * tile_elems_per_thread ==
         elems_per_thread)
      << "Tile partition inconsistent with buffer size: " << warp_row_tiles
      << "x" << warp_col_tiles << " tiles of " << kTileM << "x" << kTileN
      << " = " << warp_row_tiles * warp_col_tiles * tile_elems_per_thread
      << " elems/thread, expected " << elems_per_thread;

  PrimExpr warp_m = FloorMod(warp_id, m_warp);
  PrimExpr warp_n = FloorDiv(warp_id, m_warp);

  Array<Stmt> stmts;
  for (int i = 0; i < warp_row_tiles; i++) {
    for (int j = 0; j < warp_col_tiles; j++) {
      int tile_idx = i * warp_col_tiles + j;
      PrimExpr row = dst_row_base + warp_m * warp_M + i * kTileM;
      PrimExpr col = dst_col_base + warp_n * warp_N + j * kTileN;
      PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                          {BufferLoad(op.dst, {row, col})});
      int kMMAK = kTileSize;
      stmts.push_back(Evaluate(Call(
          DataType::Handle(), tl::cooperative_tensor_store(),
          {op.src->data, IntImm(DataType::Int(32), tile_idx), ptr, dst_stride,
           IntImm(DataType::Int(32), kTileM), IntImm(DataType::Int(32), kTileN),
           Cast(DataType::Bool(), IntImm(DataType::Int(32), 0)),
           IntImm(DataType::Int(32), kTileM), IntImm(DataType::Int(32), kTileN),
           IntImm(DataType::Int(32), kMMAK), IntImm(DataType::Int(32), 2)})));
    }
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return SeqStmt(stmts);
}

bool CheckSIMDGroupCopy(const CopyNode &op, Target target) {
  if (!TargetIsMetal(target) || !IsSIMDGroupBuffer(op.src)) {
    return false;
  }
  if (!IsSharedBuffer(op.dst) && !IsGlobalBuffer(op.dst)) {
    return false;
  }
  if (op.src->dtype != op.dst->dtype) {
    // fp16-OUTPUT (and any narrowing-from-accumulator) simdgroup store.  Metal's
    // ``simdgroup_store`` requires the destination element type to MATCH the
    // matrix element type (``simdgroup_matrix<float,8,8>`` only stores to
    // ``device/threadgroup float*``), so we cannot store a float accumulator
    // directly into a half buffer.  The previous behaviour bailed to
    // LowerNormalCopy, which emitted an ILLEGAL ``(half4)C_l[...]`` C-style cast
    // of the simdgroup matrix (rejected by the Metal compiler).  Instead, route
    // float-accumulator -> {float16,bfloat16,float} destinations through the
    // simdgroup path: stage into a threadgroup float tile via the legal
    // dtype-matched ``simdgroup_store`` and then per-element cast+store into the
    // destination (see LowerSIMDGroupCopy).  Only float-family conversions are
    // supported here; anything else must still go through LowerNormalCopy.
    if (!(op.src->dtype.is_float() && op.dst->dtype.is_float())) {
      return false;
    }
  }
  if (op.src_range.size() != 2 || op.dst_range.size() != 2 ||
      op.dst->shape.size() != 2) {
    return false;
  }

  int total_elements = 1;
  for (auto extent : op.src->shape) {
    auto imm = extent.as<IntImmNode>();
    if (!imm) {
      return false;
    }
    total_elements *= imm->value;
  }
  if (total_elements % 64 != 0) {
    return false;
  }

  for (int i = 0; i < 2; ++i) {
    auto src_shape = op.src->shape[i].as<IntImmNode>();
    auto src_min = op.src_range[i]->min.as<IntImmNode>();
    auto src_extent = op.src_range[i]->extent.as<IntImmNode>();
    auto dst_extent = op.dst_range[i]->extent.as<IntImmNode>();
    if (!src_shape || !src_min || src_min->value != 0 || !src_extent ||
        !dst_extent || src_extent->value != src_shape->value ||
        src_extent->value != dst_extent->value || src_extent->value % 8 != 0) {
      return false;
    }
  }
  return true;
}

Stmt LowerSIMDGroupCopy(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  ICHECK(IsSIMDGroupBuffer(op.src));
  int total_elements = 1;
  for (auto s : op.src->shape) {
    auto imm = s.as<IntImmNode>();
    ICHECK(imm) << "simdgroup buffer must have constant shape";
    total_elements *= imm->value;
  }
  ICHECK(total_elements % 64 == 0)
      << "simdgroup buffer size must be multiple of 64 (8x8), got "
      << total_elements;

  ICHECK(op.dst_range.size() == 2)
      << "Expected 2D destination for simdgroup store";
  PrimExpr dst_row_base = op.dst_range[0]->min;
  PrimExpr dst_col_base = op.dst_range[1]->min;
  ICHECK_EQ(op.dst->shape.size(), 2U)
      << "simdgroup store currently supports 2D destination buffers";
  Array<PrimExpr> dst_strides = op.dst->strides;
  if (dst_strides.empty()) {
    PrimExpr stride = 1;
    dst_strides.resize(op.dst->shape.size());
    for (int i = static_cast<int>(op.dst->shape.size()) - 1; i >= 0; --i) {
      dst_strides.Set(i, stride);
      stride *= op.dst->shape[i];
    }
  }
  if (dst_strides.size() != op.dst->shape.size()) {
    return LowerNormalCopy(op, T, analyzer);
  }
  if (!analyzer->CanProveEqual(dst_strides[1], 1)) {
    return LowerNormalCopy(op, T, analyzer);
  }
  PrimExpr dst_stride = dst_strides[0];

  int warp_size = TargetGetWarpSize(T.target);
  auto block_extent = T.thread_bounds->extent.as<IntImmNode>();
  if (!block_extent || warp_size <= 0 || block_extent->value % warp_size != 0) {
    return LowerNormalCopy(op, T, analyzer);
  }
  int block_size = block_extent->value;
  int num_warps = block_size / warp_size;
  if (num_warps <= 0) {
    return LowerNormalCopy(op, T, analyzer);
  }
  PrimExpr relative_thread = T.thread_var - T.thread_bounds->min;
  PrimExpr warp_id = FloorDiv(relative_thread, warp_size);

  auto M_imm = op.src_range[0]->extent.as<IntImmNode>();
  auto N_imm = op.src_range[1]->extent.as<IntImmNode>();
  if (!M_imm || !N_imm) {
    return LowerNormalCopy(op, T, analyzer);
  }
  int M = M_imm->value;
  int N = N_imm->value;

  int kMPerWarp = 8;
  int kNPerWarp = 8;
  int m_warp = 1, n_warp = num_warps;
  int max_m = M / kMPerWarp;
  int max_n = N / kNPerWarp;
  if (max_m <= 0 || max_n <= 0) {
    return LowerNormalCopy(op, T, analyzer);
  }
  float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
  float best_score = std::numeric_limits<float>::max();
  for (int m = 1; m <= std::min(num_warps, max_m); ++m) {
    if (num_warps % m != 0) {
      continue;
    }
    int n = num_warps / m;
    if (n > max_n) {
      continue;
    }
    if (M % (m * kMPerWarp) != 0 || N % (n * kNPerWarp) != 0) {
      continue;
    }
    float m_per = static_cast<float>(M) / (m * kMPerWarp);
    float n_per = static_cast<float>(N) / (n * kNPerWarp);
    float score = std::abs(m_per / n_per - ideal);
    if (score < best_score) {
      best_score = score;
      m_warp = m;
      n_warp = n;
    }
  }

  if (best_score == std::numeric_limits<float>::max() || M < m_warp * 8 ||
      N < n_warp * 8) {
    return LowerNormalCopy(op, T, analyzer);
  }
  int warp_row_tiles = M / m_warp / 8;
  int warp_col_tiles = N / n_warp / 8;
  if (warp_row_tiles <= 0 || warp_col_tiles <= 0 ||
      warp_row_tiles * warp_col_tiles * 64 > total_elements) {
    return LowerNormalCopy(op, T, analyzer);
  }

  PrimExpr warp_m = FloorMod(warp_id, m_warp);
  PrimExpr warp_n = FloorDiv(warp_id, m_warp);

  const bool needs_cast = op.src->dtype != op.dst->dtype;

  if (!needs_cast) {
    // --------------------------------------------------------------------
    // Matched dtype (e.g. float accumulator -> float destination): store the
    // simdgroup fragments DIRECTLY into the destination buffer.  This is the
    // original, hardware-verified fast path.
    // --------------------------------------------------------------------
    Array<Stmt> stmts;
    for (int i = 0; i < warp_row_tiles; i++) {
      for (int j = 0; j < warp_col_tiles; j++) {
        int tile_idx = i * warp_col_tiles + j;
        PrimExpr row = dst_row_base + warp_m * (warp_row_tiles * 8) + i * 8;
        PrimExpr col = dst_col_base + warp_n * (warp_col_tiles * 8) + j * 8;
        PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                            {BufferLoad(op.dst, {row, col})});
        stmts.push_back(Evaluate(Call(
            DataType::Handle(), builtin::simdgroup_store(),
            {op.src->data, IntImm(DataType::Int(32), tile_idx), ptr, dst_stride,
             IntImm(DataType::Int(32), 8), IntImm(DataType::Int(32), 8),
             Cast(DataType::Bool(), IntImm(DataType::Int(32), 0))})));
      }
    }
    if (stmts.size() == 1) {
      return stmts[0];
    }
    return SeqStmt(stmts);
  }

  // ----------------------------------------------------------------------
  // Differing dtype (e.g. float accumulator -> fp16/bf16 destination).  Metal's
  // ``simdgroup_store`` requires the destination element type to match the
  // matrix element type, so a float matrix cannot be stored straight into a
  // half buffer.  Stage every 8x8 fragment into a threadgroup float tile of
  // shape [M, N] (the legal dtype-matched simdgroup_store), synchronise, then
  // have ALL threads cooperatively cast each float element down to the
  // destination dtype and write it into the (possibly strided) destination.
  // ----------------------------------------------------------------------
  // Allocate the staging tile through the plumbed workspace callback so the
  // storage-allocation passes know about it (a raw decl_buffer injected here is
  // NOT picked up by the shared-memory allocator).  ``AddWorkspace`` returns a
  // ``shared.dyn`` access pointer; we wrap a flat Buffer view over its data Var
  // so we can index it with BufferLoad / simdgroup_store.
  ICHECK(T.AddWorkspace != nullptr)
      << "Metal fp16-output simdgroup store needs T.AddWorkspace for staging";
  PrimExpr ws_ptr = T.AddWorkspace(M * N, op.src->dtype);
  Var stage_var = GetVarFromAccessPtr(ws_ptr);
  Buffer stage =
      decl_buffer({IntImm(DataType::Int(32), M * N)}, op.src->dtype,
                  "C_simd_stage", "shared.dyn");
  // Rebind the staging buffer's data to the workspace-owned variable so codegen
  // resolves it to the allocated threadgroup storage.
  {
    auto n = make_object<BufferNode>(*stage.get());
    n->data = stage_var;
    stage = Buffer(n);
  }
  PrimExpr stage_stride = IntImm(DataType::Int(32), N);

  Array<Stmt> body;
  // Phase 1: simdgroup_store each fragment into the threadgroup staging tile.
  for (int i = 0; i < warp_row_tiles; i++) {
    for (int j = 0; j < warp_col_tiles; j++) {
      int tile_idx = i * warp_col_tiles + j;
      PrimExpr srow = warp_m * (warp_row_tiles * 8) + i * 8;
      PrimExpr scol = warp_n * (warp_col_tiles * 8) + j * 8;
      PrimExpr soff = srow * stage_stride + scol;
      PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                          {BufferLoad(stage, {soff})});
      // First arg is the SIMD-group SOURCE matrix (op.src, the C accumulator);
      // ``ptr`` addresses the threadgroup staging tile we store the fragment to.
      body.push_back(Evaluate(Call(
          DataType::Handle(), builtin::simdgroup_store(),
          {op.src->data, IntImm(DataType::Int(32), tile_idx), ptr, stage_stride,
           IntImm(DataType::Int(32), 8), IntImm(DataType::Int(32), 8),
           Cast(DataType::Bool(), IntImm(DataType::Int(32), 0))})));
    }
  }
  // Barrier: the threadgroup tile is now fully populated.
  body.push_back(Evaluate(
      Call(DataType::Int(32), builtin::tvm_storage_sync(), {StringImm("shared")})));

  // Phase 2: every thread casts a strided subset of the [M, N] tile down to the
  // destination dtype and stores it.  Threads of the block stride over all M*N
  // elements so the work is balanced and independent of the warp layout.
  PrimExpr tid = relative_thread;
  Var lv("simd_cast_i", DataType::Int(32));
  PrimExpr elem = tid + lv * IntImm(DataType::Int(32), block_size);
  PrimExpr e_row = FloorDiv(elem, IntImm(DataType::Int(32), N));
  PrimExpr e_col = FloorMod(elem, IntImm(DataType::Int(32), N));
  PrimExpr src_val = BufferLoad(stage, {e_row * stage_stride + e_col});
  PrimExpr dst_val = Cast(op.dst->dtype, src_val);
  Stmt store = BufferStore(
      op.dst, dst_val, {dst_row_base + e_row, dst_col_base + e_col});
  // Loop ``elem`` from ``tid`` to M*N stepping by block_size.  ``trip`` is the
  // exact number of strided iterations this thread performs.
  int trip = (M * N - 1 - 0) >= 0 ? ((M * N) + block_size - 1) / block_size : 0;
  Stmt loop_body =
      IfThenElse(elem < IntImm(DataType::Int(32), M * N), store);
  body.push_back(For(lv, IntImm(DataType::Int(32), 0),
                     IntImm(DataType::Int(32), trip), ForKind::kSerial,
                     loop_body));
  // Barrier: the staging tile may be reused by a subsequent pipeline stage.
  body.push_back(Evaluate(
      Call(DataType::Int(32), builtin::tvm_storage_sync(), {StringImm("shared")})));

  // The staging storage is owned by the workspace allocator (AddWorkspace), so
  // we must NOT wrap it in our own AllocBuffer here.
  return SeqStmt(body);
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    return op.InferSIMTLayout(T, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (CheckSIMDGroupCopy(op, T.target)) {
      return LowerSIMDGroupCopy(op, T, analyzer);
    }
    if (CheckCooperativeTensorCopy(op, T.target)) {
      return LowerCooperativeTensorCopy(op, T, analyzer);
    }
    return LowerNormalCopy(op, T, analyzer);
  }
};

} // namespace metal

namespace {

bool MatchMetalCopyTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalCopy() {
  RegisterCopyImpl(CopyImpl{
      "metal.Copy",
      MatchMetalCopyTarget,
      100,
      metal::Copy::InferLayout,
      metal::Copy::Lower,
  });
  return true;
}

const bool metal_copy_registered = RegisterMetalCopy();

} // namespace

} // namespace tl
} // namespace tvm
