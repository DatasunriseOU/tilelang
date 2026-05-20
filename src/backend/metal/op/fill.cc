/*!
 * \file tl/backend/metal/op/fill.cc
 * \brief Metal implementation for tl.fill lowering.
 */

#include "op/fill.h"
#include <tvm/runtime/logging.h>

#include <tvm/tirx/builtin.h>

#include "op/utils.h"
#include "target/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (IsSIMDGroupBuffer(op.dst)) {
      int region_elements = 1;
      for (auto r : op.region) {
        auto imm = r->extent.as<IntImmNode>();
        ICHECK(imm) << "simdgroup fill region must have constant extents";
        region_elements *= imm->value;
      }
      int total_elements = region_elements;
      ICHECK(total_elements % 64 == 0)
          << "simdgroup buffer size must be multiple of 64 (8x8), got "
          << total_elements;
      int num_matrices = total_elements / 64;
      PrimExpr fill_value = Cast(op.dst->dtype, op.value);
      Array<PrimExpr> strides = op.dst->strides;
      if (strides.empty()) {
        PrimExpr stride = 1;
        strides.resize(op.dst->shape.size());
        for (int i = static_cast<int>(op.dst->shape.size()) - 1; i >= 0; --i) {
          strides.Set(i, stride);
          stride *= op.dst->shape[i];
        }
      }
      ICHECK_EQ(strides.size(), op.dst->shape.size())
          << "simdgroup fill requires complete destination strides";
      PrimExpr element_offset = 0;
      for (size_t i = 0; i < op.region.size(); ++i) {
        element_offset += op.region[i]->min * strides[i];
      }
      PrimExpr matrix_elements = IntImm(element_offset.dtype(), 64);
      ICHECK(
          analyzer->CanProveEqual(FloorMod(element_offset, matrix_elements), 0))
          << "simdgroup fill region must start on an 8x8 matrix boundary";
      PrimExpr matrix_index_base = FloorDiv(element_offset, matrix_elements);
      // Verify that the region truly spans `num_matrices` consecutive 8x8
      // simdgroup matrices. If the symbolic region end does not coincide with
      // matrix_index_base + num_matrices - 1, the indices may overlap or skip
      // matrices; in that case fall back to a dense per-lane emission instead
      // of emitting consecutive simdgroup writes that could be unsound.
      PrimExpr last_element_offset =
          element_offset +
          IntImm(element_offset.dtype(), total_elements - 1);
      PrimExpr last_matrix_index =
          FloorDiv(last_element_offset, matrix_elements);
      PrimExpr expected_last =
          matrix_index_base +
          IntImm(matrix_index_base.dtype(), num_matrices - 1);
      bool consecutive_provable =
          analyzer->CanProveEqual(last_matrix_index, expected_last);
      if (!consecutive_provable) {
        // Fall back: cannot prove the simdgroup matrix span is contiguous, so
        // emit a dense per-lane SIMT fill instead of the simdgroup intrinsic.
        auto init_loop = op.MakeSIMTLoop(analyzer);
        auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
        auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
        return unrolled_loop;
      }
      Array<Stmt> stmts;
      for (int i = 0; i < num_matrices; i++) {
        stmts.push_back(Evaluate(
            Call(DataType::Handle(), builtin::make_filled_simdgroup_matrix(),
                 {op.dst->data, matrix_index_base + IntImm(DataType::Int(32), i),
                  fill_value, IntImm(DataType::Int(32), 8),
                  IntImm(DataType::Int(32), 8)})));
      }
      if (stmts.size() == 1)
        return stmts[0];
      return SeqStmt(stmts);
    }
    if (IsFragmentBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var,
                                       analyzer, par_op->GetLoopLayout());
      auto vectorized_loop = VectorizeLoop(thread_loop, analyzer, T.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);

      if (par_op->GetPredicate(T.thread_var).defined()) {
        return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    if (IsLocalBuffer(op.dst) || IsLocalVarBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    if (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var,
                                       analyzer, par_op->GetLoopLayout());
      auto vectorized_loop = VectorizeLoop(thread_loop, analyzer, T.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
      if (par_op->GetPredicate(T.thread_var).defined()) {
        return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    LOG(FATAL) << "Unsupported scope " << op.dst.scope();
    return Stmt();
  }
};

} // namespace metal

namespace {

bool MatchMetalFillTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalFill() {
  RegisterFillImpl(FillImpl{
      "metal.Fill",
      MatchMetalFillTarget,
      metal::Fill::Lower,
  });
  return true;
}

const bool metal_fill_registered = RegisterMetalFill();

} // namespace

} // namespace tl
} // namespace tvm
