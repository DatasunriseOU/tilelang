// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/op/finalize_reducer.h
 * \brief Define finalize_reducer operator.
 */

#ifndef TVM_TL_OP_FINALIZE_REDUCER_H_
#define TVM_TL_OP_FINALIZE_REDUCER_H_

#include "../transform/layout_reducer.h"
#include "./operator.h"
#include "./reduce.h"

#include <cstdint>
#include <string>

/**
 * Get the Op singleton for the public FinalizeReducerOp handle.
 *
 * @return A reference to the Op describing FinalizeReducer.
 */
namespace tvm {
namespace tl {

using namespace tirx;

class FinalizeReducerOpNode : public TileOperatorNode {
public:
  tirx::Buffer reducer;
  ReducerOpType op;
  // Batch size for batched AllReduce (1 = scalar path, same as T.reduce
  // default).
  int batch{1};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.FinalizeReducerOp",
                                    FinalizeReducerOpNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FinalizeReducerOpNode>()
        .def_ro("reducer", &FinalizeReducerOpNode::reducer)
        .def_ro("op", &FinalizeReducerOpNode::op)
        .def_ro("batch", &FinalizeReducerOpNode::batch);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  static const Op &Get();
  TileOperator Clone() const override;
};

using FinalizeReducerTargetPredicate = bool (*)(Target target);

struct FinalizeReducerImpl {
  const char *name;
  FinalizeReducerTargetPredicate match_target;
  int priority;

  std::string (*make_scalar_allreduce)(const FinalizeReducerOpNode &op,
                                       const LowerArgs &T,
                                       const ReductionPlan &plan,
                                       const std::string &op_str);
  std::string (*make_batch_allreduce)(const FinalizeReducerOpNode &op,
                                      const LowerArgs &T,
                                      const ReductionPlan &plan,
                                      const std::string &op_str,
                                      int64_t batch);

  bool (*needs_scalar_workspace)(const LowerArgs &T,
                                 const ReductionPlan &plan);
  int (*scalar_workspace_size)(const LowerArgs &T,
                               const ReductionPlan &plan);
  bool (*needs_batch_workspace)(const LowerArgs &T,
                                const ReductionPlan &plan, int64_t batch);
  int (*batch_workspace_size)(const LowerArgs &T,
                              const ReductionPlan &plan, int64_t batch);

  void (*append_scalar_args)(Array<PrimExpr> *args, const LowerArgs &T,
                             bool need_workspace, const PrimExpr &workspace);
  void (*append_batch_args)(Array<PrimExpr> *args, const LowerArgs &T,
                            bool need_workspace, const PrimExpr &workspace);
};

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl);

class FinalizeReducerOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FinalizeReducerOp, TileOperator,
                                             FinalizeReducerOpNode);
  TVM_DLL FinalizeReducerOp(
      Array<PrimExpr> args,
      Map<String, ffi::ObjectRef> annotations = Map<String, ffi::ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_FINALIZE_REDUCER_H_
