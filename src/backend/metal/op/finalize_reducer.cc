/*!
 * \file tl/backend/metal/op/finalize_reducer.cc
 * \brief Metal implementation for finalize_reducer AllReduce decisions.
 */

#include "op/finalize_reducer.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

int ScalarWorkspaceStride(const ReductionPlan &plan) {
  return plan.same_simdgroup_metal_fast_path_safe ? 0 : plan.reducing_threads;
}

int BatchWorkspaceStride(const LowerArgs &T, const ReductionPlan &plan) {
  return plan.same_simdgroup_metal_fast_path_safe
             ? 0
             : static_cast<int>(*as_const_int(T.thread_bounds->extent));
}

} // namespace

struct FinalizeReducer {
  static std::string MakeScalarAllReduce(const FinalizeReducerOpNode &op,
                                         const LowerArgs &T,
                                         const ReductionPlan &plan,
                                         const std::string &op_str) {
    (void)op;
    (void)T;
    std::stringstream ss;
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset
       << ", tl::SyncThreadsBarrier, 1, " << ScalarWorkspaceStride(plan)
       << ">::run";
    return ss.str();
  }

  static std::string MakeBatchAllReduce(const FinalizeReducerOpNode &op,
                                        const LowerArgs &T,
                                        const ReductionPlan &plan,
                                        const std::string &op_str,
                                        int64_t batch) {
    (void)op;
    std::stringstream ss;
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset
       << ", tl::SyncThreadsBarrier, " << batch << ", "
       << BatchWorkspaceStride(T, plan) << ">::run_batch";
    return ss.str();
  }

  static bool NeedsScalarWorkspace(const LowerArgs &T,
                                   const ReductionPlan &plan) {
    (void)T;
    return !plan.same_simdgroup_metal_fast_path_safe;
  }

  static int ScalarWorkspaceSize(const LowerArgs &T,
                                 const ReductionPlan &plan) {
    (void)T;
    return ScalarWorkspaceStride(plan);
  }

  static bool NeedsBatchWorkspace(const LowerArgs &T, const ReductionPlan &plan,
                                  int64_t batch) {
    (void)T;
    (void)batch;
    return !plan.same_simdgroup_metal_fast_path_safe;
  }

  static int BatchWorkspaceSize(const LowerArgs &T, const ReductionPlan &plan,
                                int64_t batch) {
    return BatchWorkspaceStride(T, plan) * static_cast<int>(batch);
  }

  static void AppendArgs(Array<PrimExpr> *args, const LowerArgs &T,
                         bool need_workspace, const PrimExpr &workspace) {
    args->push_back(T.thread_var);
    if (need_workspace) {
      args->push_back(workspace);
    }
  }
};

} // namespace metal

namespace {

bool MatchMetalFinalizeReducerTarget(Target target) {
  return TargetIsMetal(target);
}

bool RegisterMetalFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "metal.FinalizeReducer",
      MatchMetalFinalizeReducerTarget,
      100,
      metal::FinalizeReducer::MakeScalarAllReduce,
      metal::FinalizeReducer::MakeBatchAllReduce,
      metal::FinalizeReducer::NeedsScalarWorkspace,
      metal::FinalizeReducer::ScalarWorkspaceSize,
      metal::FinalizeReducer::NeedsBatchWorkspace,
      metal::FinalizeReducer::BatchWorkspaceSize,
      metal::FinalizeReducer::AppendArgs,
      metal::FinalizeReducer::AppendArgs,
  });
  return true;
}

const bool metal_finalize_reducer_registered = RegisterMetalFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
