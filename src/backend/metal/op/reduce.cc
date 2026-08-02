/*!
 * \file tl/backend/metal/op/reduce.cc
 * \brief Metal implementation for tl.reduce AllReduce lowering decisions.
 */

#include "op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

int WorkspaceStride(const ReductionPlan &plan) {
  return plan.same_simdgroup_metal_fast_path_safe ? 0 : plan.reducing_threads;
}

} // namespace

struct Reduce {
  static std::string MakeScalarAllReduce(const ReduceOpNode &op,
                                         const LowerArgs &T,
                                         const ReductionPlan &plan) {
    (void)T;
    std::stringstream ss;
    ss << "tl::AllReduce<" << op.MakeCodegenReducer() << ", "
       << plan.reducing_threads << ", " << plan.scale << ", "
       << plan.thread_offset << ", tl::SyncThreadsBarrier, 1, "
       << WorkspaceStride(plan) << ">::run";
    return ss.str();
  }

  static std::string MakeBatchAllReduce(const ReduceOpNode &op,
                                        const LowerArgs &T,
                                        const ReductionPlan &plan, int batch) {
    (void)T;
    std::stringstream ss;
    ss << "tl::AllReduce<" << op.MakeCodegenReducer() << ", "
       << plan.reducing_threads << ", " << plan.scale << ", "
       << plan.thread_offset << ", tl::SyncThreadsBarrier, " << batch << ", "
       << WorkspaceStride(plan) << ">::run_batch";
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
    return WorkspaceStride(plan);
  }

  static bool NeedsBatchWorkspace(const LowerArgs &T, const ReductionPlan &plan,
                                  int batch) {
    (void)T;
    (void)batch;
    return !plan.same_simdgroup_metal_fast_path_safe;
  }

  static int BatchWorkspaceSize(const LowerArgs &T, const ReductionPlan &plan,
                                int batch) {
    (void)T;
    return WorkspaceStride(plan) * batch;
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

bool MatchMetalReduceTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalReduce() {
  RegisterReduceImpl(ReduceImpl{
      "metal.Reduce",
      MatchMetalReduceTarget,
      100,
      metal::Reduce::MakeScalarAllReduce,
      metal::Reduce::MakeBatchAllReduce,
      metal::Reduce::NeedsScalarWorkspace,
      metal::Reduce::ScalarWorkspaceSize,
      metal::Reduce::NeedsBatchWorkspace,
      metal::Reduce::BatchWorkspaceSize,
      metal::Reduce::AppendArgs,
      metal::Reduce::AppendArgs,
  });
  return true;
}

const bool metal_reduce_registered = RegisterMetalReduce();

} // namespace

} // namespace tl
} // namespace tvm
