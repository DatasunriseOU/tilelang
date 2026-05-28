/*!
 * \file tl/backend/rocm/op/finalize_reducer.cc
 * \brief ROCm implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target) { return 64; }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset, PrimExpr,
                                        int batch, int workspace_stride,
                                        Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", " << batch << ", "
       << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset, PrimExpr,
                                         Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }
};

} // namespace rocm

namespace {

bool MatchROCmFinalizeReducerTarget(Target target) {
  return TargetIsRocm(target);
}

// CPPMEGA TODO(merge): same FinalizeReducerImpl shape mismatch as cuda;
// see src/backend/cuda/op/finalize_reducer.cc for the full explanation.
// Registration is suppressed until the rocm backend is ported to the new
// (make_scalar_allreduce, make_batch_allreduce, …) contract.
#if 0
bool RegisterROCmFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "rocm.FinalizeReducer",
      MatchROCmFinalizeReducerTarget,
      rocm::FinalizeReducer::Lower,
  });
  return true;
}

const bool rocm_finalize_reducer_registered = RegisterROCmFinalizeReducer();
#else
[[maybe_unused]] static auto _rocm_finalize_match_unused =
    &MatchROCmFinalizeReducerTarget;
[[maybe_unused]] static auto _rocm_finalize_lower_unused =
    &rocm::FinalizeReducer::Lower;
#endif

} // namespace

} // namespace tl
} // namespace tvm
