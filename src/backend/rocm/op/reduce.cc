/*!
 * \file tl/backend/rocm/op/reduce.cc
 * \brief ROCm implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

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

bool MatchROCmReduceTarget(Target target) { return TargetIsRocm(target); }

// CPPMEGA TODO(merge): superseded by default.Reduce (src/op/reduce.cc), whose
// MakeDefaultBatchAllReduce has an explicit TargetIsRocm branch reproducing
// this backend's logic against the new ReduceImpl contract. The legacy
// rocm::Reduce uses the old {name, match_target, Lower(CRTP)} struct shape and
// no longer compiles. Disable the redundant registration; ROCm reduce is fully
// served by default.Reduce. See src/backend/cuda/op/reduce.cc for full context.
#if 0
bool RegisterROCmReduce() {
  RegisterReduceImpl(ReduceImpl{
      "rocm.Reduce",
      MatchROCmReduceTarget,
      rocm::Reduce::Lower,
  });
  return true;
}

const bool rocm_reduce_registered = RegisterROCmReduce();
#else
[[maybe_unused]] static auto _rocm_reduce_match_unused = &MatchROCmReduceTarget;
#endif

} // namespace

} // namespace tl
} // namespace tvm
