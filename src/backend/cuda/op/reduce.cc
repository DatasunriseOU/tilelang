/*!
 * \file tl/backend/cuda/op/reduce.cc
 * \brief CUDA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace cuda {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsCuda(target);
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace cuda

namespace {

bool MatchCudaReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

// CPPMEGA TODO(merge): this backend-specific cuda.Reduce registration is now
// superseded by the `default.Reduce` impl in src/op/reduce.cc, whose
// MatchDefaultReduceTarget (= !TargetIsMetal) already covers CUDA/CuTeDSL and
// whose MakeDefault{Scalar,Batch}AllReduce reproduce exactly this backend's
// SM90 NamedBarrier / pre-SM90 SyncThreadsBarrier logic against the NEW
// ReduceImpl contract ({name, match_target, int priority,
// make_scalar_allreduce, make_batch_allreduce, needs/size workspace fns,
// append_args}). The legacy cuda::Reduce here still uses the OLD 3-field {name,
// match_target, Lower(CRTP)} shape, which no longer compiles. Disable the
// redundant registration; the CUDA reduce path is fully served by
// default.Reduce. (The cuda::Reduce struct is retained above for reference /
// future CUDA-specific specialization.)
#if 0
bool RegisterCudaReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cuda.Reduce",
      MatchCudaReduceTarget,
      cuda::Reduce::Lower,
  });
  return true;
}

const bool cuda_reduce_registered = RegisterCudaReduce();
#else
[[maybe_unused]] static auto _cuda_reduce_match_unused = &MatchCudaReduceTarget;
#endif

} // namespace

} // namespace tl
} // namespace tvm
