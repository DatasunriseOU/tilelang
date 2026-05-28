/*!
 * \file tl/backend/cuda/op/cumsum.cc
 * \brief CUDA implementation for tl.cumsum lowering.
 */

#include "backend/common/op/cumsum.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchCudaCumSumTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

// CPPMEGA TODO(merge): RegisterCumSumImpl / the CumSumImpl registry were only
// ever DECLARED (src/op/reduce.h:245) — never defined, and CumSumOpNode::Lower
// (src/op/reduce.cc) lowers cumsum INLINE via tl::CumSum1D / tl::CumSum2D
// without consulting any registry. Calling RegisterCumSumImpl here therefore
// referenced an undefined symbol (load-time 'undefined symbol:
// tvm::tl::RegisterCumSumImpl'). Disable the no-op registration; cumsum is
// fully served by the inline CumSumOpNode::Lower path. The backend
// CumSum::Lower helper is retained in backend/common/op/cumsum.h for a future
// registry-based refactor if the abstraction is ever completed.
#if 0
bool RegisterCudaCumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "cuda.CumSum",
      MatchCudaCumSumTarget,
      backend::CumSum::Lower,
  });
  return true;
}

const bool cuda_cumsum_registered = RegisterCudaCumSum();
#else
[[maybe_unused]] static auto _cuda_cumsum_match_unused = &MatchCudaCumSumTarget;
#endif

} // namespace

} // namespace tl
} // namespace tvm
