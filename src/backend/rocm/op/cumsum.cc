/*!
 * \file tl/backend/rocm/op/cumsum.cc
 * \brief ROCm implementation for tl.cumsum lowering.
 */

#include "backend/common/op/cumsum.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchROCmCumSumTarget(Target target) { return TargetIsRocm(target); }

// CPPMEGA TODO(merge): RegisterCumSumImpl is declared-but-undefined and
// CumSumOpNode::Lower lowers inline without a registry — see
// src/backend/cuda/op/cumsum.cc for the full explanation. Disable the no-op
// registration to resolve the undefined symbol; cumsum works via the inline
// CumSumOpNode::Lower path.
#if 0
bool RegisterROCmCumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "rocm.CumSum",
      MatchROCmCumSumTarget,
      backend::CumSum::Lower,
  });
  return true;
}

const bool rocm_cumsum_registered = RegisterROCmCumSum();
#else
[[maybe_unused]] static auto _rocm_cumsum_match_unused = &MatchROCmCumSumTarget;
#endif

} // namespace

} // namespace tl
} // namespace tvm
