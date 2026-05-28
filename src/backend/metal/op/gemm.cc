/*!
 * \file tl/backend/metal/op/gemm.cc
 * \brief Metal implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "target/utils.h"

#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

constexpr const char *kMetalSIMDGroup = "metal.simdgroup";
constexpr const char *kMetalCooperativeTensor = "metal.cooperative_tensor";

std::pair<int, int>
ComputeSIMDGroupWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                              int num_warps, String gemm_inst) {
  int m_warp = 1;
  int n_warp = 1;
  int kMPerWarp;
  int kNPerWarp;
  if (gemm_inst == kMetalCooperativeTensor) {
    // M5 mpp::tensor_ops::matmul2d minimum micro tile is 16x32.
    kMPerWarp = 16;
    kNPerWarp = 32;
  } else {
    // metal.simdgroup retains the 8x8 micro tile.
    kMPerWarp = 8;
    kNPerWarp = 8;
  }

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (policy.isFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0) {
        n_warp = 1;
      }
    }
  } else if (policy.isFullCol()) {
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
  } else if (policy.isSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; ++m) {
      int n = num_warps / m;
      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      if (m_per_warp < 1 || n_per_warp < 1) {
        continue;
      }
      if (m * n != num_warps) {
        continue;
      }

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }
    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

// Returns true when the shape (M, N, K) and warp count permit the cooperative
// tensor 16x32x16 micro-tile partition.  See PR tile-ai/tilelang#2252.
bool CanUseCooperativeTensor(const GemmWarpPolicyNode &policy, int M, int N,
                             int K, int num_warps) {
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 32;
  if (M % kMPerWarp != 0 || N % kNPerWarp != 0 || K % 16 != 0) {
    return false;
  }
  int max_m = M / kMPerWarp;
  int max_n = N / kNPerWarp;
  if (policy.isFullRow()) {
    int m_warp = num_warps;
    if (M % (m_warp * kMPerWarp) != 0) {
      m_warp = max_m;
    }
    return m_warp > 0 && num_warps % m_warp == 0 && num_warps / m_warp <= max_n;
  }
  if (policy.isFullCol()) {
    int n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      n_warp = max_n;
    }
    return n_warp > 0 && num_warps % n_warp == 0 && num_warps / n_warp <= max_m;
  }
  if (policy.isSquare()) {
    for (int m = 1; m <= std::min(num_warps, max_m); ++m) {
      if (num_warps % m == 0 && num_warps / m <= max_n) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (!TargetIsMetal(target)) {
      ICHECK(0) << "metal::Gemm::SelectInst called for a non-Metal target "
                << target->str();
    }
    // Fragment / simdgroup-scoped accumulators always use the legacy
    // simdgroup path; the cooperative tensor path supports only shared C
    // (see PR tile-ai/tilelang#2252).
    if (op.c_.scope() == "local.fragment" ||
        op.c_.scope() == "metal.simdgroup") {
      return kMetalSIMDGroup;
    }
    int num_warps = block_size / TargetGetWarpSize(target);
    if (CanUseCooperativeTensor(*op.policy_.operator->(), op.m_, op.n_, op.k_,
                                num_warps)) {
      return kMetalCooperativeTensor;
    }
    return kMetalSIMDGroup;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    ICHECK(gemm_inst == kMetalSIMDGroup ||
           gemm_inst == kMetalCooperativeTensor)
        << "Unsupported Metal GEMM instruction: " << gemm_inst;
    int num_warps = block_size / TargetGetWarpSize(target);
    return ComputeSIMDGroupWarpPartition(policy, M, N, num_warps, gemm_inst);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kMetalSIMDGroup) {
      return "metal_simdgroup";
    }
    if (gemm_inst == kMetalCooperativeTensor) {
      return "metal_cooperative_tensor";
    }
    return "unknown";
  }
};

} // namespace metal

namespace {

bool MatchMetalGemmTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalGemm() {
  RegisterGemmImpl(GemmImpl{
      "metal.Gemm",
      MatchMetalGemmTarget,
      metal::Gemm::SelectInst,
      metal::Gemm::ComputeWarpPartition,
      metal::Gemm::ReuseExistingSharedLayout,
      metal::Gemm::InstructionKind,
  });
  return true;
}

const bool metal_gemm_registered = RegisterMetalGemm();

} // namespace

} // namespace tl
} // namespace tvm
