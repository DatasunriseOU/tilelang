/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// CPPMEGA fix-B2 (idea712): RAII scope helper for Z3Prover::EnterConstraint.
//
// Background: every Z3-using transform on this branch (loop_vectorize.cc
// alignment proof, predicate_fusion.cc well-definedness probe, and
// — once ported from `z3-stack` — the negative-stride probe) follows
// the same pattern:
//
//     std::vector<std::function<void()>> recoverers;
//     for (...) recoverers.push_back(z3.EnterConstraint(bound));
//     ...; bool ok = z3.CanProve(goal); ...
//     for (auto it = recoverers.rbegin(); it != recoverers.rend(); ++it)
//       (*it)();
//
// The cleanup runs in normal control flow but if `EnterConstraint`
// throws partway through the push loop, OR if any other code between
// `push` and the manual `pop` throws, the recoverers leak: the Z3
// solver scope stack stays unbalanced and subsequent queries on the
// same Analyzer see stale assertions.
//
// `ConstraintScope` packages the "push on construction / pop on
// destruction" idiom in a small RAII type. It is move-only so a
// vector<ConstraintScope> can hold the pushed scopes; destruction order
// is reverse-iteration, matching the `solver.pop()` requirement.
//
// Usage:
//
//     std::vector<ConstraintScope> scopes;
//     for (...) scopes.emplace_back(z3, bound);
//     bool ok = z3.CanProve(goal);
//     // scopes destruct in reverse order at end of block.
//
// The RAII unwind happens before any catch block higher up the stack,
// so the Z3 solver state is always rebalanced even on exception.
//
// References:
//   * `src/transform/loop_partition.cc:148` (the analyzer-side analog of
//     this pattern that runs RAII via `tvm::With<arith::ConstraintContext>`).
//   * `src/transform/vendored/z3_prover.cc:194-222` (the implementation
//     of `EnterConstraint`'s push/pop function that we wrap here).

#ifndef TILELANG_VENDORED_Z3_CONSTRAINT_SCOPE_H_
#define TILELANG_VENDORED_Z3_CONSTRAINT_SCOPE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <utility>

#include "z3_prover.h"

namespace tilelang {
namespace tlz3 {

class ConstraintScope {
 public:
  ConstraintScope() = default;

  // Push a constraint into `prover`. Records the recover lambda so that
  // destruction (or explicit `Pop`) calls it exactly once.
  ConstraintScope(::tilelang::tlz3::Z3Prover& prover,
                  const ::tvm::PrimExpr& constraint, bool is_assume = false) {
    recover_ = prover.EnterConstraint(constraint, is_assume);
  }

  ConstraintScope(ConstraintScope&& other) noexcept
      : recover_(std::move(other.recover_)) {
    other.recover_ = {};
  }

  ConstraintScope& operator=(ConstraintScope&& other) noexcept {
    if (this != &other) {
      Pop();
      recover_ = std::move(other.recover_);
      other.recover_ = {};
    }
    return *this;
  }

  ConstraintScope(const ConstraintScope&) = delete;
  ConstraintScope& operator=(const ConstraintScope&) = delete;

  ~ConstraintScope() { Pop(); }

  // Idempotent explicit pop. Safe to call multiple times.
  void Pop() noexcept {
    if (recover_) {
      // Recovery lambda calls `solver.pop()` and erases solver-side
      // memo entries. It must not throw — but if Z3 ever propagates an
      // exception out of `solver.pop()`, swallow it: a leaked frame is
      // strictly better than std::terminate from a destructor.
      try {
        recover_();
      } catch (...) {
        // intentional swallow — destructor noexcept contract
      }
      recover_ = {};
    }
  }

 private:
  std::function<void()> recover_;
};

// CPPMEGA fix-B4 (idea712): dtype-aware BV bounds.
//
// Every Z3 well-definedness / alignment query in this branch bit-bounds
// each free Var to a small range so the solver can decide quantifier-
// free affine queries quickly. The previous bound was a flat
// `[0, 2^31)` regardless of dtype, which is unsound for signed int32
// vars that may legitimately be negative (sub-buffer offsets,
// `LegalizeNegativeIndex`-rewritten indices, iter-domain biases). It
// was also overly conservative for unsigned types and 64-bit types.
//
// `BVBoundsForDtype` returns `(lo, hi)` such that the expression
// `(var >= lo) && (var < hi)` is the tightest sound BV bound for the
// given dtype.
//
//   * unsigned uintN, N < 63   →  [0, 1<<N)
//   * unsigned uint63/uint64   →  [0, INT64_MAX]   (clamped to int64)
//   * signed   intN, N <= 63   →  [-(1<<(N-1)), 1<<(N-1))
//   * signed   int64           →  [INT64_MIN, INT64_MAX]
//
// Returned as `int64_t` because the entire constraint stack works in
// int64 PrimExpr constants. Float/handle/bool dtypes return {0, 0} —
// callers must check `dt.is_int() || dt.is_uint()` separately.
inline std::pair<int64_t, int64_t> BVBoundsForDtype(
    const ::tvm::runtime::DataType& dt) {
  if (!dt.is_int() && !dt.is_uint()) {
    return {0, 0};
  }
  int bits = dt.bits();
  if (dt.is_uint()) {
    if (bits >= 63) {
      return {0, std::numeric_limits<int64_t>::max()};
    }
    return {0, int64_t(1) << bits};
  }
  // signed
  if (bits >= 64) {
    return {std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max()};
  }
  return {-(int64_t(1) << (bits - 1)), int64_t(1) << (bits - 1)};
}

}  // namespace tlz3
}  // namespace tilelang

#endif  // TILELANG_VENDORED_Z3_CONSTRAINT_SCOPE_H_
