// CPPMEGA: Real Z3-backed prover, vendored from TileLang fork's
// `arith::Analyzer::z3_prover` extension. Replaces the conservative stub at
// `vendored/z3_prover_stub.h` so the sync rewriter can take partial-sync
// (perf) paths instead of always falling back to full-sync.
//
// In the TileLang fork the prover lived as a member sub-analyzer of
// `tvm::arith::Analyzer`. apache/tvm latest does not have that hook and we
// are not allowed to patch apache. Instead we expose the real prover as
// `tilelang::z3::Z3Prover` and keep the old free-function entry point
// `tvm::arith::Z3Prover(Analyzer&)` (used at every call site) which now
// returns a per-Analyzer cached real instance.
//
// Constraint propagation: in the fork, `Analyzer::Bind` /
// `ConstraintContext::EnterWithScope` forwarded to `z3_prover.{Bind,
// EnterConstraint}`. Apache does not. Per-Analyzer state is therefore
// reseeded lazily from `analyzer.const_int_bound` (which apache populates on
// `Analyzer::Bind(var, range)`). `ConstraintContext`-pushed predicates are
// not visible — the worst this costs is that a few partial-sync queries
// behave like the unconstrained range-only case, which is still strictly
// better than the stub's blanket -1 / `requires_hoist=true`.
#ifndef TILELANG_VENDORED_Z3_PROVER_H_
#define TILELANG_VENDORED_Z3_PROVER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/tirx/var.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "tvm/ffi/optional.h"
#include "tvm/ffi/string.h"

namespace tilelang {
namespace tlz3 {

class Z3ProverImpl;

class Z3Prover {
 public:
  explicit Z3Prover(::tvm::arith::Analyzer* parent);
  ~Z3Prover();

  Z3Prover(const Z3Prover&) = delete;
  Z3Prover& operator=(const Z3Prover&) = delete;

  // Public surface mirrors the TileLang-fork class. Return types use
  // ffi::String upstream; we surface std::string for caller convenience —
  // every call site streams the result into a stringstream, so this is
  // lossless.
  void Bind(const ::tvm::tirx::Var& var, const ::tvm::Range& new_range,
            bool allow_override = false);
  void Bind(const ::tvm::tirx::Var& var, const ::tvm::PrimExpr& expr,
            bool allow_override = false);
  bool CanProve(const ::tvm::PrimExpr& expr);
  std::function<void()> EnterConstraint(const ::tvm::PrimExpr& constraint,
                                        bool is_assume = false);
  std::string GetSMTLIB2(::tvm::ffi::Optional<::tvm::PrimExpr> expr);
  // Convenience overload — accepts std::nullopt directly without forcing the
  // caller to construct an `ffi::Optional<PrimExpr>`.
  std::string GetSMTLIB2(std::nullopt_t);
  std::string GetStats();
  std::string GetModel(const ::tvm::PrimExpr& expr);
  int64_t CountSatisfyingValues(const ::tvm::tirx::Var& var, int64_t max_count,
                                int64_t min_consecutive = 1);

  void SetTimeoutMs(unsigned timeout_ms);
  void SetRLimit(unsigned rlimit);

  // Switch the prover between unbounded Int sort (default, width=0) and a
  // signed BitVector sort of the given width (32 or 64). In BV mode, all
  // arithmetic uses bvadd/bvsub/bvmul/bvsdiv/bvsmod and comparisons use the
  // signed bv predicates (bvslt/bvsle/bvsgt/bvsge); equality stays as
  // operator==/operator!=, which work for either sort. The mode is
  // per-Analyzer-cached-instance and persists across calls until reset by
  // another `SetBitVectorMode(width)`. Switching mode invalidates any
  // previously declared variables: callers must re-`Bind` after a mode
  // change. Default behavior (width=0) is bit-identical to the prior API.
  //
  // PREFER `ScopedBVMode` (defined below) over raw `SetBitVectorMode` calls.
  // Because the prover instance is cached per-Analyzer, a bare
  // `SetBitVectorMode(32)` leaks BV semantics into the next caller that
  // expects Int mode. `ScopedBVMode` records the prior width and restores it
  // on scope exit.
  void SetBitVectorMode(int width);
  int GetBitVectorWidth() const;

 private:
  std::unique_ptr<Z3ProverImpl> impl_;
};

// RAII guard that switches a `Z3Prover` to a target BV width on
// construction and restores the previous width on destruction. Strongly
// preferred over raw `SetBitVectorMode(width)` calls because the prover
// instance is per-Analyzer-cached and the mode persists across calls; a
// bare `SetBitVectorMode(32)` therefore leaks BV state into the next
// caller (which may want Int-mode proofs). Use `ScopedBVMode` whenever a
// caller needs BV semantics for a bounded scope:
//
//   {
//     ScopedBVMode g(prover, 32);
//     prover.Bind(...);
//     prover.CanProve(...);
//   }  // <-- prover is back in its prior mode here.
//
// Switching mode invalidates previously declared variables (see
// `SetBitVectorMode` docs); on enter and on exit the solver state is
// reset, so do not nest scopes that share variable bindings.
class ScopedBVMode {
 public:
  ScopedBVMode(Z3Prover& prover, int width)
      : prover_(prover), prev_width_(prover.GetBitVectorWidth()) {
    prover_.SetBitVectorMode(width);
  }
  ~ScopedBVMode() { prover_.SetBitVectorMode(prev_width_); }

  ScopedBVMode(const ScopedBVMode&) = delete;
  ScopedBVMode& operator=(const ScopedBVMode&) = delete;
  ScopedBVMode(ScopedBVMode&&) = delete;
  ScopedBVMode& operator=(ScopedBVMode&&) = delete;

 private:
  Z3Prover& prover_;
  int prev_width_;
};

// Per-Analyzer cache — owns the real `Z3Prover` instances. Keyed by
// `Analyzer*` so successive `Z3Prover(analyzer)` calls in the same logical
// scope share solver state. `thread_local` guards Z3 context affinity (Z3's
// own context is not thread-safe).
Z3Prover& GetOrCreate(::tvm::arith::Analyzer* analyzer);

}  // namespace tlz3
}  // namespace tilelang

// Free-function shim — replaces the `Z3ProverStub` shim. Call sites continue
// to write `arith::Z3Prover(analyzer).Foo(...)`; this resolves to the free
// function (NOT a constructor) because the real class lives in the
// `tilelang::z3::` namespace, not `tvm::arith::`.
namespace tvm {
namespace arith {
inline ::tilelang::tlz3::Z3Prover& Z3Prover(Analyzer& a) {
  return ::tilelang::tlz3::GetOrCreate(&a);
}
inline ::tilelang::tlz3::Z3Prover& Z3Prover(Analyzer* a) {
  return ::tilelang::tlz3::GetOrCreate(a);
}
}  // namespace arith
}  // namespace tvm

#endif  // TILELANG_VENDORED_Z3_PROVER_H_
