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
  explicit Z3Prover(::tvm::arith::Analyzer *parent);
  ~Z3Prover();

  Z3Prover(const Z3Prover &) = delete;
  Z3Prover &operator=(const Z3Prover &) = delete;

  // Public surface mirrors the TileLang-fork class. Return types use
  // ffi::String upstream; we surface std::string for caller convenience —
  // every call site streams the result into a stringstream, so this is
  // lossless.
  void Bind(const ::tvm::tirx::Var &var, const ::tvm::Range &new_range,
            bool allow_override = false);
  void Bind(const ::tvm::tirx::Var &var, const ::tvm::PrimExpr &expr,
            bool allow_override = false);
  // Queries the solver to see if `expr` (interpreted as a boolean
  // condition) is universally true under the current constraint stack.
  // Returns true if proven, false otherwise (timeout, unsupported
  // operator, or counter-example found).
  bool CanProve(const ::tvm::PrimExpr &expr);
  std::function<void()> EnterConstraint(const ::tvm::PrimExpr &constraint,
                                        bool is_assume = false);
  std::string GetSMTLIB2(::tvm::ffi::Optional<::tvm::PrimExpr> expr);
  // Convenience overload — accepts std::nullopt directly without forcing the
  // caller to construct an `ffi::Optional<PrimExpr>`.
  std::string GetSMTLIB2(std::nullopt_t);
  std::string GetStats();
  std::string GetModel(const ::tvm::PrimExpr &expr);
  int64_t CountSatisfyingValues(const ::tvm::tirx::Var &var, int64_t max_count,
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
  //
  // CPPMEGA fix-A2 audit (z3-stack, 2026-05-07): the only call sites of
  // `SetBitVectorMode` outside this header/impl are the test-FFI helpers
  // `BvCanProve` and `BvScopedRoundTrip` (see z3_prover.cc, both of which
  // create a fresh `Analyzer` and exit immediately after the call, so
  // mode leak is impossible). There are zero production call sites today.
  // Future production callers MUST use `ScopedBVMode` and not the raw
  // method — adding a new direct call requires reviewer sign-off.
  void SetBitVectorMode(int width);
  int GetBitVectorWidth() const;

  // CPPMEGA z3-stack fix-A6: per-pass solver reset. Clears memo/scope
  // stack and rebuilds the solver while preserving the current
  // `bv_width_`. Use this between pass invocations instead of
  // `SetBitVectorMode(currentWidth)` (which is a no-op due to the
  // mode-equality fast-path and therefore does NOT clear state).
  // Currently has no in-tree caller; provided for future pass drivers
  // that want a clean proof context without flipping mode.
  //
  // CPPMEGA fix-B7 (idea712): atomic reset of (var memo + solver assertions
  // + scope stack). This is the audit hook for any future
  // `SetBitVectorMode(width)` port from the parallel `z3-stack` branch:
  // when the prover swaps int sort for BV<width>, every memoized
  // `PrimExpr -> z3::expr` mapping becomes invalid (it's still pointing at
  // the old sort), AND the solver's accumulated assertions reference those
  // stale exprs, AND the scope stack remembers their push/pop pairings.
  //
  // The right invariant is "state is one indivisible unit". Tearing it
  // apart — clearing memo_ but not the solver, or vice versa — produces
  // exactly the silent correctness bugs the cross-checked review flagged.
  //
  // `Reset()` enforces the invariant: must be called when the constraint
  // scope stack is at its root (no outstanding `EnterConstraint`
  // recoverers); fails-fast otherwise to surface the lifecycle violation.
  void Reset();

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
  ScopedBVMode(Z3Prover &prover, int width)
      : prover_(prover), prev_width_(prover.GetBitVectorWidth()) {
    prover_.SetBitVectorMode(width);
  }
  // CPPMEGA fix-A1: dtor is `noexcept` because RAII destructors must not
  // throw — terminate() if SetBitVectorMode ever propagated an exception
  // would be the only safe outcome. The infallible variant of
  // SetBitVectorMode (catches any internal exception, logs, defaults to
  // bv_width=0) means we don't actually rely on this in practice, but the
  // marker makes the contract explicit at the type-system level.
  ~ScopedBVMode() noexcept { prover_.SetBitVectorMode(prev_width_); }

  ScopedBVMode(const ScopedBVMode &) = delete;
  ScopedBVMode &operator=(const ScopedBVMode &) = delete;
  ScopedBVMode(ScopedBVMode &&) = delete;
  ScopedBVMode &operator=(ScopedBVMode &&) = delete;

private:
  Z3Prover &prover_;
  int prev_width_;
};

// Per-Analyzer cache — owns the real `Z3Prover` instances. Keyed by
// `Analyzer*` so successive `Z3Prover(analyzer)` calls in the same logical
// scope share solver state. `thread_local` guards Z3 context affinity (Z3's
// own context is not thread-safe).
Z3Prover &GetOrCreate(::tvm::arith::Analyzer *analyzer);

// CPPMEGA z3-final per-pass gate (2026-05-07). Granular complement to the
// blanket `TILELANG_DISABLE_Z3=1` global gate at `Z3Prover::CanProve`.
// Each Z3-using pass calls `Z3PassGate::IsEnabled("<NAME>")` before invoking
// the prover. The gate returns false if EITHER:
//   * the global env `TILELANG_DISABLE_Z3` is set to a non-empty / non-"0"
//     value (kill-switch), OR
//   * the per-pass env `TILELANG_DISABLE_Z3_<NAME>` is set the same way.
// Default (env unset): returns true → pass uses the prover normally.
//
// Pass names defined today (see README for table mapping name → idea):
//   VECTORIZE          (loop_vectorize.cc, ideas #1/#12)
//   PREDICATE_FUSION   (predicate_fusion.cc, idea #7)
//   DROP_BOUND_CHECKS  (drop_provable_bound_checks.cc, idea #4)
//   TMA_LEGALITY       (op/copy.cc, idea #6)
//   BARRIER_ELISION    (thread_storage_sync.cc, idea #11)
//   ALIAS_SHAPE        (reserved central alias/shape proof hook)
//   AUTO_DOUBLE_BUFFER (auto_double_buffer.cc, idea #2; reserved — currently
//                       no live Z3 call sites in stub mode)
//   INT24              (analysis/int24_overflow_proof.py, idea #5)
//   DOT4_LEGALITY      (language/fp8_op.py, idea #10)
//   SIMDGROUP          (transform/metal_*_to_simdgroup.py / simd_lift.py,
//                       ideas #8/#9)
//
// Lookups are thread-local-cached — first hit reads `getenv`, subsequent
// hits hit a small unordered_map. This keeps the gate cheap on the hot
// path (vectorize/drop-bound-checks call `IsEnabled` per probe).
class Z3PassGate {
public:
  static bool IsEnabled(const char *pass_name);
};

// CPPMEGA z3-stack fix-A8 (NEW-2): per-pass cache hygiene. Pass drivers
// MUST call `ClearProverCache()` (or `ResetProverFor(specific_analyzer)`)
// at pass entry to prevent cross-pass contamination — the per-thread
// `Analyzer*`-keyed cache survives across passes and a heap-address
// reuse for a freed Analyzer would otherwise hand a stale prover (with
// stale memo/scope/bv-mode) to the new owner. Cheap; idempotent.
void ClearProverCache();
void ResetProverFor(::tvm::arith::Analyzer *analyzer);

} // namespace tlz3
} // namespace tilelang

// Free-function shim — replaces the `Z3ProverStub` shim. Call sites continue
// to write `arith::Z3Prover(analyzer).Foo(...)`; this resolves to the free
// function (NOT a constructor) because the real class lives in the
// `tilelang::z3::` namespace, not `tvm::arith::`.
namespace tvm {
namespace arith {
inline ::tilelang::tlz3::Z3Prover &Z3Prover(Analyzer &a) {
  return ::tilelang::tlz3::GetOrCreate(&a);
}
inline ::tilelang::tlz3::Z3Prover &Z3Prover(Analyzer *a) {
  return ::tilelang::tlz3::GetOrCreate(a);
}
} // namespace arith
} // namespace tvm

#endif // TILELANG_VENDORED_Z3_PROVER_H_
