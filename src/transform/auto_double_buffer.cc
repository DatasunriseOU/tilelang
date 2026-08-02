/*!
 * \file auto_double_buffer.cc
 * \brief Automatic double-buffer (ping-pong) insertion for shared-memory tile
 *        loads, gated by a Z3 soundness proof.
 *
 * High-level idea (Z3 Roadmap Idea #2):
 *
 *   For a canonical pattern of the form:
 *
 *     for k in range(N):
 *       A_local[idx] = load(A_global, k * stride + idx)   # pattern A
 *       use(A_local[idx])                                  # pattern B
 *
 *   we may overlap iteration k+1's load with iteration k's `use(...)` by
 *   allocating a second ("pong") buffer and swapping which buffer is the
 *   load target / use source on alternating iterations.
 *
 *   The transformation is SOUND only if we can prove that no thread reads
 *   buf[i] while another thread writes buf[i+1] (the new write target). The
 *   simplest sufficient condition is that the source address of iteration
 *   k+1's load is independent of iteration k's read in the same buffer slot.
 *   Symbolically:
 *
 *     forall tx, k:  read_slot(tx, k)  !=  write_slot(tx, k+1)
 *
 *   In a single-buffer scheme this fails by construction (same buffer, same
 *   slot). Adding a pong buffer makes it trivially true (different buffers).
 *   The remaining proof obligation is: the user-visible semantics is
 *   preserved — i.e. the load address at k+1 does not depend on the value
 *   produced at k. We approximate that with the symbolic check below; when
 *   Z3 cannot discharge it we conservatively leave the IR untouched.
 *
 * Status (this commit):
 *
 *   This pass is the SAFE-STUB form requested in the roadmap:
 *
 *     - Default OFF: gated by `tl.auto_double_buffer = True`.
 *     - The detector recognizes the canonical pattern via `BlockAnalyzer`.
 *     - The Z3 obligation is constructed but the prover is invoked in a
 *       conservative mode: any uncertainty -> NO transformation.
 *     - When a candidate is detected and config is ON, we still return the
 *       IRModule unchanged and emit a `LOG(INFO)` recording the detection.
 *       The actual ping-pong rewrite is deferred to a future iteration —
 *       this lets the wiring (Pass registration, phase slot, PassConfig,
 *       Python binding, tests) ship safely while a real transformation is
 *       developed and validated.
 *
 *   This is intentionally conservative-by-default: the pass is a perf
 *   optimization, not a correctness fix.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <string>
#include <unordered_set>

#include "../op/builtin.h"
#include "vendored/z3_prover.h"

namespace tvm {
namespace tl {

using namespace tirx;

// PassConfig key — opt-in. Default OFF.
static constexpr const char *kAutoDoubleBuffer = "tl.auto_double_buffer";

namespace {

/*!
 * \brief Result of the canonical-pattern detector for a single For loop.
 *
 * `candidate_buffer` is non-null when the detector found a shared buffer
 * that is loaded then used inside the same K-iteration, with no
 * cross-iteration dependency observed by the heuristic.
 */
struct CandidateInfo {
  Buffer candidate_buffer;     // shared-memory tile that could be ping-ponged
  Buffer source_global_buffer; // global source for the load
  PrimExpr load_address_expr;  // symbolic address of the load (in `k`)
};

/*!
 * \brief Walk a single For body and try to recognize the canonical pattern:
 *        write-shared-from-global followed by read-shared in the same iter.
 *
 * Heuristics (kept simple by design):
 *   - Find a `BufferStore` whose target buffer has scope "shared" or
 *     "shared.dyn".
 *   - The store value must be a `BufferLoad` (or trivial cast of one) of a
 *     buffer with scope "" or "global".
 *   - A subsequent `BufferLoad` of the same shared buffer must occur in the
 *     same loop body (i.e. used in the same iteration).
 */
class CanonicalPatternDetector : public StmtExprVisitor {
public:
  CandidateInfo info;
  bool found_load = false;
  bool found_use_after_load = false;

  void VisitStmt_(const BufferStoreNode *op) final {
    if (!found_load) {
      const std::string scope = op->buffer.scope();
      if (scope == "shared" || scope == "shared.dyn") {
        // Check the rhs: is it a global -> shared load?
        const BufferLoadNode *src = op->value.as<BufferLoadNode>();
        if (!src) {
          if (const auto *cast = op->value.as<CastNode>()) {
            src = cast->value.as<BufferLoadNode>();
          }
        }
        if (src) {
          const std::string src_scope = src->buffer.scope();
          if (src_scope.empty() || src_scope == "global") {
            info.candidate_buffer = op->buffer;
            info.source_global_buffer = src->buffer;
            // First (and only the first) index — keep it simple.
            if (!src->indices.empty()) {
              info.load_address_expr = src->indices[0];
            } else {
              info.load_address_expr = IntImm(DataType::Int(32), 0);
            }
            found_load = true;
            return; // do not descend — preserve scan order
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (found_load && !found_use_after_load &&
        info.candidate_buffer.defined() &&
        op->buffer.same_as(info.candidate_buffer)) {
      found_use_after_load = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

/*!
 * \brief Build the symbolic Z3 obligation for ping-pong soundness.
 *
 * The obligation has two parts:
 *   1. The next-iteration load address must be computable independently of
 *      the previous iteration's USE. We approximate this as: the load
 *      address expression in `k+1` does not refer to the candidate buffer.
 *   2. The candidate buffer slot for read at iter k must differ from the
 *      write target slot at iter k+1. With ping-pong (two physical
 *      allocations) this is trivially true; we still emit it symbolically
 *      so the prover has a chance to confirm.
 *
 * Returned PrimExpr is fed into `Z3Prover::CanProve(...)`. If the prover
 * returns false (either disproved or unknown), the pass MUST NOT
 * transform.
 */
PrimExpr BuildSoundnessObligation(const CandidateInfo &info, const Var &k_var) {
  Var k_next("k_next", k_var.dtype());

  bool load_addr_independent =
      !UsesVar(info.load_address_expr, [&](const VarNode *v) {
        return v == info.candidate_buffer->data.get();
      });

  if (!load_addr_independent) {
    return Bool(false);
  }

  return floormod(k_var, 2) != floormod(k_next, 2);
}

/*!
 * \brief Top-level mutator. In stub mode this does NOT modify the IR; it
 *        only logs detected candidates. The Z3 obligation is built and
 *        evaluated for an audit trail, but the result is discarded.
 *
 * To turn this into a real transformation a future change should:
 *   - Allocate a second buffer (`<name>_pong`) of identical layout.
 *   - Rewrite the loop body to swap the pair on alternating iterations.
 *   - Insert a guard for the first iteration's prefetch.
 *   - Return the Z3-proof-protected new IR.
 */
class AutoDoubleBufferRewriter : public StmtExprMutator {
public:
  explicit AutoDoubleBufferRewriter(bool enabled) : enabled_(enabled) {}

  int candidates_detected() const { return candidates_detected_; }

  Stmt VisitStmt_(const ForNode *op) final {
    // Recurse first to handle nested patterns bottom-up.
    Stmt body = VisitStmt(op->body);

    if (enabled_ && op->kind == ForKind::kSerial) {
      CanonicalPatternDetector det;
      det(body);
      if (det.found_load && det.found_use_after_load) {
        candidates_detected_++;

        arith::Analyzer analyzer;
        analyzer.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
        PrimExpr obligation = BuildSoundnessObligation(det.info, op->loop_var);
        bool proved =
            arith::Z3Prover(analyzer).CanProve(analyzer.Simplify(obligation));

        std::ostringstream candidate_name;
        if (det.info.candidate_buffer.defined()) {
          candidate_name << det.info.candidate_buffer->name;
        } else {
          candidate_name << "<unnamed>";
        }

        if (proved) {
          LOG(INFO) << "[AutoDoubleBuffer] candidate detected for buffer '"
                    << candidate_name.str()
                    << "', soundness obligation proved by Z3, but "
                    << "no transformation emitted yet (safe-stub mode).";
        } else {
          LOG(INFO) << "[AutoDoubleBuffer] candidate detected for buffer '"
                    << candidate_name.str()
                    << "', but Z3 could not prove ping-pong soundness; "
                    << "falling back to single buffer.";
        }
      }
    }

    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations);
  }

private:
  bool enabled_;
  int candidates_detected_{0};
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass AutoDoubleBuffer() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    bool enabled = ctx->GetConfig<Bool>(kAutoDoubleBuffer, Bool(false)).value();
    if (!enabled) {
      // Default OFF: skip the IR traversal entirely.
      return f;
    }
    AutoDoubleBufferRewriter rewriter(enabled);
    Stmt new_body = rewriter(f->body);
    // Even when enabled, the safe-stub does not modify the IR; this returns
    // `f` whose body is structurally equal to the input.
    if (!new_body.same_as(f->body)) {
      f.CopyOnWrite()->body = new_body;
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AutoDoubleBuffer", {});
}

TVM_REGISTER_PASS_CONFIG_OPTION(kAutoDoubleBuffer, Bool);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoDoubleBuffer", AutoDoubleBuffer);
}

} // namespace tl
} // namespace tvm
