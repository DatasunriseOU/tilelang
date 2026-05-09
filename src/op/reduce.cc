/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"

#include <cstdint>
#include <cmath>
#include <limits>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"
#include "tirx/transform/ir_utils.h"
#include "tvm/ir/expr.h"
#include "tvm/tirx/expr.h"
#include "tvm/tirx/stmt.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

struct ReductionPlan {
  int reducing_threads{0};
  int scale{0};
  PrimExpr thread_offset;
  bool same_simdgroup_metal_fast_path_safe{false};
};

bool IsPositivePowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int64_t ConstIntAfterSimplify(const PrimExpr &expr, arith::Analyzer *analyzer,
                              const char *name) {
  PrimExpr simplified = analyzer != nullptr ? analyzer->Simplify(expr) : expr;
  const int64_t *value = as_const_int(simplified);
  ICHECK(value != nullptr)
      << "ReduceOp: " << name
      << " must be a compile-time constant after arith::Analyzer simplification; got "
      << simplified;
  return *value;
}

ReductionPlan MakeReductionPlan(const PrimExpr &extent_expr,
                                 const PrimExpr &scale_expr,
                                 const PrimExpr &thread_offset_expr,
                                 const Target &target,
                                 arith::Analyzer *analyzer,
                                 const char *context,
                                 int reduce_type) {
  int64_t extent = ConstIntAfterSimplify(extent_expr, analyzer, "extent");
  int64_t scale = ConstIntAfterSimplify(scale_expr, analyzer, "scale");
  ICHECK_GT(extent, 0) << "ReduceOp" << context
                      << ": extent must be positive; got " << extent;
  ICHECK_GT(scale, 0) << "ReduceOp" << context
                     << ": scale must be positive; got " << scale;
  ICHECK_LE(extent, std::numeric_limits<int>::max() / scale)
      << "ReduceOp" << context << ": reducing_threads overflows int: extent="
      << extent << ", scale=" << scale;

  int64_t reducing_threads = extent * scale;

  PrimExpr thread_offset =
      analyzer != nullptr ? analyzer->Simplify(thread_offset_expr)
                          : thread_offset_expr;
  const int64_t *thread_offset_value = as_const_int(thread_offset);
  ICHECK(thread_offset_value != nullptr)
      << "ReduceOp" << context
      << ": thread_offset must be a compile-time constant after "
      << "arith::Analyzer simplification; got " << thread_offset;

  if (extent != 1) {
    ICHECK(IsPositivePowerOfTwo(reducing_threads))
        << "ReduceOp" << context
        << ": reducing_threads must be a power of two for the AllReduce "
        << "XOR-butterfly to be correct; got " << reducing_threads
        << " for type=" << reduce_type
        << ". Adjust the fragment layout / thread split or insert an "
        << "identity-pad before the reduce.";
  }

  bool same_simdgroup_metal_fast_path_safe =
      extent != 1 &&
      IsSameSimdgroupMetalReductionSafe(
          target, static_cast<int>(reducing_threads), static_cast<int>(scale),
          thread_offset, analyzer);

  return ReductionPlan{static_cast<int>(reducing_threads),
                       static_cast<int>(scale), thread_offset,
                       same_simdgroup_metal_fast_path_safe};
}

}  // namespace

bool IsSameSimdgroupMetalReductionSafe(const Target &target,
                                       int reducing_threads, int scale,
                                       const PrimExpr &thread_offset_expr,
                                       arith::Analyzer *analyzer) {
  if (!TargetIsMetal(target) || scale != 1 || reducing_threads <= 0 ||
      reducing_threads > 32 || !IsPositivePowerOfTwo(reducing_threads)) {
    return false;
  }

  // Z3 is not useful for this legality check. `thread_offset` is printed into
  // the MSL `AllReduce<..., thread_offset, ...>` template argument, so codegen
  // already requires the Analyzer to reduce it to an integer constant. Once it
  // is constant, the exact condition is modular arithmetic over the 32-wide
  // Metal simdgroup lane id: every xor partner used by the butterfly must stay
  // inside the same contiguous reduce group.
  PrimExpr thread_offset =
      analyzer != nullptr ? analyzer->Simplify(thread_offset_expr)
                          : thread_offset_expr;
  const int64_t *thread_offset_value = as_const_int(thread_offset);
  if (thread_offset_value == nullptr || *thread_offset_value < 0) {
    return false;
  }

  const int64_t simd_lane = *thread_offset_value % 32;
  return simd_lane + reducing_threads <= 32 &&
         simd_lane % reducing_threads == 0;
}

// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

ReduceOp::ReduceOp(Array<PrimExpr> args, Map<String, ffi::ObjectRef> annotations) {
  ObjectPtr<ReduceOpNode> node = tvm::ffi::make_object<ReduceOpNode>();
  // Accept BufferRegion/BufferLoad for src/dst
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  node->type = ReduceType(reduce_type);
  node->clear = args[4].as<Bool>().value();
  // Optional "batch" annotation: number of output elements per batched
  // AllReduce call (default 1 = scalar).
  if (auto opt = annotations.Get("batch")) {
    if (auto i = opt.value().as<IntImm>()) {
      node->batch = static_cast<int>(i.value()->value);
      CHECK_GE(node->batch, 1) << "ReduceOp: batch must be >= 1";
    }
  }
  // Optional annotation: "nan_propagate" — for fp16/bf16 max/min/absmax,
  // when true, lower to CUDA __hmax_nan/__hmin_nan so NaNs propagate.
  if (auto opt = annotations.Get("nan_propagate")) {
    if (auto b = opt.value().as<Bool>()) {
      node->nan_propagate = b.value();
    } else if (auto i = opt.value().as<IntImm>()) {
      node->nan_propagate = i.value()->value != 0;
    }
  }
  data_ = std::move(node);
}

AccessRegions ReduceOpNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(srcRegion_);
  if (!clear) {
    result.reads.push_back(dstRegion_);
  }
  result.writes.push_back(dstRegion_);
  return result;
}

TileOperator ReduceOpNode::Clone() const {
  auto op = tvm::ffi::make_object<ReduceOpNode>(*this);
  return ReduceOp(op);
}

TileOperator CumSumOpNode::Clone() const {
  auto op = tvm::ffi::make_object<CumSumOpNode>(*this);
  return CumSumOp(op);
}

PrimExpr ReduceOpNode::MakeInitValue() const {
  auto dst_dtype = dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  if (type->isSum()) {
    return make_zero(dst->dtype);
  } else if (type->isAbsSum()) {
    return make_zero(dst->dtype);
  } else if (type->isMax()) {
    if (is_int) {
      return make_const(dst->dtype, -(int64_t(1) << (bits - 1)));
    } else if (is_uint) {
      return make_const(dst->dtype, 0);
    } else {
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isMin()) {
    if (is_int) {
      return make_const(dst->dtype, (int64_t(1) << (bits - 1)) - 1);
    } else if (is_uint) {
      return make_const(dst->dtype, (int64_t(1) << bits) - 1);
    } else {
      return make_const(dst->dtype, INFINITY);
    }
  } else if (type->isAbsMax()) {
    return make_const(dst->dtype, 0);
  } else if (type->isBitAnd()) {
    if (is_int) {
      return make_const(dst->dtype, -1);
    } else if (is_uint) {
      return make_const(dst->dtype, (int64_t(1) << bits) - 1);
    } else {
      // Should not arrive here
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isBitOr()) {
    return make_zero(dst->dtype);
  } else if (type->isBitXor()) {
    return make_zero(dst->dtype);
  } else if (type->isMul()) {
    // Wave-8 #5: identity element for product reduction is 1.
    //
    // Wave-11 #1 (closes grok rev_d1fb5da1bb / meta rev_528f578233 HIGH on
    // "warp lane mask exploit" — silent wrong product on non-warp-divisible
    // dim length). The MakeInitValue() result here writes the multiplicative
    // identity into *every* clear_buffer slot of every participating thread
    // BEFORE the per-thread reduce loop unrolls. Threads that the fragment
    // layout maps to zero src elements (because the reduce-dim length is
    // smaller than reducing_threads, e.g. dim=33 lowered onto 32 threads
    // means lane 32..reducing_threads-1 see no src element) therefore enter
    // AllReduce<MulOp,...> holding T(1), not garbage. The subsequent XOR-
    // butterfly thus combines `acc * 1 == acc` for the inactive lanes — the
    // identity-pad contract documented in src/tl_templates/{cuda,hip}/reduce.h
    // is enforced *at lowering time* by this initial store, not by callers.
    // The static_assert((threads & (threads-1)) == 0, ...) on AllReduce
    // additionally rejects non-power-of-2 reducing_threads at compile time.
    return make_const(dst->dtype, 1);
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return PrimExpr();
  }
}

PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &acc,
                                  const PrimExpr &b) const {
  PrimExpr rhs = b;
  if (acc->dtype != rhs->dtype) {
    rhs = Cast(acc->dtype, rhs);
  }
  const bool use_nan_op =
      nan_propagate && (acc.dtype().is_float16() || acc.dtype().is_bfloat16());
  if (type->isSum()) {
    return acc + rhs;
  } else if (type->isAbsSum()) {
    return acc + Max(rhs, -rhs);
  } else if (type->isMax()) {
    if (use_nan_op) {
      return Call(acc.dtype(), tl::max_nan(), {acc, rhs});
    }
    return Max(acc, rhs);
  } else if (type->isMin()) {
    if (use_nan_op) {
      return Call(acc.dtype(), tl::min_nan(), {acc, rhs});
    }
    return Min(acc, rhs);
  } else if (type->isAbsMax()) {
    if (use_nan_op) {
      return Call(acc.dtype(), tl::max_nan(), {acc, tvm::abs(rhs)});
    }
    return Max(acc, tvm::abs(rhs));
  } else if (type->isBitAnd()) {
    return acc & rhs;
  } else if (type->isBitOr()) {
    return acc | rhs;
  } else if (type->isBitXor()) {
    return acc ^ rhs;
  } else if (type->isMul()) {
    // Wave-8 #5: combine via floating/integer multiply. The dtype-cast
    // above already aligned `rhs` with `acc`, so this commutes with the
    // sum/max paths without inserting any non-last-axis vector ops
    // (the bug Wave-7 #5 documented).
    return acc * rhs;
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
  }
}

std::string ReduceOpNode::MakeCodegenReducer() const {
  const bool use_nan_op =
      nan_propagate && (dst->dtype.is_float16() || dst->dtype.is_bfloat16());
  if (type->isSum()) {
    return "tl::SumOp";
  } else if (type->isAbsSum()) {
    return "tl::SumOp";
  } else if (type->isMax()) {
    return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
  } else if (type->isMin()) {
    return use_nan_op ? "tl::MinOpNan" : "tl::MinOp";
  } else if (type->isAbsMax()) {
    return use_nan_op ? "tl::MaxOpNan" : "tl::MaxOp";
  } else if (type->isBitAnd()) {
    return "tl::BitAndOp";
  } else if (type->isBitOr()) {
    return "tl::BitOrOp";
  } else if (type->isBitXor()) {
    return "tl::BitXorOp";
  } else if (type->isMul()) {
    // Wave-8 #5: emits the warp-level product-reducer template.
    // The matching `tl::MulOp` lives in the runtime templates under
    // `src/tl_templates/{cuda,hip,metal}/reduce.h` next to `tl::SumOp`.
    // If a backend has not added a `MulOp` template yet, codegen will
    // surface a clean undeclared-identifier error at compile time.
    return "tl::MulOp";
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return "";
  }
}

static Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

static Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;

  auto fwd = InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));

  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }

  auto reducer_layout =
      Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt)
          ->CondenseReplicateVar()
          ->BindThreadRange(src_layout->ThreadRange());
  return reducer_layout;
}

/**
 * @brief Lower the Reduce operator to a TIR statement.
 *
 * Lowers a ReduceOpNode operating on fragment-scoped buffers into a sequence of
 * TIR statements implementing: optional initialization, thread-local reduction
 * (unrolled inner loops), inter-thread reduction via a runtime AllReduce call
 * (Hopper targets use `NamedBarrier` instead of the default
 * `SyncThreadsBarrier`), and an optional accumulation or copy back to the
 * destination buffer when a temporary clear buffer is used.
 *
 * Behavior notes:
 * - Only supports src and dst in "local.fragment" scope; otherwise it checks
 *   and aborts with "Reduce for shared memory not implemented.".
 * - Supports both 1D reductions (scalar output) and reductions along a single
 *   extra dimension; validates layout dimensionality consistency.
 * - If `clear` is set (or for sum/abssum reductions), an initial value is
 *   written to the clear buffer; for non-clearing sum/abssum a duplicate
 *   temporary buffer is allocated and accumulated back into dst after
 * reduction.
 * - Performs iterator compression for local reduction loops using `analyzer`.
 * - Detects parallel thread splitting from the normalized iterator sum and
 *   emits a call to a templated `tl::AllReduce<...>::run`
 *   via `builtin::call_extern`. For sufficiently large reducing thread counts
 *   (> 32) a workspace is allocated via T.AddWorkspace and passed to the
 *   AllReduce call.
 * - The final body is wrapped in parallel loops over the destination spatial
 *   dimensions and partitioned by the lowering thread variable. If a temporary
 *   clear buffer is used, it is allocated for the body.
 *
 * @param T Lowering context providing buffer and layout maps, thread bounds,
 *          target information, thread variable, and workspace allocation
 * helper.
 * @param analyzer Analyzer used for iterator compression and arithmetic
 * normalization.
 * @return Stmt Lowered TIR statement implementing the reduction.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (nan_propagate && (dst->dtype.is_float16() || dst->dtype.is_bfloat16()) &&
      !TargetIsCuda(T.target)) {
    LOG(FATAL) << "ReduceOp: nan_propagate=True for fp16/bf16 max/min/absmax "
                  "is only supported on CUDA targets (requires "
                  "__hmax_nan/__hmin_nan intrinsics). Target was: "
               << T.target->str();
  }
  auto get_buffer = [&](const Buffer &buf) {
    if (T.buffer_remap.count(buf))
      return T.buffer_remap[buf];
    return buf;
  };

  auto src_scope = this->src.scope();
  auto dst_scope = this->dst.scope();

  if (src_scope == "local.fragment" && dst_scope == "local.fragment") {

    auto src_buffer = get_buffer(this->src);
    auto dst_buffer = get_buffer(this->dst);
    auto src_layout = T.layout_map[this->src].as<Fragment>().value();
    auto dst_layout = T.layout_map[this->dst].as<Fragment>().value();
    auto red_layout = ComputeReducerLayout(src_layout, dim);
    auto src_dim = src_layout->InputDim();
    auto dst_dim = dst_layout->InputDim();

    auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

    if (is_1d_reduce) {
      ICHECK(is_one(dst_layout->OutputShape().back()))
          << "Reduce for scalar not implemented.";
    } else {
      ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
    }

    Array<IterVar> dst_vars;
    for (size_t i = 0; i < dst_dim; ++i) {
      Var var = Var(std::string{char('i' + i)});
      dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                                 IterVarType::kDataPar));
    }

    Array<IterVar> src_vars;
    if (!is_1d_reduce) {
      src_vars = dst_vars;
    }
    Range reduce_dom(0, src_layout->InputShape()[this->dim]);
    IterVar reduce_iv(reduce_dom, Var("rv"), IterVarType::kDataPar);
    src_vars.insert(src_vars.begin() + this->dim, reduce_iv);

    auto src_indices = src_layout->Forward(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto dst_indices = dst_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto red_indices = red_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

    Array<Stmt> stmts;

    auto require_init = this->clear;
    if (this->type->isSum() || this->type->isAbsSum() ||
        this->type->isBitAnd() || this->type->isBitOr() ||
        this->type->isBitXor()) {
      require_init = true;
    }

    auto clear_buffer = dst_buffer;
    auto need_duplicate = false;
    auto need_update = false;
    if ((this->type->isSum() || this->type->isAbsSum()) && !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if (this->type->isBitAnd() && !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if ((this->type->isBitOr() || this->type->isBitXor()) &&
               !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if ((this->type->isMax() || this->type->isMin() ||
                this->type->isAbsMax()) &&
               !this->clear) {
      need_duplicate = true;
      need_update = true;
    }

    // red_layout should always contain dst_layout
    // if we can prove they are the same, no need to duplicate buffer
    // otherwise, red_layout contains more replicated dimensions than dst_layout
    if (!analyzer->CanProve(dst_layout->ReplicateExtent() ==
                            red_layout->ReplicateExtent())) {
      need_duplicate = true;
    }
    ICHECK(!analyzer->CanProve(dst_layout->ReplicateExtent() >
                               red_layout->ReplicateExtent()))
        << "Inconsistent layouts between src and dst in ReduceOp: "
        << "dst_layout=" << dst_layout << "red_layout=" << red_layout;

    if (need_duplicate) {
      // Create a new buffer with same shape and dtype as dst_buffer
      clear_buffer = decl_buffer(red_layout->OutputShape(), dst_buffer->dtype,
                                 dst_buffer->name + "_clear",
                                 GetPtrStorageScope(dst_buffer->data));
    }
    // make reduce-init stmt
    // For max/min/absmax with clear=false and need_duplicate, we still need to
    // initialize the temporary buffer with identity values since the original
    // dst values will be combined later via need_update
    if (require_init ||
        (need_duplicate && (this->type->isMax() || this->type->isMin() ||
                            this->type->isAbsMax()))) {
      stmts.push_back(
          BufferStore(clear_buffer, this->MakeInitValue(), red_indices));
    }

    // make thread-local reduce
    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
      auto [expr, var] = CompressIterator(src_indices[i], src_vars,
                                          src_vars[this->dim]->var, analyzer);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }

    Stmt reduce_local = BufferStore(
        clear_buffer,
        this->MakeReduce(BufferLoad(clear_buffer, red_indices),
                         BufferLoad(src_buffer, src_indice_compressed)),
        red_indices);

    for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
      reduce_local =
          For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent,
              ForKind::kUnrolled, reduce_local, std::nullopt,
              {{tirx::attr::pragma_unroll_explicit, Bool(false)}});
    }
    stmts.push_back(reduce_local);

    auto src_thread = src_layout->ForwardThread(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
    auto iter_sum =
        arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);

    // batch is set by the user via the "batch" annotation (default 1 = scalar).
    // When batch > 1 the compiler phases the reduction:
    //   1. init + local reduce loop
    //   2. ceil(N/batch) batched AllReduce calls, each sharing one barrier pair
    //   3. copy-back loop (only when need_duplicate)
    const int batch = this->batch;

    // Validate batch against the actual per-thread element count N.
    if (batch > 1) {
      int64_t N_total = 1;
      for (const auto &s : clear_buffer->shape) {
        const int64_t *p = as_const_int(s);
        ICHECK(p != nullptr) << "ReduceOp: batch > 1 requires compile-time "
                                "constant output shape";
        N_total *= *p;
      }
      CHECK_LE(batch, N_total)
          << "ReduceOp: batch=" << batch
          << " exceeds per-thread output element count N=" << N_total;
      CHECK_EQ(N_total % batch, 0)
          << "ReduceOp: batch=" << batch << " must evenly divide N=" << N_total;
    }

    bool use_batch = batch > 1;

    // Helper: wrap a body in the dst_vars loops with partitioning & unrolling.
    auto make_dst_loop = [&](Stmt body, const Array<IterVar> &vars) -> Stmt {
      for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
        body = For(vars[i]->var, 0, vars[i]->dom->extent, ForKind::kParallel,
                   body);
      }
      body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                           red_layout);
      body = PragmaUnrollLoop(Downcast<For>(body));
      return body;
    };

    // Helper: create fresh dst loop variables (needed so that pre/post loops
    // do not reuse the same Var objects).
    auto make_fresh_dst_vars = [&](const std::string &suffix)
        -> std::tuple<Array<IterVar>, Array<PrimExpr>, Array<PrimExpr>> {
      Array<IterVar> vars;
      for (size_t i = 0; i < dst_dim; ++i) {
        Var v(std::string{char('i' + i)} + suffix);
        vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), v,
                               IterVarType::kDataPar));
      }
      auto d_idx = dst_layout->Forward(
          vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
      auto r_idx = red_layout->Forward(
          vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
      return {vars, d_idx, r_idx};
    };

    if (use_batch) {
      // ================================================================
      // Batched AllReduce path — three phases:
      //   1. Loop: init + thread-local reduce
      //   2. Flat: batched AllReduce (single butterfly pass for all values)
      //   3. Loop: copy-back (only when need_duplicate)
      // ================================================================

      // Phase 1: pre-reduce loop
      Stmt pre_body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
      pre_body = make_dst_loop(pre_body, dst_vars);

      Array<Stmt> phases;
      phases.push_back(pre_body);

      // Phase 2: batched AllReduce call(s).
      // workspace_stride = reducing_threads (SoA layout in smem:
      //   slot for batch item b, thread t = red_buf[b * reducing_threads + t])
      for (const auto &iter_split : iter_sum->args) {
        auto mark = iter_split->source->source.as<Var>();
        if (!mark)
          continue;
        if (!mark.value().same_as(src_vars[this->dim]->var))
          continue;
        auto plan = MakeReductionPlan(iter_split->extent, iter_split->scale,
                                      T.thread_bounds->min, T.target, analyzer,
                                      "(batched)", this->type->type);
        if (plan.reducing_threads == plan.scale)
          continue;

        int reducing_threads = plan.reducing_threads;
        int scale = plan.scale;
        auto thread_offset = plan.thread_offset;
        std::stringstream ss;
        int workspace_stride =
            plan.same_simdgroup_metal_fast_path_safe ? 0 : reducing_threads;

        // Use run_batch (not run) to avoid overload-resolution ambiguity when
        // a pointer is passed as first argument.
        if (TargetHasSMVersionGE(T.target, 90)) {
          auto all_threads = T.thread_bounds->extent;
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << scale << ", " << thread_offset
             << ", tl::NamedBarrier<" << all_threads << ">, " << batch << ", "
             << reducing_threads << ">::run_batch";
        } else if (TargetIsRocm(T.target)) {
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << scale << ", " << thread_offset
             << ", " << batch << ", " << workspace_stride << ">::run_batch";
        } else if (TargetIsMetal(T.target)) {
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << scale << ", " << thread_offset
             << ", tl::SyncThreadsBarrier, " << batch << ", "
             << workspace_stride << ">::run_batch";
        } else {
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << scale << ", " << thread_offset
             << ", tl::SyncThreadsBarrier, " << batch << ", "
             << workspace_stride << ">::run_batch";
        }

        // Workspace is only needed for cross-warp reduce (> 32 threads), or for
        // Metal reductions that cannot be proven to stay inside one simdgroup.
        PrimExpr workspace;
        bool need_workspace =
            (TargetIsMetal(T.target) &&
             !plan.same_simdgroup_metal_fast_path_safe) ||
            (!TargetIsMetal(T.target) && reducing_threads > 32);
        if (need_workspace) {
          int ws_size = workspace_stride * batch;
          workspace = T.AddWorkspace(ws_size, clear_buffer->dtype);
        }

        // Compute N_total and num_chunks for this buffer.
        int64_t N_total = 1;
        for (const auto &s : clear_buffer->shape)
          N_total *= *as_const_int(s);
        int num_chunks = static_cast<int>(N_total / batch);

        // Compute strides for reverse-linearisation of clear_buffer->shape.
        int buf_ndim = static_cast<int>(clear_buffer->shape.size());
        std::vector<int64_t> buf_shape_vals;
        for (const auto &s : clear_buffer->shape)
          buf_shape_vals.push_back(*as_const_int(s));
        std::vector<int64_t> buf_strides(buf_ndim, 1);
        for (int d = buf_ndim - 2; d >= 0; d--)
          buf_strides[d] = buf_strides[d + 1] * buf_shape_vals[d + 1];

        for (int chunk = 0; chunk < num_chunks; chunk++) {
          int64_t flat_offset = (int64_t)chunk * batch;
          // Map flat_offset to multi-dim indices in clear_buffer.
          Array<PrimExpr> chunk_indices;
          for (int d = 0; d < buf_ndim; d++) {
            int64_t idx = (flat_offset / buf_strides[d]) % buf_shape_vals[d];
            chunk_indices.push_back(Integer(idx));
          }
          // Pointer to the start of this chunk's elements in clear_buffer.
          PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                              {BufferLoad(clear_buffer, chunk_indices)});

          Array<PrimExpr> args = {StringImm(ss.str()), ptr};
          if (TargetIsMetal(T.target)) {
            args.push_back(T.thread_var);
            if (need_workspace) {
              args.push_back(workspace);
            }
          } else if (need_workspace) {
            args.push_back(workspace);
          }
          phases.push_back(
              Evaluate(Call(DataType::Handle(), builtin::call_extern(), args)));
        }
      }

      // Phase 3: copy-back (only when a temp buffer was used)
      if (need_duplicate) {
        auto [post_vars, post_dst_idx, post_red_idx] =
            make_fresh_dst_vars("_p");

        // Recompute predicate with post_vars.
        PrimExpr predicate = Bool(true);
        {
          auto dst_th = post_dst_idx;
          dst_th.push_back(T.thread_var);
          auto inv = dst_layout->Inverse()->Forward(dst_th);
          inv.pop_back();
          for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++)
            predicate = predicate && (inv[i] == post_vars[i]->var);
          predicate = analyzer->Simplify(predicate);
        }

        PrimExpr update;
        if (need_update) {
          auto src_val = BufferLoad(clear_buffer, post_red_idx);
          auto dst_val = BufferLoad(dst_buffer, post_dst_idx);
          if (this->type->isSum() || this->type->isAbsSum()) {
            update = dst_val + src_val;
          } else if (this->type->isBitAnd()) {
            update = this->clear ? src_val : bitwise_and(dst_val, src_val);
          } else if (this->type->isBitOr()) {
            update = bitwise_or(dst_val, src_val);
          } else if (this->type->isBitXor()) {
            update = bitwise_xor(dst_val, src_val);
          } else if (this->type->isMax() || this->type->isAbsMax()) {
            update = Max(dst_val, src_val);
          } else if (this->type->isMin()) {
            update = Min(dst_val, src_val);
          } else {
            LOG(FATAL) << "Unsupported reduce type: " << this->type->type;
          }
        } else {
          update = BufferLoad(clear_buffer, post_red_idx);
        }
        auto store = BufferStore(dst_buffer, update, post_dst_idx);
        Stmt post_body;
        if (analyzer->CanProve(predicate)) {
          post_body = store;
        } else {
          post_body = IfThenElse(predicate, store);
        }
        phases.push_back(make_dst_loop(post_body, post_vars));
      }

      Stmt body = phases.size() > 1 ? SeqStmt(phases) : phases[0];
      if (need_duplicate) {
        // CPPMEGA: apache/tvm latest replaced Allocate(buffer_var, dtype, shape, cond, body)
        // with AllocBuffer(buffer) as a standalone scope-introducing stmt in SeqStmt context.
        body = SeqStmt({AllocBuffer(clear_buffer), body});
      }
      return body;

    } else {
      // ================================================================
      // Original scalar AllReduce path (unchanged).
      // ================================================================
      for (const auto &iter_split : iter_sum->args) {
        auto mark = iter_split->source->source.as<Var>();
        if (!mark)
          continue;
        if (mark.value().same_as(src_vars[this->dim]->var)) {
          auto plan = MakeReductionPlan(iter_split->extent, iter_split->scale,
                                        T.thread_bounds->min, T.target,
                                        analyzer, "", this->type->type);
          if (plan.reducing_threads == plan.scale)
            continue;

          // Wave-11 #1: lowering-time enforcement of the AllReduce contract.
          // The XOR-butterfly recursion in
          // src/tl_templates/{cuda,hip}/reduce.h halves `threads` at every
          // step and shuffles with offset = threads/2; this only converges
          // correctly when reducing_threads is a power of two. The CUDA
          // template's `shfl_xor_sync(0xffffffff, ...)` additionally
          // requires every participating lane's red_buf slot to hold a
          // valid identity-padded value — which `MakeInitValue()` already
          // writes before the per-thread reduce loop runs (see the kMul
          // comment there). This ICHECK therefore closes the meta/grok
          // wave-10 HIGH on "silent wrong product on non-warp-divisible N"
          // by refusing to lower to AllReduce when the structural
          // power-of-two invariant cannot hold.
          int reducing_threads = plan.reducing_threads;
          int scale = plan.scale;
          std::stringstream ss;

          auto thread_offset = plan.thread_offset;
          int workspace_stride =
              plan.same_simdgroup_metal_fast_path_safe ? 0 : reducing_threads;
          if (TargetHasSMVersionGE(T.target, 90)) {
            auto all_threads = T.thread_bounds->extent;
            ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
               << reducing_threads << ", " << scale << ", " << thread_offset
               << ", tl::NamedBarrier<" << all_threads << ">>::run";
          } else if (TargetIsMetal(T.target)) {
            ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
               << reducing_threads << ", " << scale << ", "
               << thread_offset << ", tl::SyncThreadsBarrier, 1, "
               << workspace_stride << ">::run";
          } else {
            ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
               << reducing_threads << ", " << scale << ", " << thread_offset
               << ">::run";
          }
          Array<PrimExpr> thread_reduce_args = {
              StringImm(ss.str()), BufferLoad(clear_buffer, red_indices)};
          if (TargetIsMetal(T.target)) {
            thread_reduce_args.push_back(T.thread_var);
            if (!plan.same_simdgroup_metal_fast_path_safe) {
              PrimExpr workspace =
                  T.AddWorkspace(workspace_stride, clear_buffer->dtype);
              thread_reduce_args.push_back(workspace);
            }
          } else if (reducing_threads > 32) {
            int workspace_size =
                static_cast<int>(*as_const_int(T.thread_bounds->extent));
            PrimExpr workspace =
                T.AddWorkspace(workspace_size, clear_buffer->dtype);
            thread_reduce_args.push_back(workspace);
          }
          auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                           thread_reduce_args);
          stmts.push_back(BufferStore(clear_buffer, call, red_indices));
        }
      }

      PrimExpr predicate = Bool(true);
      {
        auto dst_th_indices = dst_indices;
        dst_th_indices.push_back(T.thread_var);
        auto inv = dst_layout->Inverse()->Forward(dst_th_indices);
        inv.pop_back();
        for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
          predicate = predicate && (inv[i] == dst_vars[i]->var);
        }
        predicate = analyzer->Simplify(predicate);
      }
      if (need_duplicate) {
        PrimExpr update;
        if (need_update) {
          auto src_val = BufferLoad(clear_buffer, red_indices);
          auto dst_val = BufferLoad(dst_buffer, dst_indices);
          if (this->type->isSum() || this->type->isAbsSum()) {
            update = dst_val + src_val;
          } else if (this->type->isBitAnd()) {
            update = this->clear ? src_val : bitwise_and(dst_val, src_val);
          } else if (this->type->isBitOr()) {
            update = bitwise_or(dst_val, src_val);
          } else if (this->type->isBitXor()) {
            update = bitwise_xor(dst_val, src_val);
          } else if (this->type->isMax() || this->type->isAbsMax()) {
            update = Max(dst_val, src_val);
          } else if (this->type->isMin()) {
            update = Min(dst_val, src_val);
          } else {
            LOG(FATAL) << "Unsupported reduce type: " << this->type->type;
          }
        } else {
          update = BufferLoad(clear_buffer, red_indices);
        }
        auto store = BufferStore(dst_buffer, update, dst_indices);
        if (analyzer->CanProve(predicate)) {
          stmts.push_back(store);
        } else {
          stmts.push_back(IfThenElse(predicate, store));
        }
      }

      auto body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
      for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
        body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                   ForKind::kParallel, body);
      }

      if (dst_layout->InputDim() > 0) {
        body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                             red_layout);
        body = PragmaUnrollLoop(Downcast<For>(body));
      } else {
        auto guard = (T.thread_var == T.thread_bounds->min);
        body = IfThenElse(guard, body);
      }

      if (need_duplicate) {
        // CPPMEGA: see comment above re Allocate→AllocBuffer migration.
        body = SeqStmt({AllocBuffer(clear_buffer), body});
      }
      return body;
    }
  }

  LOG(FATAL) << "Reduce for buffers in scope (" << src_scope << ", "
             << dst_scope << ") is not implemented.";
  return Stmt();
}

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();
    auto reducer_layout = ComputeReducerLayout(src_layout, this->dim);

    if (!T.layout_map.count(dst)) {
      return {{dst, reducer_layout}};
    }

    auto orig_dst_layout = T.layout_map.Get(dst).value().as<Fragment>().value();
    ICHECK(reducer_layout->InputDim() == orig_dst_layout->InputDim());

    auto indices = InputPlaceholders(reducer_layout->InputDim());
    arith::Analyzer analyzer;
    for (size_t i = 0; i < indices.size(); i++) {
      analyzer.Bind(Downcast<Var>(indices[i]),
                    Range(0, reducer_layout->InputShape()[i]));
    }
    if (!ProveFragmentContains(orig_dst_layout, reducer_layout, indices,
                               indices, analyzer)) {
      std::ostringstream oss;
      oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
          << src << "\n"
          << "src_layout = " << src_layout << "\n"
          << "reducer_layout = " << reducer_layout << "\n"
          << "orig_dst_layout = " << orig_dst_layout << "\n"
          << "You may need to use a shared memory to transform the "
             "layout";
      throw LayoutConflictException(oss.str());
    }
  }
  return {};
}

TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Normalize "Buffer" to BufferRegion. Use the shape of the buffer as the
// ranges.
static BufferRegion ConvertBufferToBufferRegion(const Buffer &buf) {
  Array<Range> ranges;
  for (PrimExpr extent : buf->shape) {
    ranges.push_back(Range(IntImm(extent->dtype, 0), extent));
  }
  return BufferRegion(buf, ranges);
}

CumSumOp::CumSumOp(Array<PrimExpr> args, Map<String, ffi::ObjectRef> annotations) {
  /// CumSum constructor arguments:
  /// - src: input buffer
  /// - dst: output buffer
  /// - dim: dimension to cumsum
  /// - reverse: whether to cumsum in reverse order
  CHECK_EQ(args.size(), 4);
  ObjectPtr<CumSumOpNode> node = tvm::ffi::make_object<CumSumOpNode>();
  // node->src = vmap[GetVarFromAccessPtr(args[0])];
  // node->dst = vmap[GetVarFromAccessPtr(args[1])];
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  node->dim = args[2].as<IntImm>().value()->value;
  node->reverse = args[3].as<Bool>().value();
  CHECK_LT(node->dim, static_cast<int>(node->src->shape.size()))
      << "The dim of cumsum should be less than the number of dimensions. Got "
         "dim="
      << node->dim << ", but src has " << node->src->shape.size() << " dims.";

  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (IsFragmentBuffer(this->src) && IsFragmentBuffer(this->dst)) {
    LOG(FATAL) << "CumSum for fragment not implemented, please raise an issue "
                  "if you need this feature.";
  } else if (IsSharedBuffer(this->src)) {
    ICHECK(IsSharedBuffer(this->dst));
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    Array<PrimExpr> args;

    // Build access pointers from regions locally
    PrimExpr srcPtr = MakeAccessPtrFromRegion(srcRegion_, 1);
    PrimExpr dstPtr = MakeAccessPtrFromRegion(dstRegion_, 2);

    // Use region extents instead of buffer shape for correct slice handling
    Array<PrimExpr> src_extents;
    for (const auto &range : srcRegion_->region) {
      src_extents.push_back(range->extent);
    }
    int ndim = static_cast<int>(src_extents.size());

    if (ndim == 1) {
      ICHECK_EQ(dim, 0) << "Cumulative sum over a 1D buffer only supports dim "
                           "= 0.";
      ss << "tl::CumSum1D<" << threads << ", " << (reverse ? "true" : "false")
         << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0]};
    } else if (ndim == 2) {
      ss << "tl::CumSum2D<" << threads << ", " << dim << ", "
         << (reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0],
              src_extents[1]};
    } else {
      LOG(FATAL) << "CumSum currently supports only 1D or 2D buffers, got "
                 << ndim << "D.";
    }
    return Evaluate(Call(dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower cumsum for " << this->src.scope() << " and "
                  << this->dst.scope();
  }

  return Stmt();
}

LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  // Only infer layout in strict mode
  if (level != InferLevel::kStrict) {
    return {};
  }

  LayoutMap result_map;

  auto make_linear_layout = [](const Buffer &buf) -> Layout {
    return makeLinearLayout(buf->shape);
  };

  auto check_or_set_linear_layout = [&](const Buffer &buf) {
    if (!IsSharedBuffer(buf))
      return;

    Layout linear_layout = make_linear_layout(buf);
    if (T.layout_map.count(buf)) {
      // Check if existing layout is linear
      Layout existing = T.layout_map.Get(buf).value().as<Layout>().value();
      ICHECK(StructuralEqual()(existing, linear_layout))
          << "CumSum requires linear layout for shared buffer " << buf->name
          << ", but got non-linear layout.";
    } else {
      result_map.Set(buf, linear_layout);
    }
  };

  check_or_set_linear_layout(src);
  check_or_set_linear_layout(dst);

  return result_map;
}

TIR_REGISTER_TL_TILE_OP(CumSumOp, cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

bool MetalReductionSameSimdgroupFastPathSafeForTest(
    Target target, int reducing_threads, int scale, int64_t thread_offset) {
  arith::Analyzer analyzer;
  return IsSameSimdgroupMetalReductionSafe(
      target, reducing_threads, scale,
      IntImm(DataType::Int(64), thread_offset), &analyzer);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ReduceOpNode::RegisterReflection();
  CumSumOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
  refl::GlobalDef().def("tl.metal.reduce_same_simdgroup_fast_path_safe",
                        MetalReductionSameSimdgroupFastPathSafeForTest);
}

} // namespace tl
} // namespace tvm
