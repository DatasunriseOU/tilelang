/*!
 * \file src/op/finalize_reducer.cc
 *
 * Define finalize_reducer operator.
 */

#include "finalize_reducer.h"

#include <array>
#include <limits>
#include <sstream>
#include <vector>

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "../target/utils.h"
#include "reduce.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

std::vector<FinalizeReducerImpl> &FinalizeReducerImplRegistry() {
  static std::vector<FinalizeReducerImpl> registry;
  return registry;
}

const FinalizeReducerImpl &ResolveFinalizeReducerImpl(Target target) {
  const auto &registry = FinalizeReducerImplRegistry();
  const FinalizeReducerImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const FinalizeReducerImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr)
      << "finalize_reducer requires a target-specific implementation, but no "
         "implementation is registered for "
      << target;
  return *best_impl;
}

int ThreadBlockExtent(const LowerArgs &T) {
  return static_cast<int>(*as_const_int(T.thread_bounds->extent));
}

bool MatchDefaultFinalizeReducerTarget(Target target) {
  return !TargetIsMetal(target);
}

std::string MakeDefaultFinalizeScalarAllReduce(const FinalizeReducerOpNode &op,
                                               const LowerArgs &T,
                                               const ReductionPlan &plan,
                                               const std::string &op_str) {
  (void)op;
  std::stringstream ss;
  if (TargetHasSMVersionGE(T.target, 90)) {
    auto all_threads = T.thread_bounds->extent;
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset << ", tl::NamedBarrier<"
       << all_threads << ">>::run";
  } else {
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset << ">::run";
  }
  return ss.str();
}

std::string MakeDefaultFinalizeBatchAllReduce(const FinalizeReducerOpNode &op,
                                              const LowerArgs &T,
                                              const ReductionPlan &plan,
                                              const std::string &op_str,
                                              int64_t batch) {
  (void)op;
  std::stringstream ss;
  const int workspace_stride = ThreadBlockExtent(T);
  if (TargetHasSMVersionGE(T.target, 90)) {
    auto all_threads = T.thread_bounds->extent;
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset << ", tl::NamedBarrier<"
       << all_threads << ">, " << batch << ", " << workspace_stride
       << ">::run_batch";
  } else if (TargetIsRocm(T.target)) {
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset << ", " << batch << ", "
       << workspace_stride << ">::run_batch";
  } else {
    ss << "tl::AllReduce<" << op_str << ", " << plan.reducing_threads << ", "
       << plan.scale << ", " << plan.thread_offset
       << ", tl::SyncThreadsBarrier, " << batch << ", " << workspace_stride
       << ">::run_batch";
  }
  return ss.str();
}

bool DefaultFinalizeNeedsScalarWorkspace(const LowerArgs &T,
                                         const ReductionPlan &plan) {
  (void)T;
  return plan.reducing_threads >= 32;
}

int DefaultFinalizeScalarWorkspaceSize(const LowerArgs &T,
                                       const ReductionPlan &plan) {
  (void)plan;
  return ThreadBlockExtent(T);
}

bool DefaultFinalizeNeedsBatchWorkspace(const LowerArgs &T,
                                        const ReductionPlan &plan,
                                        int64_t batch) {
  (void)T;
  (void)plan;
  (void)batch;
  return true;
}

int DefaultFinalizeBatchWorkspaceSize(const LowerArgs &T,
                                      const ReductionPlan &plan,
                                      int64_t batch) {
  (void)plan;
  return ThreadBlockExtent(T) * static_cast<int>(batch);
}

void AppendDefaultFinalizeArgs(Array<PrimExpr> *args, const LowerArgs &T,
                               bool need_workspace, const PrimExpr &workspace) {
  (void)T;
  if (need_workspace) {
    args->push_back(workspace);
  }
}

bool RegisterDefaultFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "default.FinalizeReducer",
      MatchDefaultFinalizeReducerTarget,
      0,
      MakeDefaultFinalizeScalarAllReduce,
      MakeDefaultFinalizeBatchAllReduce,
      DefaultFinalizeNeedsScalarWorkspace,
      DefaultFinalizeScalarWorkspaceSize,
      DefaultFinalizeNeedsBatchWorkspace,
      DefaultFinalizeBatchWorkspaceSize,
      AppendDefaultFinalizeArgs,
      AppendDefaultFinalizeArgs,
  });
  return true;
}

const bool default_finalize_reducer_registered =
    RegisterDefaultFinalizeReducer();

} // namespace

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.make_scalar_allreduce != nullptr);
  ICHECK(impl.make_batch_allreduce != nullptr);
  ICHECK(impl.needs_scalar_workspace != nullptr);
  ICHECK(impl.scalar_workspace_size != nullptr);
  ICHECK(impl.needs_batch_workspace != nullptr);
  ICHECK(impl.batch_workspace_size != nullptr);
  ICHECK(impl.append_scalar_args != nullptr);
  ICHECK(impl.append_batch_args != nullptr);
  FinalizeReducerImplRegistry().push_back(impl);
}

/**
 * @brief Construct a FinalizeReducerOp from TL operator arguments and a buffer
 * map.
 *
 * Extracts the reducer Buffer from `vmap` using the variable referenced by
 * `args[0]` and sets the reduction operation type from the integer code in
 * `args[1]`.
 *
 * @param args TL operator arguments: expects at least two elements where
 *             `args[0]` is an access pointer identifying the reducer variable
 * and `args[1]` is an integer encoding a `ReducerOpType` (e.g., Sum/Max/Min).
 */
FinalizeReducerOp::FinalizeReducerOp(Array<PrimExpr> args,
                                     Map<String, ffi::ObjectRef> annotations) {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>();
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region =
      BufferRegion::FullRegion(reducer_access.region->buffer);
  reducer_access.access_mask = kAccessReadWrite;
  node->reducer = reducer_access.region->buffer;
  node->SetAccessRegions({reducer_access});
  node->op = (ReducerOpType)*as_const_int(args[1]);
  // Read explicit batch size from annotations (0 means auto-detect).
  if (annotations.count("batch")) {
    node->batch = (int)*as_const_int(Downcast<PrimExpr>(annotations["batch"]));
    CHECK_GE(node->batch, 1)
        << "finalize_reducer: batch must be >= 1, got " << node->batch;
  }
  data_ = std::move(node);
}

/**
 * @brief Lower the finalize_reducer TL operator to a TIR statement.
 *
 * Lowers the operator that finalizes a reducer by performing a thread-wide
 * AllReduce across the reducer's output elements and writing the reduced value
 * back into the reducer buffer. The function:
 * - Fetches the reducer buffer and expects its layout to be a Fragment.
 * - Builds index Vars for each output dimension.
 * - Reads the layout's ReplicateExtent and:
 *   - if extent == 1, emits a no-op Evaluate(0);
 *   - otherwise constructs an AllReduce extern call (uses `NamedBarrier` when
 *     the compilation target is Hopper) with an optional workspace (allocated
 * via T.AddWorkspace when reducing_threads >= 32) and stores the result via
 *     BufferStore.
 * - Wraps the store in parallel outer For loops over each output dimension.
 *
 * @param T Lowering context containing buffer remapping, layout map, thread
 * bounds, target, and helper methods (e.g., AddWorkspace).
 * @param analyzer Arithmetic analyzer (unused by this implementation but
 * provided for consistency with lowering API).
 * @return Stmt The lowered TIR statement representing the AllReduce and
 * surrounding loops.
 *
 * @note The function ICHECKs that the reducer layout is present and a Fragment,
 *       and that ReplicateExtent is either 1 or equal to the thread block
 * extent; violations cause a fatal check failure.
 */
Stmt FinalizeReducerOpNode::Lower(const LowerArgs &T,
                                  arith::Analyzer *analyzer) const {
  auto buffer = T.buffer_remap[reducer];
  auto opt_layout = T.layout_map.Get(reducer);
  ICHECK(opt_layout);
  ICHECK(opt_layout->as<Fragment>());
  auto layout = opt_layout->as<Fragment>().value();
  Array<PrimExpr> indices_0;
  indices_0.reserve(layout->OutputDim());
  for (int i = 0; i < layout->OutputDim(); ++i)
    indices_0.push_back(Var("__finred_" + std::to_string(i)));

  const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
  ICHECK(p_extent);
  int extent = *p_extent, scale = 1;
  ICHECK(extent == 1 || extent == *as_const_int(T.thread_bounds->extent))
      << "Illegal finalize_reducer: extent=" << extent
      << "; T.thread_bounds=" << T.thread_bounds;

  if (extent == 1)
    return Evaluate(0);

  std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp", "tl::MulOp"};
  std::string op_str = op_names[(int)op];

  // adopted from ReduceOp
  int reducing_threads = extent;
  auto thread_offset = T.thread_bounds->min;
  bool same_simdgroup_metal_fast_path_safe = IsSameSimdgroupMetalReductionSafe(
      T.target, reducing_threads, scale, thread_offset, analyzer);
  ReductionPlan plan{reducing_threads, scale, thread_offset,
                     same_simdgroup_metal_fast_path_safe};
  const FinalizeReducerImpl &finalize_impl =
      ResolveFinalizeReducerImpl(T.target);

  // Validate batch against the layout's total output element count.
  int64_t layout_batch_size = 1;
  for (int i = 0; i < layout->OutputDim(); ++i) {
    const int64_t *p = as_const_int(layout->OutputShape()[i]);
    if (p == nullptr) {
      layout_batch_size = -1;
      break;
    }
    layout_batch_size *= *p;
  }

  int64_t effective_batch = static_cast<int64_t>(this->batch);

  if (effective_batch > 1 && layout_batch_size > 0) {
    CHECK_LE(effective_batch, layout_batch_size)
        << "finalize_reducer: batch (" << effective_batch
        << ") exceeds total output elements (" << layout_batch_size << ")";
    CHECK_EQ(layout_batch_size % effective_batch, 0)
        << "finalize_reducer: batch (" << effective_batch
        << ") must evenly divide total output elements (" << layout_batch_size
        << ")";
  }

  // ROCm wavefronts are 64-wide; only batch when reducing across warps.
  const int warp_size = TargetIsRocm(T.target) ? 64 : 32;
  bool use_batch = effective_batch > 1 && reducing_threads > warp_size;
  ICHECK(reducing_threads > 0 &&
         (reducing_threads & (reducing_threads - 1)) == 0)
      << "finalize_reducer: reducing_threads must be a power of two for the "
      << "AllReduce XOR-butterfly to be correct; got " << reducing_threads
      << "; op=" << static_cast<int>(op) << "; dtype=" << buffer->dtype
      << "; batch=" << effective_batch << "; target=" << T.target->str();

  if (use_batch) {
    // Batched AllReduce: single butterfly pass for all output elements.
    std::string allreduce = finalize_impl.make_batch_allreduce(
        *this, T, plan, op_str, effective_batch);
    bool need_workspace =
        finalize_impl.needs_batch_workspace(T, plan, effective_batch);
    PrimExpr workspace;
    if (need_workspace) {
      int ws_size =
          finalize_impl.batch_workspace_size(T, plan, effective_batch);
      workspace = T.AddWorkspace(ws_size, buffer->dtype);
    }
    Array<PrimExpr> args = {StringImm(allreduce), buffer->data};
    finalize_impl.append_batch_args(&args, T, need_workspace, workspace);
    return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  }

  // Scalar AllReduce path (original).
  std::string allreduce =
      finalize_impl.make_scalar_allreduce(*this, T, plan, op_str);
  Array<PrimExpr> thread_reduce_args = {StringImm(allreduce),
                                        BufferLoad(buffer, indices_0)};
  bool need_workspace = finalize_impl.needs_scalar_workspace(T, plan);
  PrimExpr workspace;
  if (need_workspace) {
    workspace = T.AddWorkspace(finalize_impl.scalar_workspace_size(T, plan),
                               buffer->dtype);
  }
  finalize_impl.append_scalar_args(&thread_reduce_args, T, need_workspace,
                                   workspace);
  auto call = Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
  Stmt body = BufferStore(buffer, call, indices_0);

  // make the outer spatial loop
  for (int i = layout->OutputDim() - 1; i >= 0; i--) {
    body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
               ForKind::kParallel, body);
  }

  return body;
}

/**
 * @brief Infer and return the layout mapping for the reducer buffer.
 *
 * Copies the existing layout for the reducer from the provided LayoutInferArgs
 * into a new LayoutMap and returns it. The inference does not modify the
 * layout; it preserves the reducer's current layout.
 *
 * @param T Provides the input layout map from which the reducer's layout is
 * copied.
 * @param level Unused by this operator; present for API compatibility.
 * @return LayoutMap A map that contains the reducer buffer mapped to its
 * original layout.
 */
LayoutMap FinalizeReducerOpNode::InferLayout(const LayoutInferArgs &T,
                                             InferLevel level) const {
  LayoutMap layout_map;
  layout_map.Set(reducer, T.layout_map.Get(reducer).value());
  return layout_map;
}

/**
 * @brief Create a deep copy of this FinalizeReducerOpNode and wrap it as a
 * TileOperator.
 *
 * Constructs a new FinalizeReducerOpNode by copying the current node state and
 * returns a TileOperator that owns the copied node.
 *
 * @return TileOperator A TileOperator that contains a deep copy of this node.
 */
TileOperator FinalizeReducerOpNode::Clone() const {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { FinalizeReducerOpNode::RegisterReflection(); }
} // namespace tl
} // namespace tvm
