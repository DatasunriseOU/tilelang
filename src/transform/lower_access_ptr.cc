/*!
 * \file lower_access_ptr.cc
 * \brief Lower TileLang frontend `tl.access_ptr` to
 * `tir.builtin.tvm_access_ptr`.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

DataType IndexDTypeFromBuffer(const Buffer &buffer) {
  if (!buffer.defined() || buffer->shape.empty()) {
    return DataType::Int(32);
  }
  return buffer->shape[0].dtype();
}

Array<PrimExpr> RowMajorStrides(const Buffer &buffer) {
  int ndim = static_cast<int>(buffer->shape.size());
  Array<PrimExpr> strides;
  DataType idx_dtype = IndexDTypeFromBuffer(buffer);
  for (int i = 0; i < ndim; ++i) {
    PrimExpr stride = make_const(idx_dtype, 1);
    for (int j = i + 1; j < ndim; ++j) {
      stride = stride * buffer->shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

PrimExpr BaseIndexForOffset(const PrimExpr &index) {
  if (const auto *ramp = index.as<RampNode>()) {
    return ramp->base;
  }
  if (const auto *broadcast = index.as<BroadcastNode>()) {
    return broadcast->value;
  }
  return index;
}

// CPPMEGA: pull from stack-c — handle if_then_else-wrapped BufferLoad
// produced by LegalizeSafeMemoryAccess. Returns the underlying load if
// the expression is `if_then_else(cond, BufferLoad, fallback)` or
// directly a `BufferLoad`; FATAL otherwise.
BufferLoad ExtractAccessPtrBaseLoad(const PrimExpr &expr) {
  if (const auto *base_load = expr.as<BufferLoadNode>()) {
    return ffi::GetRef<BufferLoad>(base_load);
  }

  if (const auto *call = expr.as<CallNode>();
      call != nullptr && call->op.same_as(builtin::if_then_else()) &&
      call->args.size() == 3U) {
    // LegalizeSafeMemoryAccess wraps predicated loads as
    // if_then_else(cond, BufferLoad(...), safe_zero).  For tl.access_ptr the
    // value is irrelevant; keep the guarded load's address expression.
    return ExtractAccessPtrBaseLoad(call->args[1]);
  }

  LOG(FATAL) << "tl.access_ptr arg0 must be BufferLoad, but got " << expr;
  return BufferLoad();
}

PrimExpr LinearOffsetFromLoad(const BufferLoad &load) {
  Buffer buffer = load->buffer;
  ICHECK(buffer.defined());
  int ndim = static_cast<int>(buffer->shape.size());
  ICHECK_EQ(static_cast<int>(load->indices.size()), ndim)
      << "tl.access_ptr expects a BufferLoad with indices matching buffer ndim";

  Array<PrimExpr> strides;
  if (!buffer->strides.empty() &&
      buffer->strides.size() == buffer->shape.size()) {
    strides = buffer->strides;
  } else {
    strides = RowMajorStrides(buffer);
  }

  DataType idx_dtype = IndexDTypeFromBuffer(buffer);
  PrimExpr offset = make_const(idx_dtype, 0);
  for (int i = 0; i < ndim; ++i) {
    PrimExpr idx = BaseIndexForOffset(load->indices[i]);
    offset = offset + idx * strides[i];
  }
  return offset;
}

class AccessPtrLowerer : public StmtExprMutator {
public:
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(tl::access_ptr())) {
      return std::move(call);
    }

    ICHECK_EQ(call->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";

    // CPPMEGA: pull from stack-c — tolerate if_then_else-wrapped BufferLoad.
    BufferLoad base_load = ExtractAccessPtrBaseLoad(call->args[0]);
    Buffer buffer = base_load->buffer;
    ICHECK(buffer.defined());

    PrimExpr extent = call->args[1];
    PrimExpr rw_mask = call->args[2];

    PrimExpr ptype = tirx::TypeAnnotation(buffer->dtype);
    PrimExpr data = buffer->data;
    PrimExpr offset = LinearOffsetFromLoad(base_load);

    Array<PrimExpr> args{ptype, data, offset, extent, rw_mask};
    return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
  }
};

PrimFunc LowerAccessPtrPrimFunc(PrimFunc f) {
  if (!f.defined() || !f->body.defined()) {
    return f;
  }
  AccessPtrLowerer lowerer;
  PrimFuncNode *n = f.CopyOnWrite();
  n->body = lowerer(std::move(n->body));
  return f;
}

} // namespace

namespace transform {

tvm::transform::Pass LowerAccessPtr() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    return LowerAccessPtrPrimFunc(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                                  "tl.LowerAccessPtr", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerAccessPtr", LowerAccessPtr);
}

} // namespace transform

} // namespace tl
} // namespace tvm
