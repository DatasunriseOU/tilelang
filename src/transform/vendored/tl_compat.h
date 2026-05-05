// SPDX-License-Identifier: Apache-2.0
// TileLang compat shim: alias old ICHECK family → apache/tvm-latest TVM_FFI_ICHECK family.
// Force-included from CMake via -include flag so every .cc gets ICHECK transparently.
#ifndef TILELANG_VENDORED_TL_COMPAT_H_
#define TILELANG_VENDORED_TL_COMPAT_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/runtime/logging.h>

// Re-expose tvm::Object / tvm::ObjectRef / tvm::ObjectPtr at top-level
// (apache/tvm latest moved them to tvm::ffi:: but TileLang code still uses
// unqualified forms heavily).
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>

namespace tvm {
using Object = ::tvm::ffi::Object;
using ObjectRef = ::tvm::ffi::ObjectRef;
template <typename T>
using ObjectPtr = ::tvm::ffi::ObjectPtr<T>;
using ObjectPtrHash = ::tvm::ffi::ObjectPtrHash;
using ObjectPtrEqual = ::tvm::ffi::ObjectPtrEqual;
using StructuralEqual = ::tvm::ffi::StructuralEqual;
using StructuralHash = ::tvm::ffi::StructuralHash;
using PackedArgs = ::tvm::ffi::PackedArgs;
}  // namespace tvm

#ifndef CHECK
#define CHECK(x) TVM_FFI_CHECK(x, InternalError)
#endif
#ifndef CHECK_EQ
#define CHECK_EQ(x, y) TVM_FFI_CHECK_EQ(x, y, InternalError)
#endif
#ifndef CHECK_NE
#define CHECK_NE(x, y) TVM_FFI_CHECK_NE(x, y, InternalError)
#endif
#ifndef CHECK_LT
#define CHECK_LT(x, y) TVM_FFI_CHECK_LT(x, y, InternalError)
#endif
#ifndef CHECK_LE
#define CHECK_LE(x, y) TVM_FFI_CHECK_LE(x, y, InternalError)
#endif
#ifndef CHECK_GT
#define CHECK_GT(x, y) TVM_FFI_CHECK_GT(x, y, InternalError)
#endif
#ifndef CHECK_GE
#define CHECK_GE(x, y) TVM_FFI_CHECK_GE(x, y, InternalError)
#endif

#ifndef ICHECK
#define ICHECK TVM_FFI_ICHECK
#endif
#ifndef ICHECK_EQ
#define ICHECK_EQ(x, y) TVM_FFI_ICHECK_EQ(x, y)
#endif
#ifndef ICHECK_NE
#define ICHECK_NE(x, y) TVM_FFI_ICHECK_NE(x, y)
#endif
#ifndef ICHECK_LT
#define ICHECK_LT(x, y) TVM_FFI_ICHECK_LT(x, y)
#endif
#ifndef ICHECK_LE
#define ICHECK_LE(x, y) TVM_FFI_ICHECK_LE(x, y)
#endif
#ifndef ICHECK_GT
#define ICHECK_GT(x, y) TVM_FFI_ICHECK_GT(x, y)
#endif
#ifndef ICHECK_GE
#define ICHECK_GE(x, y) TVM_FFI_ICHECK_GE(x, y)
#endif
#ifndef ICHECK_NOTNULL
#define ICHECK_NOTNULL(x) TVM_FFI_ICHECK_NOTNULL(x)
#endif
#ifndef DCHECK
#define DCHECK TVM_FFI_DCHECK
#endif

// CPPMEGA: alias old DLTensor field-kind names (kArr*) to apache/tvm latest
// (kDLTensor* enumerators in tirx::builtin::TVMStructFieldKind). Avoids having
// to touch dozens of TileLang call sites that still use the old short form.
#include <tvm/tirx/builtin.h>
namespace tvm {
namespace tirx {
namespace builtin {
constexpr TVMStructFieldKind kArrAddr = kDLTensorAddr;
constexpr TVMStructFieldKind kArrData = kDLTensorData;
constexpr TVMStructFieldKind kArrShape = kDLTensorShape;
constexpr TVMStructFieldKind kArrStrides = kDLTensorStrides;
constexpr TVMStructFieldKind kArrNDim = kDLTensorNDim;
constexpr TVMStructFieldKind kArrTypeCode = kDLTensorTypeCode;
constexpr TVMStructFieldKind kArrTypeBits = kDLTensorTypeBits;
constexpr TVMStructFieldKind kArrTypeLanes = kDLTensorTypeLanes;
constexpr TVMStructFieldKind kArrByteOffset = kDLTensorByteOffset;
constexpr TVMStructFieldKind kArrDeviceId = kDLTensorDeviceId;
constexpr TVMStructFieldKind kArrDeviceType = kDLTensorDeviceType;
constexpr TVMStructFieldKind kArrKindBound_ = kDLTensorKindBound_;
}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

// CPPMEGA: TileLang-private launch-param tags for SM90+ thread-block clusters.
// These never existed upstream. Add them under the `tvm::runtime::launch_param`
// namespace so the existing TileLang call sites (`runtime::launch_param::*`)
// resolve without modification.
namespace tvm {
namespace runtime {
namespace launch_param {
constexpr const char* kClusterDimX = "tilelang.cluster_dim_x";
constexpr const char* kClusterDimY = "tilelang.cluster_dim_y";
constexpr const char* kClusterDimZ = "tilelang.cluster_dim_z";
}  // namespace launch_param
}  // namespace runtime
}  // namespace tvm

// CPPMEGA: alias attribute keys that moved from `tirx::attr` to `s_tir::attr`
// (software pipelining + async copy markers) and add TileLang-only keys
// (e.g. `tilelang_assume`). Keeping them in `tirx::attr` avoids touching the
// many TileLang call sites that still use the old qualifier.
#include <tvm/s_tir/stmt.h>
namespace tvm {
namespace tirx {
namespace attr {
constexpr const char* software_pipeline_stage = ::tvm::s_tir::attr::software_pipeline_stage;
constexpr const char* software_pipeline_order = ::tvm::s_tir::attr::software_pipeline_order;
constexpr const char* software_pipeline_async_stages = ::tvm::s_tir::attr::software_pipeline_async_stages;
constexpr const char* async_scope = ::tvm::s_tir::attr::async_scope;
constexpr const char* async_commit_queue_scope = ::tvm::s_tir::attr::async_commit_queue_scope;
constexpr const char* async_wait_queue_scope = ::tvm::s_tir::attr::async_wait_queue_scope;
constexpr const char* async_wait_inflight_count = ::tvm::s_tir::attr::async_wait_inflight_count;
constexpr const char* virtual_thread = ::tvm::s_tir::attr::virtual_thread;
constexpr const char* buffer_dim_align = ::tvm::s_tir::attr::buffer_dim_align;
constexpr const char* reduce_scope = ::tvm::s_tir::attr::reduce_scope;
// `volatile_scope` was dropped from apache/tvm; mirror the TileLang vendored
// constant so `tirx::attr::volatile_scope` keeps resolving.
constexpr const char* volatile_scope = "volatile_scope";
// TileLang-private attribute key — never existed upstream.
constexpr const char* tilelang_assume = "tilelang.assume";
}  // namespace attr
}  // namespace tirx
}  // namespace tvm

// CPPMEGA: alias the vendored TileLang-private Allocate node into the
// `tvm::tirx` namespace so the legacy form `Allocate(buffer_var, dtype,
// extents, condition, body, annotations)` keeps resolving across the many
// transform passes that still build the old AST. The IR must be lowered
// (tl.transform.LowerTileLangAllocate) to apache `AllocBuffer` before
// codegen — see `vendored/allocate.h`.
#include "/tmp/tl_apache_tvm_swap/src/transform/vendored/allocate.h"
namespace tvm {
namespace tirx {
using Allocate = ::tilelang::tl_tir::Allocate;
using AllocateNode = ::tilelang::tl_tir::AllocateNode;
}  // namespace tirx
}  // namespace tvm

#endif  // TILELANG_VENDORED_TL_COMPAT_H_
