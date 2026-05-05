// SPDX-License-Identifier: Apache-2.0
//
// TileLang vendored runtime/symbol shim for apache/tvm-latest.
//
// apache/tvm latest dropped `tvm_prepare_global_barrier` (and friends) from
// `tvm/runtime/module.h` as part of the multi-device runtime cleanup. The
// previous TileLang fork (cppmega-mlx-tilelang-stack-c) still kept these
// constants in `runtime/symbol`. They are referenced by
// `src/transform/thread_storage_sync.cc` (cross-block global-barrier path)
// and by `src/target/codegen_cuda.cc` (which is currently NOT built in the
// Metal-only configuration but kept around for future CUDA targets).
//
// We re-introduce the constants in the original `tvm::runtime::symbol`
// namespace so the call sites compile unchanged. The runtime backing for
// `__tvm_prepare_global_barrier` lives inside the per-device `Module`
// implementations (e.g. CUDA, ROCm). Apache no longer ships those, so any
// kernel that emits a global barrier will fail at *runtime* with a missing
// FFI function — but the IR compiles, codegen runs, and Metal kernels (the
// only target that actually goes through this lowering today) never request
// `tvm_prepare_global_barrier` at runtime.
#ifndef TILELANG_VENDORED_TL_RUNTIME_SYMBOLS_H_
#define TILELANG_VENDORED_TL_RUNTIME_SYMBOLS_H_

#include <tvm/runtime/device_api.h>  // pulls in `namespace runtime { namespace symbol { ... } }`

namespace tvm {
namespace runtime {
namespace symbol {

// Auxiliary counter to global barrier (lives in device-side global memory).
// Matches the original constant value from
// /private/tmp/cppmega-mlx-tilelang-stack-c/3rdparty/tvm/include/tvm/runtime/module.h:55
#ifndef TILELANG_HAVE_TVM_GLOBAL_BARRIER_STATE
#define TILELANG_HAVE_TVM_GLOBAL_BARRIER_STATE
constexpr const char* tvm_global_barrier_state = "__tvm_global_barrier_state";
#endif

// Name of the host-side packed function that initialises the device-side
// barrier counter. Matches the original constant value from
// /private/tmp/cppmega-mlx-tilelang-stack-c/3rdparty/tvm/include/tvm/runtime/module.h:57
#ifndef TILELANG_HAVE_TVM_PREPARE_GLOBAL_BARRIER
#define TILELANG_HAVE_TVM_PREPARE_GLOBAL_BARRIER
constexpr const char* tvm_prepare_global_barrier = "__tvm_prepare_global_barrier";
#endif

}  // namespace symbol
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace tirx {
namespace builtin {

/*!
 * \brief Initialize the global barrier.
 *
 * Vendored back from the cppmega-mlx-tilelang-stack-c fork
 * (3rdparty/tvm/include/tvm/tir/builtin.h:532). Emitted by
 * `thread_storage_sync.cc::InitGlobalBarrier` when a kernel needs a
 * cross-block global barrier. Implemented in
 * `src/transform/vendored/global_barrier_builtin.cc`.
 */
TVM_DLL const Op& tvm_global_barrier_kinit();

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

#endif  // TILELANG_VENDORED_TL_RUNTIME_SYMBOLS_H_
