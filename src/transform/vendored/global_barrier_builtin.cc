// SPDX-License-Identifier: Apache-2.0
//
// TileLang vendored intrinsic: `tirx::builtin::tvm_global_barrier_kinit`.
//
// apache/tvm latest dropped this builtin during the multi-device runtime
// cleanup. The original definition (and its registration) lived in the
// cppmega-mlx-tilelang-stack-c fork at:
//   3rdparty/tvm/include/tvm/tir/builtin.h:532  (declaration)
//   3rdparty/tvm/src/tir/op/builtin.cc:263      (definition + op registration)
//
// We re-introduce both in the original `tvm::tirx::builtin` namespace, using
// the apache-latest `tirx.<name>` op-registry naming convention (see
// 3rdparty/tvm/src/tirx/op/builtin.cc which uses the same `tirx.` prefix and
// the same `TVM_TIR_REGISTER_OP` macro).
//
// The corresponding string symbol `__tvm_prepare_global_barrier` is provided
// alongside this builtin in `tl_runtime_symbols.h` (TileLang-side), to avoid
// patching apache 3rdparty/tvm.
//
// NOTE on runtime backing: the host-side packed function
// `__tvm_prepare_global_barrier` is normally provided by per-device runtime
// modules (e.g. apache CUDA's `CUDAModuleNode::GetFunction`). apache/tvm
// latest no longer provides it. Until a runtime-side replacement lands, any
// kernel that actually emits a global barrier and runs through the apache
// FFI registry will fail with "function __tvm_prepare_global_barrier not
// found". Metal — TileLang's primary backend on this branch — never goes
// through that FFI path, so the lowering is safe in practice. The op exists
// solely so the IR continues to type-check and round-trip.

#include "tl_runtime_symbols.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op.h>            // TVM_TIR_REGISTER_OP
#include <tvm/tirx/op_attr_types.h>  // CallEffectKind, TCallEffectKind

namespace tvm {
namespace tirx {
namespace builtin {

const Op& tvm_global_barrier_kinit() {
  static const Op& op = Op::Get("tirx.tvm_global_barrier_kinit");
  return op;
}

// Mirrors the registration form used in
// 3rdparty/tvm/src/tirx/op/builtin.cc (e.g. tvm_warp_activemask, line ~250):
//   TIR_DEFINE_BUILTIN_FUNC(tvm_global_barrier_kinit)
//       .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
// We expand it manually here (we don't have access to the upstream macro
// from this translation unit, and re-declaring it would shadow the upstream
// version). The macro body is two statements: a `Op::Get` accessor (already
// provided above) and a `TVM_REGISTER_OP("tirx." OpName)` call with a
// `TScriptPrinterName` attr.
TVM_REGISTER_OP("tirx.tvm_global_barrier_kinit")
    .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                  "tvm_global_barrier_kinit")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(static_cast<int>(CallEffectKind::kOpaque)));

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
