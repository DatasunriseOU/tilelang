/*!
 * \file intrin_rule_metal.cc
 * \brief TileLang Metal intrinsic rules.
 *
 * Registers the ``tirx.metal.*`` ops emitted by the FP8 dot4 vecmat path
 * (`tilelang/language/fp8_op.py`).  Without these registrations, the Python
 * call sites raise ``Operator ... is not registered`` (auto-promote skips
 * the dot4 path) and, even after registration, codegen would throw
 * ``InternalError: Unresolved call ir.Op tirx.metal.fp8_e4m3_dot4`` because
 * neither a TGlobalSymbol nor an FLowerIntrinsic attribute was attached.
 *
 * The ``__tvm_fp8_e4m3_dot4_*`` C symbols are provided by
 * ``CodeGenTileLangMetal::EmitFp8E4M3Helper`` (see codegen_metal.cc) which
 * emits the LUT-decoded helper prelude when ``uses_fp8_dot4_`` is set.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "../support/ffi_aliases.h"
#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;
using tirx::TGlobalSymbol;
using tirx::TCallEffectKind;
using tirx::CallEffectKind;
using tirx::TScriptPrinterName;

// FP8 E4M3 packed dot4 — 4-byte packed FP8 dot product.
// The Metal codegen (`CodeGenTileLangMetal::EmitFp8E4M3Helper`) emits an
// overload set named ``__tvm_fp8_e4m3_dot4_packed`` that accepts
// (uchar* a, uchar* b, uint a_word_idx, uint b_word_idx) across
// device / threadgroup / constant address-space combinations.
//
// TScriptPrinterName is required so the TVMScript printer can render
// ``T.<name>(...)`` for this op when it appears in IR dumps and error
// messages. Without it, the printer falls back to a basic-address printer
// and emits a "No TScriptPrinterName attribute" warning that escalates
// inside TVM_FFI_ICHECK on certain pass-error paths.
TVM_REGISTER_OP("tirx.metal.fp8_e4m3_dot4")
    .set_num_inputs(4)
    .add_argument("a_ptr", "Expr", "Pointer to packed FP8 e4m3 byte buffer A.")
    .add_argument("b_ptr", "Expr", "Pointer to packed FP8 e4m3 byte buffer B.")
    .add_argument("a_word_idx", "Expr", "uint32 word index into A (4 bytes per word).")
    .add_argument("b_word_idx", "Expr", "uint32 word index into B (4 bytes per word).")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__tvm_fp8_e4m3_dot4_packed")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_fp8_e4m3_dot4");

TVM_REGISTER_OP("tirx.metal.fp8_load_u32")
    .set_num_inputs(2)
    .add_argument("ptr", "Expr", "Pointer to packed FP8 byte or uint32 buffer.")
    .add_argument("word_idx", "Expr", "uint32 word index (4 bytes per word).")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__tvm_fp8_load_u32")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_fp8_load_u32");

TVM_REGISTER_OP("tirx.metal.fp8_e4m3_dot4_words")
    .set_num_inputs(2)
    .add_argument("a_word", "Expr", "Packed uint32 word from FP8 e4m3 buffer A.")
    .add_argument("b_word", "Expr", "Packed uint32 word from FP8 e4m3 buffer B.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__tvm_fp8_e4m3_dot4_words")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_fp8_e4m3_dot4_words");

// Metal thread-position / SIMD-lane intrinsics used by the dot4 vecmat macro.
// MSL exposes these as the ``thread_position_in_grid`` / ``thread_index_in_simdgroup``
// kernel-attribute identifiers; codegen passes them through verbatim via TGlobalSymbol.
TVM_REGISTER_OP("tirx.metal.thread_position_in_grid_x")
    .set_num_inputs(0)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "thread_position_in_grid_x")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_thread_position_in_grid_x");

TVM_REGISTER_OP("tirx.metal.thread_position_in_threadgroup_x")
    .set_num_inputs(0)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "thread_position_in_threadgroup_x")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_thread_position_in_threadgroup_x");

TVM_REGISTER_OP("tirx.metal.thread_index_in_simdgroup")
    .set_num_inputs(0)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "thread_index_in_simdgroup")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_thread_index_in_simdgroup");

TVM_REGISTER_OP("tirx.metal.simd_sum")
    .set_num_inputs(1)
    .add_argument("value", "Expr", "Value to reduce with Metal simd_sum.")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "metal_simd_sum");

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
