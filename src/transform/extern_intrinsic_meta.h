/*!
 * \file extern_intrinsic_meta.h
 * \brief Shared TIR helpers for picking up ``tl.extern_intrinsic`` metadata.
 *
 * Integration #9 follow-up. The Python decorator (see
 * ``tilelang/language/extern.py``) emits a TIR ``call_extern`` whose first
 * string argument starts with :c:macro:`kExternCallPrefix` and (optionally)
 * stashes the resolved per-Frag metadata onto the enclosing block's
 * annotations under :c:macro:`kExternBlockAttr`. Two passes need to read this
 * metadata:
 *
 *   - ``layout_inference.cc``: pulls per-buffer fragment layout strings.
 *   - ``inject_pipeline.cc``: pulls the pipeline_stage hint.
 *
 * To avoid copy-pasting the predicate, both passes call the inline helpers
 * defined here.
 */
#ifndef TVM_TL_TRANSFORM_EXTERN_INTRINSIC_META_H_
#define TVM_TL_TRANSFORM_EXTERN_INTRINSIC_META_H_

#include <tvm/ffi/string.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

/*! Symbol prefix used by ``tl.extern_intrinsic`` (see ``extern.py``). */
static constexpr const char *kExternCallPrefix = "tl.extern_intrinsic.";

/*! Block annotation key carrying serialized Frag metadata. */
static constexpr const char *kExternBlockAttr = "tl.extern_intrinsic_meta";

/*!
 * \brief Test whether a TIR ``CallNode`` is a ``tl.extern_intrinsic`` call.
 *
 * The Python decorator emits ``tir.call_extern("handle", "tl.extern_intrinsic.<name>", ...)``;
 * we recognise it via the call_extern op and the symbol prefix on the first
 * string argument.
 */
inline bool IsExternIntrinsicCall(const tirx::CallNode *call) {
  if (call == nullptr) return false;
  if (!call->op.same_as(tvm::tirx::builtin::call_extern())) return false;
  if (call->args.empty()) return false;
  const auto *name_imm = call->args[0].as<tvm::tirx::StringImmNode>();
  if (name_imm == nullptr) return false;
  const std::string &s = name_imm->value;
  const std::string prefix(kExternCallPrefix);
  if (s.length() < prefix.size()) return false;
  return s.compare(0, prefix.size(), prefix) == 0;
}

/*!
 * \brief Look up the ``tl.extern_intrinsic_meta`` annotation on a block.
 *
 * Returns the metadata map if present, ``Optional<>()`` otherwise. We
 * intentionally return ``Map<String, Any>`` (the annotation type used by
 * ``SBlockNode``) so callers don't have to redo the downcast.
 */
inline Optional<ffi::Map<ffi::String, ffi::Any>> GetExternBlockMeta(
    const tirx::SBlockNode *block) {
  using ResultMap = ffi::Map<ffi::String, ffi::Any>;
  if (block == nullptr) return Optional<ResultMap>();
  auto it = block->annotations.find(kExternBlockAttr);
  if (it == block->annotations.end()) {
    return Optional<ResultMap>();
  }
  // Annotation may be stored either as Map<String, Any> directly, or as an
  // Array<Map<String, Any>> (one entry per Frag). Both shapes are accepted —
  // callers re-dispatch as needed.
  if (auto m = (*it).second.template as<ResultMap>()) {
    return m.value();
  }
  return Optional<ResultMap>();
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_EXTERN_INTRINSIC_META_H_
