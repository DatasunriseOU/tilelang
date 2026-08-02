/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/transform/vendored/lower_allocate.cc
 * \brief Pass that rewrites every TileLang-private `tilelang::tl_tir::Allocate`
 *        into the tirx-equivalent
 *        `SeqStmt({AllocBuffer(buffer), body})` (optionally wrapped in
 *        `IfThenElse(condition, ...)` when the predicate is non-trivial).
 *
 * apache/tvm latest replaced the legacy 6-field
 * `Allocate(buffer_var, dtype, extents, condition, body, annotations)` stmt
 * with the body-less `AllocBuffer(Buffer, annotations)` — the body is now
 * expected to live in the surrounding `SeqStmt`. TileLang preserves the
 * legacy surface area via the vendored `tilelang::tl_tir::Allocate` node so
 * its many call sites compile unchanged. This pass MUST run before any
 * apache/tvm tirx pass that traverses the IR (see tilelang/engine/lower.py),
 * mirroring the architecture of the LetStmt → Bind+SeqStmt converter in
 * `lower_let_stmt.cc`.
 *
 * Registered as `tl.transform.LowerTileLangAllocate`.
 */

#include "allocate.h"

// CPPMEGA: Include let_stmt.h so this converter can pass-through any vendored
// `tilelang::tl_tir::LetStmtNode` it encounters. Although the engine pipeline
// runs `LowerTileLangLetStmt` *before* `LowerTileLangAllocate`, defending
// against a stray un-lowered LetStmt avoids the apache vtable crash
// ("NodeFunctor calls un-registered function on type tilelang.LetStmt") if a
// pass between the two re-introduces one or if the input IR is fed in an
// unexpected order (e.g. tests that invoke this pass in isolation).
#include "let_stmt.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tilelang {
namespace transform {

using tvm::IRModule;
using tvm::PrimExpr;
using tvm::tirx::AllocBuffer;
using tvm::tirx::Bind;
using tvm::tirx::Buffer;
using tvm::tirx::BufferType;
using tvm::tirx::IfThenElse;
using tvm::tirx::PrimFunc;
using tvm::tirx::PrimFuncNode;
using tvm::tirx::SeqStmt;
using tvm::tirx::Stmt;
using tvm::tirx::StmtMutator;
using tvm::transform::Pass;
using tvm::transform::PassContext;

namespace {

/*!
 * \brief Mutator that rewrites every `tilelang::tl_tir::AllocateNode` it
 *        encounters into `SeqStmt({AllocBuffer(buf), body})`.
 *
 * Implementation follows the same trick as `LowerTileLangLetStmtMutator`:
 * apache/tvm tirx's `StmtFunctor` vtable has no entry for
 * `tilelang.Allocate`, so we intercept at the top-level `VisitStmt(const
 * Stmt&)` and short-circuit before the vtable dispatch. Recursion into
 * `body` then re-enters the normal mutator path.
 */
class LowerTileLangAllocateMutator : public StmtMutator {
public:
  bool found_any() const { return found_any_; }

protected:
  Stmt VisitStmt(const Stmt &stmt) override {
    if (const auto *alloc = stmt.as<tl_tir::AllocateNode>()) {
      found_any_ = true;
      // Recurse into body first (bottom-up rewriting); use our own VisitStmt
      // so nested vendored Allocate nodes are also rewritten.
      Stmt body = this->VisitStmt(alloc->body);

      // Build the apache-compatible Buffer object from the vendored
      // Allocate's raw fields. Match the legacy semantics: data_alignment=0
      // and offset_factor=0 let tirx pick defaults; strides=[]/elem_offset
      // unset means dense, no flattening axis-separators.
      Buffer buf(/*data=*/alloc->buffer_var,
                 /*dtype=*/alloc->dtype,
                 /*shape=*/alloc->extents,
                 /*strides=*/{},
                 /*elem_offset=*/PrimExpr(),
                 /*name=*/alloc->buffer_var->name_hint,
                 /*data_alignment=*/0,
                 /*offset_factor=*/0,
                 /*buffer_type=*/BufferType::kDefault,
                 /*axis_separators=*/{},
                 /*span=*/alloc->span);

      // The vendored Allocate's annotations field is already the new
      // `Map<String, Any>` form (see allocate.h), so no conversion needed.
      AllocBuffer alloc_buf(buf, alloc->annotations, alloc->span);

      Stmt seq = SeqStmt({alloc_buf, body}, alloc->span);

      // Only wrap in IfThenElse when the predicate is non-trivial. The
      // legacy Allocate's default condition is `const_true(1)`.
      const PrimExpr &cond = alloc->condition;
      bool trivial = false;
      if (const auto *imm = cond.as<tvm::tirx::IntImmNode>()) {
        trivial = (imm->value != 0);
      }
      if (!trivial) {
        seq = IfThenElse(cond, seq, std::nullopt, alloc->span);
      }
      return seq;
    }
    // CPPMEGA: Pre-emptive pass-through for vendored
    // `tilelang::tl_tir::LetStmtNode`. Apache's `StmtFunctor` vtable does not
    // know this type — falling through to `StmtMutator::VisitStmt` on a
    // LetStmt would route via the vtable and crash with
    // "NodeFunctor calls un-registered function on type tilelang.LetStmt".
    // We rewrite it to the apache-equivalent `SeqStmt({Bind(...), body})`
    // here as a defensive measure (mirrors LowerTileLangLetStmtMutator).
    // Recurse into body via our own VisitStmt so nested vendored Allocates
    // / LetStmts in the body are still handled.
    if (const auto *let = stmt.as<tl_tir::LetStmtNode>()) {
      auto value = this->VisitExpr(let->value);
      auto body = this->VisitStmt(let->body);
      return SeqStmt({Bind(let->var, value, let->span), body}, let->span);
    }
    return StmtMutator::VisitStmt(stmt);
  }

private:
  bool found_any_{false};
};

} // namespace

Pass LowerTileLangAllocate() {
  auto pass_func = [](PrimFunc f, IRModule, PassContext) -> PrimFunc {
    LowerTileLangAllocateMutator mutator;
    Stmt new_body = mutator(std::move(f->body));
    if (!mutator.found_any()) {
      // No-op fast path — leave the function untouched.
      return f;
    }
    auto *node = f.CopyOnWrite();
    node->body = std::move(new_body);
    return f;
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, /*opt_level=*/0, "tl.LowerTileLangAllocate", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerTileLangAllocate",
                        LowerTileLangAllocate);
}

} // namespace transform
} // namespace tilelang
