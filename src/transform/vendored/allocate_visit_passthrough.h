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
 */

// CPPMEGA: Pre-emptive helper for the "NodeFunctor calls un-registered
// function on type tilelang.Allocate" drift.
//
// Background
// ----------
// apache/tvm latest's `tirx::StmtFunctor` builds its dispatch vtable from a
// fixed list of built-in `tirx::*Node` types via `set_dispatch<OP>` at
// static-init time, then calls `vtable.Finalize()` (see
// `3rdparty/tvm/include/tvm/ir/node_functor.h`). After Finalize, the vtable
// rejects any further `set_dispatch` calls — which means we **cannot**
// register a pass-through handler for `tilelang::tl_tir::AllocateNode`
// globally on the apache `StmtVisitor` / `StmtMutator` / `StmtFunctor`
// vtables. Any apache pass that traverses IR containing `tilelang.Allocate`
// before `tl.transform.LowerTileLangAllocate` has run will crash with:
//
//   InternalError: Check failed: (can_dispatch(n)) is false:
//     NodeFunctor calls un-registered function on type tilelang.Allocate
//
// The structural fix is the engine-level pass slot (see
// `tilelang/engine/phase.py` — `LowerTileLangAllocate` is already paired
// with `LowerTileLangLetStmt` at every `LowerAndLegalize` /
// `OptimizeForTarget` boundary). However, individual TileLang passes that
// (a) sub-class apache `tirx::StmtVisitor` / `StmtMutator` and
// (b) traverse IR that may still contain `tilelang.Allocate`
// must intercept the vendored type at the top-level `VisitStmt(const Stmt&)`
// override. This helper makes that intercept a one-line call.
//
// Usage (inside a `StmtMutator` subclass)
// ---------------------------------------
//
//   #include "vendored/allocate_visit_passthrough.h"
//   ...
//   Stmt VisitStmt(const Stmt &stmt) override {
//     // Pre-emptive AllocateNode pass-through. Returns std::nullopt when
//     // `stmt` is not a vendored Allocate; returns the rewritten stmt
//     // otherwise.
//     if (auto out = ::tilelang::tl_tir::TryVisitAllocateMutator(this, stmt)) {
//       return *out;
//     }
//     // ... your other vendored-type intercepts (e.g. LetStmt) ...
//     return tvm::tirx::StmtMutator::VisitStmt(stmt);
//   }
//
// And inside a `StmtVisitor` (or `StmtExprVisitor`) subclass:
//
//   void VisitStmt(const Stmt &stmt) override {
//     if (::tilelang::tl_tir::TryVisitAllocateVisitor(this, stmt)) return;
//     // ... your other vendored-type intercepts ...
//     tvm::tirx::StmtVisitor::VisitStmt(stmt);
//   }
//
// The helpers visit the body and (for the mutator variant) reconstruct the
// vendored Allocate with the rewritten body. Annotations / dtype / extents /
// condition / buffer_var are passed through inert. This preserves the
// vendored Allocate node so a downstream `LowerTileLangAllocate` pass can
// still rewrite it into `AllocBuffer + SeqStmt`.

#ifndef TILELANG_TRANSFORM_VENDORED_ALLOCATE_VISIT_PASSTHROUGH_H_
#define TILELANG_TRANSFORM_VENDORED_ALLOCATE_VISIT_PASSTHROUGH_H_

#include <optional>
#include <utility>

#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "allocate.h"

namespace tilelang {
namespace tl_tir {

/*!
 * \brief Pass-through visit for vendored `tilelang::tl_tir::AllocateNode`
 *        from inside a `tirx::StmtMutator` subclass's
 *        `VisitStmt(const Stmt&)` override.
 *
 * If `stmt` is a vendored Allocate, recurses on `body` via
 * `mutator->VisitStmt(body)` and rebuilds the Allocate (only if the body
 * actually changed — preserves COW semantics). All other fields
 * (`buffer_var`, `dtype`, `extents`, `condition`, `annotations`, `span`)
 * are passed through unchanged; the mutator does NOT visit `extents` /
 * `condition` because the apache `StmtMutator::VisitExpr` is private and
 * mutators that need to rewrite expressions inside Allocate already handle
 * the type explicitly via `as<AllocateNode>()` in their `VisitStmt_` chain.
 *
 * \param mutator The calling mutator (typically `this`).
 * \param stmt The statement currently being visited.
 * \return `std::nullopt` if `stmt` is not a vendored Allocate (caller
 *         should fall through to the default vtable dispatch).
 *         Otherwise the rewritten Stmt.
 */
inline std::optional<tvm::tirx::Stmt>
TryVisitAllocateMutator(tvm::tirx::StmtMutator *mutator,
                        const tvm::tirx::Stmt &stmt) {
  const auto *op = stmt.as<AllocateNode>();
  if (op == nullptr) {
    return std::nullopt;
  }
  // Recurse into body; nested vendored Allocates are handled by the same
  // override path on the way down. We use the public `operator()` rather
  // than `VisitStmt` since the latter is protected on `StmtMutator` and we
  // are calling it from a free function. `operator()` momentarily flips
  // `allow_copy_on_write_` to true; that matches the semantics the parent
  // mutator uses internally and is safe at this top-level dispatch point.
  tvm::tirx::Stmt new_body = (*mutator)(op->body);
  if (new_body.same_as(op->body)) {
    return stmt;
  }
  return Allocate(op->buffer_var, op->dtype, op->extents, op->condition,
                  std::move(new_body), op->annotations, op->span);
}

/*!
 * \brief Pass-through visit for vendored `tilelang::tl_tir::AllocateNode`
 *        from inside a `tirx::StmtVisitor` (or `StmtExprVisitor`) subclass's
 *        `VisitStmt(const Stmt&)` override.
 *
 * If `stmt` is a vendored Allocate, recurses on `body` via
 * `visitor->VisitStmt(body)`. Other fields are inert from a
 * statement-traversal perspective.
 *
 * \param visitor The calling visitor (typically `this`).
 * \param stmt The statement currently being visited.
 * \return `true` if the stmt was a vendored Allocate (caller should NOT
 *         fall through to the default vtable dispatch).
 *         `false` otherwise.
 */
inline bool TryVisitAllocateVisitor(tvm::tirx::StmtVisitor *visitor,
                                    const tvm::tirx::Stmt &stmt) {
  const auto *op = stmt.as<AllocateNode>();
  if (op == nullptr) {
    return false;
  }
  visitor->operator()(op->body);
  return true;
}

}  // namespace tl_tir
}  // namespace tilelang

#endif  // TILELANG_TRANSFORM_VENDORED_ALLOCATE_VISIT_PASSTHROUGH_H_
