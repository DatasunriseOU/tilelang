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

/*!
 * \file src/transform/vendored/allocate.h
 * \brief Vendored, TileLang-internal `Allocate` IR node mirroring the legacy
 *        apache/tvm `tirx::AllocateNode` API.
 *
 * apache/tvm latest replaced `Allocate(buffer_var, dtype, extents, condition,
 * body, annotations)` with the body-less `AllocBuffer(Buffer, annotations)`
 * stmt — the body is now expected to live in the surrounding SeqStmt. TileLang
 * has many vendored transforms (lower_thread_allreduce, vectorize_loop,
 * merge_shared_memory_allocations, storage_rewrite, ...) that construct the
 * old form with body.  Rather than rewriting every site, we vendor a
 * TileLang-private node that keeps the old surface area. A lowering pass
 * (`tl.transform.LowerTileLangAllocate`) MUST be run before the IR is handed
 * back to apache/tvm — it converts each `tilelang.Allocate` into
 * `SeqStmt({AllocBuffer(buf), DeclBuffer(buf), body})`.
 */
#ifndef TILELANG_TRANSFORM_VENDORED_ALLOCATE_H_
#define TILELANG_TRANSFORM_VENDORED_ALLOCATE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/var.h>

#include <utility>

namespace tilelang {
namespace tl_tir {

using tvm::DataType;
using tvm::PrimExpr;
using tvm::Span;
using tvm::tirx::Stmt;
using tvm::tirx::StmtNode;
using tvm::tirx::Var;

class AllocateNode : public StmtNode {
public:
  /*! \brief Variable that holds the pointer to the allocation. */
  Var buffer_var;
  /*! \brief Element data type. */
  DataType dtype;
  /*! \brief Per-axis extents. */
  tvm::ffi::Array<PrimExpr> extents;
  /*! \brief Predicate (condition under which the allocation is live). */
  PrimExpr condition;
  /*! \brief The body of the allocation scope. */
  Stmt body;
  /*! \brief Optional pass-through annotations. */
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations;

  /*! \brief Returns the constant total allocation size if all extents are
   *  IntImm, otherwise -1. */
  int64_t ConstantAllocationSize() const {
    int64_t result = 1;
    for (const PrimExpr &dim : extents) {
      if (const auto *imm = dim.as<tvm::tirx::IntImmNode>()) {
        result *= imm->value;
      } else {
        return -1;
      }
    }
    return result;
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllocateNode>()
        .def_ro("buffer_var", &AllocateNode::buffer_var,
                refl::AttachFieldFlag::SEqHashDef())
        .def_ro("dtype", &AllocateNode::dtype)
        .def_ro("extents", &AllocateNode::extents)
        .def_ro("condition", &AllocateNode::condition)
        .def_ro("body", &AllocateNode::body)
        .def_ro("annotations", &AllocateNode::annotations);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tilelang.Allocate", AllocateNode,
                                    StmtNode);
};

class Allocate : public Stmt {
public:
  TVM_DLL Allocate(Var buffer_var, DataType dtype,
                   tvm::ffi::Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body,
                   tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations =
                       tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>(),
                   Span span = Span());

  /*! \brief Returns the constant total allocation size if all extents are
   *  IntImm, otherwise -1. Delegates to the node method to avoid
   *  code duplication (BUG-VIR-2 fix). */
  int64_t ConstantAllocationSize() const {
    return (*this)->ConstantAllocationSize();
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Allocate, Stmt, AllocateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AllocateNode);
};

} // namespace tl_tir
} // namespace tilelang

#endif // TILELANG_TRANSFORM_VENDORED_ALLOCATE_H_
