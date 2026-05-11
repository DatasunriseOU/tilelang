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
 * \file codegen_metal.h
 * \brief Generate Metal device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_METAL_H_
#define TVM_TARGET_SOURCE_CODEGEN_METAL_H_

#include <tvm/target/codegen.h>

#include <set>
#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangMetal final : public CodeGenC {
public:
  explicit CodeGenTileLangMetal(Target target);
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const GlobalVar &gvar, const PrimFunc &func) final;
  void InitFuncState(const PrimFunc &f) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final;     // NOLINT(*)
  void PrintStorageSync(const CallNode *op) final;    // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void BindThreadIndex(const IterVar &iv) final;      // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  // overload visitor
  // CPPMEGA: drop `final` — `AllocateNode` here resolves to the vendored
  // `tilelang::tl_tir::AllocateNode`, so this overload does NOT actually
  // override a virtual base method. With `final` on a non-virtual function,
  // clang errors that it hides the apache base virtuals (WhileNode/IfThenElse
  // etc. all in codegen_c.h). The vendored type is dispatched via the short-
  // circuit `VisitStmt(const Stmt&)` pattern established elsewhere.
  void VisitStmt_(const AllocateNode *op);          // NOLINT(*)
  void VisitStmt_(const AllocBufferNode *op) final; // NOLINT(*)
  void VisitStmt_(const AttrStmtNode *op) final;    // NOLINT(*)
  void VisitStmt_(const ForNode *op) final;         // NOLINT(*)
  void VisitStmt_(const BufferStoreNode *op) final; // NOLINT(*)
  void VisitExpr_(const BufferLoadNode *op,
                  std::ostream &os) final;                          // NOLINT(*)
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;    // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;  // NOLINT(*)
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;

  // reuse parent's function.
  using CodeGenC::PrintType;

private:
  // CPPMEGA: hybrid tl_pr_c granularity + stack-c switch dispatch.
  // Emit FP8/FP4 prelude helpers only for dtypes actually referenced by the
  // kernel body (not unconditionally for all FP8 variants). Without this,
  // every Metal kernel pays ~1KB+ of dead helper code even when it is pure
  // float32. The helper-emission set is populated by a pre-walker that scans
  // the PrimFunc body in `AddFunction` before the body is printed, and
  // consumed by `EmitFPHelperPrelude` which switch-dispatches into the
  // per-dtype emitters below. New FP8 dtypes plug in by adding a switch case
  // and a matching per-dtype `EmitFp8XXXHelper()` body. See
  // docs/mlx_port_master_plan.md (Metal codegen FP8 conditional prelude).
  void CollectReferencedLowPrecisionDtypes(const PrimFunc &f);
  void EmitFPHelperPrelude();          // public dispatch entry
  void EmitAtomicAddHelperPrelude();
  void EmitFp8E3M4Helper();
  void EmitFp8E4M3Helper();
  void EmitFp8E4M3FnAliasHelper();     // delegates to E4M3
  void EmitFp8E4M3FnuzHelper();
  void EmitFp8E4M3B11FnuzHelper();
  void EmitFp8E5M2Helper();
  void EmitFp8E5M2FnuzHelper();
  void EmitFp8E8M0FnuHelper();
  void EmitFp8Dot4Helpers();           // LUT + dot4_words + dot4_packed overloads

  std::set<int> referenced_fp8_codes_;
  bool uses_fp8_dot4_{false};
  bool uses_atomic_add_{false};
  bool emitted_atomic_add_helper_{false};

  std::unordered_map<const VarNode *, std::string> simdgroup_dtype_;
  std::unordered_map<const VarNode *, IntImm> unroll_factor_;
  int thread_index_bits_{32};
  int thread_work_dim_{0};
  Target target_;
};
} // namespace codegen
} // namespace tvm

#endif // TVM_TARGET_SOURCE_CODEGEN_METAL_H_
