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

#ifndef TILELANG_VENDORED_Z3_PROOF_HOOKS_H_
#define TILELANG_VENDORED_Z3_PROOF_HOOKS_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>

#include <cstdlib>
#include <string>

namespace tilelang {
namespace tlz3 {

enum class ProofHookKind {
  kVectorization,
  kBarrierMinimization,
  kAsyncEligibility,
  kAliasShape,
};

struct ProofHookStatus {
  bool enabled{false};
  const char* config_key{nullptr};
  const char* env_pass_name{nullptr};
  const char* fallback_reason{nullptr};
};

inline constexpr const char* kProofHookVectorization =
    "tl.z3_proof.vectorization";
inline constexpr const char* kProofHookBarrierMinimization =
    "tl.z3_proof.barrier_minimization";
inline constexpr const char* kProofHookAsyncEligibility =
    "tl.z3_proof.async_eligibility";
inline constexpr const char* kProofHookAliasShape = "tl.z3_proof.alias_shape";

inline const char* ProofHookConfigKey(ProofHookKind kind) {
  switch (kind) {
    case ProofHookKind::kVectorization:
      return kProofHookVectorization;
    case ProofHookKind::kBarrierMinimization:
      return kProofHookBarrierMinimization;
    case ProofHookKind::kAsyncEligibility:
      return kProofHookAsyncEligibility;
    case ProofHookKind::kAliasShape:
      return kProofHookAliasShape;
  }
  return "";
}

inline const char* ProofHookEnvPassName(ProofHookKind kind) {
  switch (kind) {
    case ProofHookKind::kVectorization:
      return "VECTORIZE";
    case ProofHookKind::kBarrierMinimization:
      return "BARRIER_ELISION";
    case ProofHookKind::kAsyncEligibility:
      return "TMA_LEGALITY";
    case ProofHookKind::kAliasShape:
      return "ALIAS_SHAPE";
  }
  return "";
}

inline bool GetBoolPassConfig(const tvm::transform::PassContext& ctx,
                              const char* key) {
  if (!ctx.defined() || key == nullptr || key[0] == '\0') {
    return false;
  }
  return ctx->GetConfig<tvm::Bool>(key, tvm::Bool(false)).value()->value;
}

inline bool EnvProofGateEnabled(const char* pass_name) {
  const char* global = std::getenv("TILELANG_DISABLE_Z3");
  if (global != nullptr && global[0] != '\0' && global[0] != '0') {
    return false;
  }
  std::string key("TILELANG_DISABLE_Z3_");
  key += pass_name;
  const char* local = std::getenv(key.c_str());
  return !(local != nullptr && local[0] != '\0' && local[0] != '0');
}

inline ProofHookStatus GetProofHookStatus(
    const tvm::transform::PassContext& ctx, ProofHookKind kind,
    const char* legacy_config_key = nullptr,
    const char* env_pass_name_override = nullptr) {
  const char* config_key = ProofHookConfigKey(kind);
  const char* env_pass_name =
      env_pass_name_override ? env_pass_name_override : ProofHookEnvPassName(kind);

  bool config_enabled = GetBoolPassConfig(ctx, config_key);
  if (!config_enabled && legacy_config_key != nullptr) {
    config_enabled = GetBoolPassConfig(ctx, legacy_config_key);
  }
  if (!config_enabled) {
    return {false, config_key, env_pass_name, "proof_disabled"};
  }
  if (!EnvProofGateEnabled(env_pass_name)) {
    return {false, config_key, env_pass_name, "env_disabled"};
  }
  return {true, config_key, env_pass_name, ""};
}

}  // namespace tlz3
}  // namespace tilelang

#endif  // TILELANG_VENDORED_Z3_PROOF_HOOKS_H_
