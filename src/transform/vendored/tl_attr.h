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
 * \file src/transform/vendored/tl_attr.h
 * \brief TileLang-local attribute key constants vendored from legacy
 * apache/tvm `tirx::attr` namespace. Apache renamed `tir` -> `tirx`/`s_tir`
 * during the migration and dropped some attribute keys (e.g.
 * `volatile_scope`). TileLang transforms still rely on these markers, so
 * we vendor them here to avoid scattering string literals throughout the
 * codebase.
 */
#ifndef TILELANG_TRANSFORM_VENDORED_TL_ATTR_H_
#define TILELANG_TRANSFORM_VENDORED_TL_ATTR_H_

namespace tilelang {
namespace tl_attr {

/*! \brief Mark the scope as volatile access for certain handle. */
constexpr const char* volatile_scope = "volatile_scope";

/*! \brief Mark a scope as a pipeline execution stage. */
constexpr const char* pipeline_exec_scope = "pipeline_exec_scope";

/*! \brief Mark a scope as coprocessor micro-op scope (used by
 * legacy CombineContextCall pass). */
constexpr const char* coproc_uop_scope = "coproc_uop_scope";

}  // namespace tl_attr
}  // namespace tilelang

#endif  // TILELANG_TRANSFORM_VENDORED_TL_ATTR_H_
