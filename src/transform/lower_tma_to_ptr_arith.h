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
 * \file lower_tma_to_ptr_arith.h
 * \brief Decompose Hopper-style TMA descriptor loads/stores into explicit
 *        pointer-arith copy loops on non-Hopper targets (Apple Metal,
 *        AMD HIP, pre-Hopper CUDA, CPU). NV Hopper+ paths are passed
 *        through unchanged.
 */
#ifndef TVM_TL_TRANSFORM_LOWER_TMA_TO_PTR_ARITH_H_
#define TVM_TL_TRANSFORM_LOWER_TMA_TO_PTR_ARITH_H_

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief Pass that rewrites `tl::tma_load` / `tl::tma_store` /
 *        `tl::tma_load_im2col` and the surrounding
 *        `tl::create_tma_descriptor` plumbing into target-portable
 *        pointer-arith copy loops on non-Hopper targets.
 *
 * On NV Hopper+ targets the pass is a no-op (the existing
 * `LowerHopperIntrin` pass owns the lowering). On Metal / HIP / pre-Hopper
 * CUDA / CPU the pass walks each TMA call, recovers
 * `(global_ptr_base, stride_per_dim, shape_per_dim)` from the descriptor's
 * argument list (see `TMADesc::EncodeCallArgs` in `src/op/copy.cc`), and
 * emits a tile-shaped `For` nest that lowers to plain `BufferLoad` /
 * `BufferStore` against the staging shared buffer. Surrounding
 * `tma_store_arrive` / `tma_store_wait` markers are dropped because the
 * resulting copies are synchronous.
 *
 * \return The pass.
 */
tvm::transform::Pass LowerTMAToPtrArith();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_LOWER_TMA_TO_PTR_ARITH_H_
