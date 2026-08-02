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
 * \file vendored/target_info.cc
 * \brief CPPMEGA: vendored from TileLang fork of apache/tvm.
 * Apache/tvm latest dropped target_info.h entirely; TileLang's storage
 * passes still call GetMemoryInfo(scope_str) to look up per-scope memory
 * descriptors registered via "tvm.info.mem.<scope>" FFI globals.
 *
 * No ReprPrinter dispatcher is registered (apache removed ReprPrinter).
 */
#include "target_info.h"

#include <tvm/ffi/function.h>
#include <tvm/runtime/logging.h>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() { MemoryInfoNode::RegisterReflection(); }

MemoryInfo GetMemoryInfo(const std::string &scope) {
  std::string fname = "tvm.info.mem." + scope;
  const auto f = tvm::ffi::Function::GetGlobal(fname);
  if (!f.has_value()) {
    LOG(WARNING) << "MemoryInfo for scope = " << scope << " is undefined";
    return MemoryInfo();
  } else {
    return (*f)().cast<MemoryInfo>();
  }
}

} // namespace tvm
