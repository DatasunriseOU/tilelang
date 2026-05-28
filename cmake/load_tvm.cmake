# todo: support prebuilt tvm

set(TVM_BUILD_FROM_SOURCE TRUE)
set(TVM_SOURCE ${CMAKE_SOURCE_DIR}/3rdparty/tvm)

if(DEFINED ENV{TVM_ROOT})
  if(EXISTS $ENV{TVM_ROOT}/cmake/config.cmake)
    set(TVM_SOURCE $ENV{TVM_ROOT})
    message(STATUS "Using TVM_ROOT from environment variable: ${TVM_SOURCE}")
  endif()
endif()

message(STATUS "Using TVM source: ${TVM_SOURCE}")

set(TVM_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
)

# dmlc-core/include — required by some TVM runtime sources (e.g.
# `src/runtime/cuda/l2_cache_flush.cc` pulls `nvbench/l2_cache_flush.h`,
# which in turn includes `dmlc/logging.h`). The header is part of the
# TVM source tree's standard 3rdparty layout but was being missed by
# the post-codegen-reorg include list. Add it explicitly so CUDA hosts
# build cleanly.
if(EXISTS ${TVM_SOURCE}/3rdparty/dmlc-core/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/dmlc-core/include)
endif()

if(EXISTS ${TVM_SOURCE}/ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/ffi/include)
elseif(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/include)
endif()

if(EXISTS ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
  list(APPEND TVM_INCLUDES ${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
endif()
