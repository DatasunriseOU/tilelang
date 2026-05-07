# Vendored helper for triton-shared integration with the unified fused-kernel compiler.
# Copyright (c) 2026 Project Contributors.
# Original triton-shared sources Copyright (c) Microsoft Corporation and Meta Platforms, Inc.
# Licensed under the MIT License.
#
# =============================================================================
# Usage:
#   cmake -S poc/triton_frontend/vendored/triton_shared              \
#         -B build/triton_shared_vendored                            \
#         -DMLIR_DIR=<llvm-build>/lib/cmake/mlir                     \
#         -DLLVM_DIR=<llvm-build>/lib/cmake/llvm                     \
#         -DTRITON_INSTALL_DIR=<openai-triton-build>                 \
#         -DCMAKE_BUILD_TYPE=Release
#
# Required cache variables (no defaults, must be supplied by the caller):
#   MLIR_DIR           - Directory containing MLIRConfig.cmake.
#   LLVM_DIR           - Directory containing LLVMConfig.cmake (must match
#                        the LLVM build that MLIR was linked against).
#   TRITON_INSTALL_DIR - Root of an OpenAI Triton MLIR build tree. Must
#                        provide:
#                          ${TRITON_INSTALL_DIR}/include/triton/Dialect/Triton/IR/Dialect.h
#                          ${TRITON_INSTALL_DIR}/include/triton/Dialect/Triton/IR/TritonTypes.td
#                          ${TRITON_INSTALL_DIR}/lib/libTritonIR.a (or .so / dylib)
#
# Targets exported:
#   TritonStructuredDialect       - the `tts` MLIR dialect (TableGen +
#                                   TritonStructuredDialect.cpp +
#                                   TritonStructuredOps.cpp).
#   TritonSharedUseAnalysis       - upstream UseAnalysis pass.
#   TritonSharedAnalysisStructured - tts::PtrAnalysis + MaskAnalysis +
#                                   OpFoldResultUtils. This is the target
#                                   downstream pybind modules should link
#                                   against (target_link_libraries(... PRIVATE
#                                   TritonSharedAnalysisStructured)).
# =============================================================================

cmake_minimum_required(VERSION 3.20)
project(triton_shared_vendored CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Required external configuration
# -----------------------------------------------------------------------------
foreach(_var IN ITEMS MLIR_DIR LLVM_DIR TRITON_INSTALL_DIR)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR
      "${_var} must be set on the cmake command line. See the # Usage: block "
      "at the top of this file for details.")
  endif()
endforeach()

if(NOT EXISTS "${TRITON_INSTALL_DIR}/include/triton/Dialect/Triton/IR/TritonTypes.td")
  message(FATAL_ERROR
    "TRITON_INSTALL_DIR='${TRITON_INSTALL_DIR}' does not look like a valid "
    "Triton MLIR build tree (missing TritonTypes.td).")
endif()

# -----------------------------------------------------------------------------
# LLVM + MLIR
# -----------------------------------------------------------------------------
find_package(LLVM REQUIRED CONFIG HINTS "${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG HINTS "${MLIR_DIR}")

message(STATUS "[triton_shared_vendored] Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
message(STATUS "[triton_shared_vendored] Using MLIR from ${MLIR_DIR}")
message(STATUS "[triton_shared_vendored] Using Triton install ${TRITON_INSTALL_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# -----------------------------------------------------------------------------
# Include search paths
# -----------------------------------------------------------------------------
set(_VENDOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(_VENDOR_INCLUDE "${_VENDOR_ROOT}/include")
set(_VENDOR_LIB "${_VENDOR_ROOT}/lib")
set(_VENDOR_BUILD_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/include")

# Make tablegen-generated *.inc files reachable by the same #include path
# used in vendored sources, e.g.
#   #include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"
file(MAKE_DIRECTORY "${_VENDOR_BUILD_INCLUDE}/triton-shared/Dialect/TritonStructured/IR")

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  "${TRITON_INSTALL_DIR}/include"
)
include_directories(
  "${_VENDOR_INCLUDE}"
  "${_VENDOR_BUILD_INCLUDE}"
)

# -----------------------------------------------------------------------------
# TableGen for the TritonStructured dialect
# -----------------------------------------------------------------------------
# The .td imports `triton/Dialect/Triton/IR/TritonTypes.td` from the Triton
# install tree, so we add that as an extra include.
set(LLVM_TARGET_DEFINITIONS
    "${_VENDOR_INCLUDE}/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.td")

# The mlir_tablegen() macro emits into ${CMAKE_CURRENT_BINARY_DIR}, so set it
# up so the outputs land at the correct sub-path.
set(_TS_TG_OUT_DIR "${_VENDOR_BUILD_INCLUDE}/triton-shared/Dialect/TritonStructured/IR")

# Per-call include-dir override (mlir_tablegen reads LLVM_TABLEGEN_FLAGS):
list(APPEND LLVM_TABLEGEN_FLAGS
     "-I${TRITON_INSTALL_DIR}/include"
     "-I${MLIR_INCLUDE_DIRS}"
     "-I${_VENDOR_INCLUDE}")

mlir_tablegen("${_TS_TG_OUT_DIR}/TritonStructuredDialect.h.inc"
              -gen-dialect-decls -dialect=tts)
mlir_tablegen("${_TS_TG_OUT_DIR}/TritonStructuredDialect.cpp.inc"
              -gen-dialect-defs -dialect=tts)
mlir_tablegen("${_TS_TG_OUT_DIR}/TritonStructuredOps.h.inc"
              -gen-op-decls)
mlir_tablegen("${_TS_TG_OUT_DIR}/TritonStructuredOps.cpp.inc"
              -gen-op-defs)
add_public_tablegen_target(TritonStructuredVendoredTableGen)

# -----------------------------------------------------------------------------
# Helper: prefer MLIR's add_mlir_library when available so the resulting
# targets integrate with downstream `find_package(MLIR)` consumers (proper
# install rules, dependency graph, etc). Fall back to plain `add_library`
# in environments that did not load AddMLIR (e.g. pure LLVM, no MLIR cmake
# helpers on the module path). Both paths produce a STATIC library.
# -----------------------------------------------------------------------------
function(_tl_add_vendored_library NAME)
  set(_sources ${ARGN})
  if(COMMAND add_mlir_library)
    add_mlir_library(${NAME}
      ${_sources}
      EXCLUDE_FROM_LIBMLIR
      DISABLE_INSTALL
    )
  else()
    add_library(${NAME} STATIC ${_sources})
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Library: TritonStructuredDialect (the `tts` dialect)
# -----------------------------------------------------------------------------
_tl_add_vendored_library(TritonStructuredDialect
  "${_VENDOR_LIB}/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp"
  "${_VENDOR_LIB}/Dialect/TritonStructured/IR/TritonStructuredOps.cpp"
)
add_dependencies(TritonStructuredDialect TritonStructuredVendoredTableGen)
target_include_directories(TritonStructuredDialect PUBLIC
  "${_VENDOR_INCLUDE}"
  "${_VENDOR_BUILD_INCLUDE}"
  "${TRITON_INSTALL_DIR}/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
# Link MLIR + Triton dialect libs. We rely on TritonIR being importable from
# the Triton install tree.
target_link_libraries(TritonStructuredDialect PUBLIC
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRMemRefDialect
)
# Triton install often ships its dialect lib as `TritonIR`. We add the
# directory to the link search path; the actual symbol resolution happens
# at link time of the final pybind shared object.
target_link_directories(TritonStructuredDialect PUBLIC
  "${TRITON_INSTALL_DIR}/lib"
)
target_link_libraries(TritonStructuredDialect PUBLIC TritonIR)

# -----------------------------------------------------------------------------
# Library: TritonSharedUseAnalysis (upstream UseAnalysis sibling pass)
# -----------------------------------------------------------------------------
_tl_add_vendored_library(TritonSharedUseAnalysis
  "${_VENDOR_LIB}/Analysis/UseAnalysis.cpp"
)
add_dependencies(TritonSharedUseAnalysis TritonStructuredVendoredTableGen)
target_include_directories(TritonSharedUseAnalysis PUBLIC
  "${_VENDOR_INCLUDE}"
  "${_VENDOR_BUILD_INCLUDE}"
  "${TRITON_INSTALL_DIR}/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(TritonSharedUseAnalysis PUBLIC
  TritonStructuredDialect
  MLIRAnalysis
  MLIRPass
  MLIRIR
  MLIRSupport
)

# -----------------------------------------------------------------------------
# Library: TritonSharedAnalysisStructured (PtrAnalysis + MaskAnalysis +
#          OpFoldResultUtils)
# -----------------------------------------------------------------------------
# We point at sources from the already-vendored Phase-1 layout:
#   lib/AnalysisStructured/PtrAnalysis.cpp   <- structured tts::PtrAnalysis
#   lib/Analysis/MaskAnalysis.cpp            <- shared mask decomposition
#   lib/Analysis/OpFoldResultUtils.cpp       <- arithmetic helpers
_tl_add_vendored_library(TritonSharedAnalysisStructured
  "${_VENDOR_LIB}/AnalysisStructured/PtrAnalysis.cpp"
  "${_VENDOR_LIB}/Analysis/MaskAnalysis.cpp"
  "${_VENDOR_LIB}/Analysis/OpFoldResultUtils.cpp"
)
add_dependencies(TritonSharedAnalysisStructured TritonStructuredVendoredTableGen)
target_include_directories(TritonSharedAnalysisStructured PUBLIC
  "${_VENDOR_INCLUDE}"
  "${_VENDOR_BUILD_INCLUDE}"
  "${TRITON_INSTALL_DIR}/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(TritonSharedAnalysisStructured PUBLIC
  TritonStructuredDialect
  TritonSharedUseAnalysis
  MLIRAnalysis
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRTransformUtils
  MLIRIR
  MLIRSupport
)

# -----------------------------------------------------------------------------
# Optional: registration helper (RegisterTritonStructured.cc) — small public
# library so a downstream pybind module can call
# `mlir::triton_shared_vendored::registerTritonStructured(registry)`.
# -----------------------------------------------------------------------------
_tl_add_vendored_library(TritonSharedRegister
  "${_VENDOR_ROOT}/RegisterTritonStructured.cc"
)
add_dependencies(TritonSharedRegister TritonStructuredVendoredTableGen)
target_include_directories(TritonSharedRegister PUBLIC
  "${_VENDOR_ROOT}"
  "${_VENDOR_INCLUDE}"
  "${_VENDOR_BUILD_INCLUDE}"
  "${TRITON_INSTALL_DIR}/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(TritonSharedRegister PUBLIC
  TritonStructuredDialect
  MLIRIR
)

# -----------------------------------------------------------------------------
# Export targets so super-projects can `add_subdirectory(... vendored/triton_shared)`
# and immediately `target_link_libraries(... PRIVATE TritonSharedAnalysisStructured)`.
# -----------------------------------------------------------------------------
set_target_properties(
  TritonStructuredDialect
  TritonSharedUseAnalysis
  TritonSharedAnalysisStructured
  TritonSharedRegister
  PROPERTIES
    EXPORT_NAME ${PROJECT_NAME}::$<TARGET_PROPERTY:NAME>
)

# `EXPORT` keyword on install() is what super-projects use to grab the
# imported targets. Even when consumers use add_subdirectory, the targets
# above are visible by name — install is purely for out-of-tree consumers.
install(TARGETS
  TritonStructuredDialect
  TritonSharedUseAnalysis
  TritonSharedAnalysisStructured
  TritonSharedRegister
  EXPORT triton_shared_vendoredTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(EXPORT triton_shared_vendoredTargets
  FILE triton_shared_vendoredTargets.cmake
  NAMESPACE triton_shared_vendored::
  DESTINATION lib/cmake/triton_shared_vendored
)
install(DIRECTORY "${_VENDOR_INCLUDE}/triton-shared"
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.td")
install(DIRECTORY "${_VENDOR_BUILD_INCLUDE}/triton-shared"
  DESTINATION include
  FILES_MATCHING PATTERN "*.inc")
install(FILES "${_VENDOR_ROOT}/RegisterTritonStructured.h"
  DESTINATION include/triton_shared_vendored)

# -----------------------------------------------------------------------------
# Optional pybind shim: register_triton_structured
#
# The standalone Python extension that exposes
# `mlir::triton_shared_vendored::registerTritonStructured` to
# `verify_dialect_loads.py` and `tests/test_triton_structured_walk.py`. The
# source lives outside this directory at
# `../../_cxx/register_triton_structured_pybind.cc` (Wave-1 added it there
# alongside the existing `_triton_frontend_cxx` pybind module). Build it from
# inside this CMakeLists so the shim ships with the vendoring step rather than
# being entangled with the unrelated PtrAnalysis pybind module.
#
# Gate behind `pybind11` being findable -- when a super-project does not have
# pybind11 available we silently skip (the rest of the vendored libs still
# build and the dialect-walk test auto-skips when the import fails).
# -----------------------------------------------------------------------------
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  set(_REGISTER_PYBIND_SRC
      "${_VENDOR_ROOT}/../../_cxx/register_triton_structured_pybind.cc")
  if(EXISTS "${_REGISTER_PYBIND_SRC}")
    pybind11_add_module(register_triton_structured
      "${_REGISTER_PYBIND_SRC}"
    )
    target_include_directories(register_triton_structured PRIVATE
      "${_VENDOR_ROOT}"
      "${_VENDOR_INCLUDE}"
      ${MLIR_INCLUDE_DIRS}
      ${LLVM_INCLUDE_DIRS}
    )
    target_link_libraries(register_triton_structured PRIVATE
      TritonSharedRegister
    )
    # Place the .so where verify_dialect_loads.py / test_triton_structured_walk.py
    # try to import from: poc/triton_frontend/_cxx/register_triton_structured.<ext>
    set_target_properties(register_triton_structured PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY
        "${_VENDOR_ROOT}/../../_cxx"
    )
    message(STATUS "register_triton_structured: enabled (pybind11 found)")
  else()
    message(STATUS
      "register_triton_structured: source not found at ${_REGISTER_PYBIND_SRC}; "
      "skipping shim.")
  endif()
else()
  message(STATUS
    "register_triton_structured: pybind11 not found; skipping shim "
    "(verify_dialect_loads.py + test_triton_structured_walk.py will auto-skip).")
endif()
