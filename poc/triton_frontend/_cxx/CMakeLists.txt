# =====================================================================
# poc/triton_frontend/_cxx/CMakeLists.txt
#
# Builds the `_triton_frontend_cxx` Python extension that drives
# `mlir::tts::PtrAnalysis::rewriteOp` on a parsed MLIR module.
#
# REQUIRED ENVIRONMENT VARIABLES (pass via `cmake -D...` or `export`):
#   MLIR_DIR            CMake package dir for an MLIR install
#                       (e.g. /opt/llvm/lib/cmake/mlir)
#   LLVM_DIR            CMake package dir for the matching LLVM install
#                       (e.g. /opt/llvm/lib/cmake/llvm)
#   TRITON_INSTALL_DIR  Prefix containing the upstream Triton install
#                       (must expose include/triton/Dialect/Triton/IR/Dialect.h
#                       and lib/libTritonIR.a -- TODO integration #5).
#
# The vendored microsoft/triton-shared sources live alongside this file at
# ../vendored/triton_shared/ and are compiled in directly (no separate
# install step).
# =====================================================================

cmake_minimum_required(VERSION 3.20)
project(triton_frontend_cxx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optional: swap the hand-rolled RFC-8259 JSON escaper in
# tl_pa_extract_states_json for nlohmann::json. The two encoders MUST emit
# byte-identical output for the existing minimal `[{"op":"..."}]` schema; the
# regression test in tests/test_ptr_analysis.py guards that contract. When
# enabled, the build expects a vendored single-include header at
# third_party/nlohmann/json.hpp (see third_party/nlohmann/README for fetch
# instructions).
option(TRITON_FRONTEND_USE_NLOHMANN_JSON
       "Use nlohmann::json instead of the hand-rolled RFC-8259 escaper" OFF)

# ---- LLVM / MLIR -----------------------------------------------------

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIR_DIR  = ${MLIR_DIR}")
message(STATUS "Using LLVM_DIR  = ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# ---- pybind11 --------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ---- Triton (upstream) ------------------------------------------------
# When TRITON_INSTALL_DIR is set, locate the Triton headers + libraries so the
# vendored PtrAnalysis.cpp can resolve `triton::AddPtrOp`, `triton::MakeRangeOp`,
# etc. Without it we still configure (C-ABI builds) but `tl_pa_run_rewrite`
# returns TL_PA_ERR_INTERNAL because the `__has_include` gates in
# ptr_analysis_shim.cc disable the rewrite path.
set(TL_PA_LINK_LIBS "")
if(DEFINED ENV{TRITON_INSTALL_DIR} OR DEFINED TRITON_INSTALL_DIR)
  if(NOT DEFINED TRITON_INSTALL_DIR)
    set(TRITON_INSTALL_DIR "$ENV{TRITON_INSTALL_DIR}")
  endif()
  message(STATUS "Using TRITON_INSTALL_DIR = ${TRITON_INSTALL_DIR}")
  include_directories(${TRITON_INSTALL_DIR}/include)
  link_directories(${TRITON_INSTALL_DIR}/lib)
  # The Triton dialect splits across a handful of static libs; PtrAnalysis.cpp
  # references symbols from at least TritonIR + TritonAnalysis. Use REQUIRED
  # for the core library and treat the rest as best-effort so older Triton
  # checkouts still link.
  find_library(TRITON_IR_LIB
    NAMES TritonIR
    PATHS ${TRITON_INSTALL_DIR}/lib
    NO_DEFAULT_PATH
    REQUIRED)
  list(APPEND TL_PA_LINK_LIBS ${TRITON_IR_LIB})
  foreach(_optlib TritonAnalysis TritonTransforms TritonGPUIR)
    find_library(TRITON_${_optlib}_LIB
      NAMES ${_optlib}
      PATHS ${TRITON_INSTALL_DIR}/lib
      NO_DEFAULT_PATH)
    if(TRITON_${_optlib}_LIB)
      list(APPEND TL_PA_LINK_LIBS ${TRITON_${_optlib}_LIB})
    endif()
  endforeach()
else()
  message(WARNING
    "TRITON_INSTALL_DIR not set: building C-ABI surface only; "
    "tl_pa_run_rewrite will return TL_PA_ERR_INTERNAL until integration #5.")
endif()

# ---- Vendored triton-shared sources ----------------------------------

set(VENDORED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendored/triton_shared")
include_directories(${VENDORED_DIR}/include)

set(VENDORED_SOURCES
  ${VENDORED_DIR}/lib/AnalysisStructured/PtrAnalysis.cpp
)

# ---- Library + Python module ----------------------------------------

list(APPEND TL_PA_LINK_LIBS
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRAffineDialect
  MLIRTransforms
)

pybind11_add_module(_triton_frontend_cxx
  ptr_analysis_shim.cc
  pybind_module.cc
  ${VENDORED_SOURCES}
)
target_include_directories(_triton_frontend_cxx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${VENDORED_DIR}/include
)
target_link_libraries(_triton_frontend_cxx PRIVATE ${TL_PA_LINK_LIBS})

if(TRITON_FRONTEND_USE_NLOHMANN_JSON)
  set(_NLOHMANN_HDR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/json.hpp")
  if(NOT EXISTS "${_NLOHMANN_HDR}")
    message(FATAL_ERROR
      "TRITON_FRONTEND_USE_NLOHMANN_JSON=ON but ${_NLOHMANN_HDR} is missing. "
      "Vendor the single-include header by running:\n"
      "  curl -L -o ${_NLOHMANN_HDR} \\\n"
      "       https://github.com/nlohmann/json/releases/latest/download/json.hpp")
  endif()
  target_include_directories(_triton_frontend_cxx PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
  target_compile_definitions(_triton_frontend_cxx PRIVATE TL_PA_USE_NLOHMANN_JSON=1)
  message(STATUS "PtrAnalysis JSON encoder: nlohmann::json")
else()
  message(STATUS "PtrAnalysis JSON encoder: hand-rolled RFC-8259")
endif()

set_target_properties(_triton_frontend_cxx PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
