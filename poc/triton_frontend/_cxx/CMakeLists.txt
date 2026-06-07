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
# C is enabled in addition to CXX because LLVM's HandleLLVMOptions.cmake (loaded
# below from ${LLVM_CMAKE_DIR}) calls `check_c_compiler_flag(...)` while
# probing host-compiler features. Without `C` in LANGUAGES, configure aborts
# with "check_compiler_flag: C: needs to be enabled before use" on cmake >=3.27.
project(triton_frontend_cxx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Stub build: compile only the C ABI + pybind shim so the resulting Python
# extension is *loadable*, but `tl_pa_run_rewrite` returns TL_PA_ERR_INTERNAL.
# Auto-enabled whenever the Triton dialect headers are not findable (i.e.
# TRITON_INSTALL_DIR is unset). Useful on Mac developer boxes where building
# upstream Triton is heavy and we just want the import path to work.
option(TRITON_FRONTEND_STUB_BUILD
       "Skip vendored PtrAnalysis sources; build C ABI + pybind shim only" OFF)

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
# Allow callers (typically build_cxx.py) to override TRITON_INSTALL_DIR via
# `-DTRITON_INSTALL_DIR=...`. The vendored fallback path under
# `_cxx/../vendored/triton` ships only stub archives (no architecture, empty
# members) suitable for header-only configure-time use; for an actual link the
# caller must supply a real Triton build/install dir whose `lib/` contains
# native `libTritonIR.a` etc. `build_cxx.py::_stage_triton_install` does this
# automatically by harvesting `.o` files from a local triton source checkout.
if(NOT DEFINED TRITON_INSTALL_DIR OR TRITON_INSTALL_DIR STREQUAL "")
  set(TRITON_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendored/triton")
endif()
message(STATUS "Using TRITON_INSTALL_DIR = ${TRITON_INSTALL_DIR}")
include_directories(${TRITON_INSTALL_DIR}/include)
link_directories(${TRITON_INSTALL_DIR}/lib)

# ---------------------------------------------------------------------------
# Triton MLIR dialect symbols: link against the *installed* triton wheel's
# `libtriton.so` (one shared object, the SAME binary the triton runtime loads).
#
# Why not the static `libTritonIR.a` etc.?  The core `tt` dialect
# (`mlir::triton::TritonDialect`) is NOT self-contained: `libTritonIR.a`
# references `mlir::triton::gpu::MemDescType`'s TypeID (lives in TritonGPUIR),
# TritonGPUIR in turn references `mlir::inferDstEncoding` (TritonGPUTransforms,
# which cannot build standalone without the nvidia-backend NVWS TableGen),
# `mlir::triton::ml(LinearLayout,...)` (TritonTools), and the NvidiaGPU
# `TensorMemoryEncodingAttr` TypeID. Stitching that whole web from individually
# `ar`-archived object dirs is brittle AND risks *duplicate* MLIR TypeID
# storage (two `TritonDialect` TypeIDs -> registry mismatch at runtime).
#
# The triton wheel already ships the complete, internally-consistent closure in
# `triton/_C/libtriton.so` with default-visibility (global `T`) exports for
# every symbol the shim + vendored triton-shared need (verified: of the Triton
# symbols our objects reference, the only two NOT exported by libtriton.so are
# the vendored `tts` functions we build ourselves). Linking it gives a single
# TypeID source identical to the runtime's, so dialect registration is
# guaranteed consistent. The staged `${TRITON_INSTALL_DIR}/include` headers are
# still required for *compilation* (the wheel ships no MLIR headers).
find_library(TRITON_RUNTIME_SO
  NAMES triton libtriton
  PATHS ${TRITON_INSTALL_DIR}/lib ${TRITON_RUNTIME_LIBDIR}
  NO_DEFAULT_PATH)
if(NOT TRITON_RUNTIME_SO)
  # Locate triton/_C/libtriton.so from the active Python's triton install.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import triton,os,sys;p=os.path.join(os.path.dirname(triton.__file__),'_C','libtriton.so');print(p if os.path.exists(p) else '',end='')"
    OUTPUT_VARIABLE _TRITON_SO_PATH)
  if(_TRITON_SO_PATH AND EXISTS "${_TRITON_SO_PATH}")
    set(TRITON_RUNTIME_SO "${_TRITON_SO_PATH}")
  endif()
endif()
if(NOT TRITON_RUNTIME_SO)
  message(FATAL_ERROR
    "Could not locate libtriton.so. Set -DTRITON_RUNTIME_LIBDIR=<dir containing "
    "libtriton.so> or ensure the active Python (${Python3_EXECUTABLE}) imports "
    "`triton` with a built `_C/libtriton.so`. (No silent stub fallback.)")
endif()
message(STATUS "Using Triton runtime .so = ${TRITON_RUNTIME_SO}")
set(TL_PA_TRITON_RUNTIME_LIB "${TRITON_RUNTIME_SO}")
# `find_package(MLIR)` above already pulled the static MLIR archives into the
# link; the shim links those for the core dialects it registers. No Triton
# static archives and no --whole-archive are needed: libtriton.so resolves the
# Triton-dialect symbols + TypeIDs at link and load time.
set(TL_PA_WHOLE_ARCHIVE_LIBS "")

# ---- Vendored triton-shared sources ----------------------------------

set(VENDORED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendored/triton_shared")
include_directories(${VENDORED_DIR}/include)

# Stub builds skip the vendored PtrAnalysis.cpp because it transitively
# requires `triton/Dialect/Triton/IR/Dialect.h` from the upstream Triton MLIR
# install. The C ABI itself compiles regardless thanks to the `__has_include`
# gates in ptr_analysis_shim.cc.
if(TRITON_FRONTEND_STUB_BUILD)
  message(STATUS "TRITON_FRONTEND_STUB_BUILD=ON: skipping vendored PtrAnalysis.cpp")
  set(VENDORED_SOURCES "")
else()
  # In non-stub mode add the vendored triton-shared CMakeLists as a subdirectory
  # so its TableGen step (TritonStructuredVendoredTableGen) generates the
  # required `*.h.inc` headers and so the resulting STATIC libs
  # (TritonStructuredDialect, TritonSharedAnalysisStructured) are built and
  # linked correctly. We then add the build-tree include dir so the shim's TUs
  # can resolve the generated `triton-shared/.../*.h.inc` headers.
  add_subdirectory("${VENDORED_DIR}"
                   "${CMAKE_BINARY_DIR}/vendored_triton_shared")
  include_directories("${CMAKE_BINARY_DIR}/vendored_triton_shared/include")
  list(APPEND TL_PA_LINK_LIBS
       TritonSharedAnalysisStructured
       TritonSharedConversionStructuredToMemref
       TritonStructuredDialect
       TritonSharedRegister)
  set(VENDORED_SOURCES "")
endif()

# ---- Library + Python module ----------------------------------------

list(APPEND TL_PA_LINK_LIBS
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRAffineDialect
  MLIRTransforms
  MLIRTransformUtils
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

# Triton-dialect symbols + TypeIDs come from the installed wheel's libtriton.so
# (resolved above into TL_PA_TRITON_RUNTIME_LIB). Link it directly and bake an
# RPATH to its directory so the symbols resolve at module-load time. Because the
# triton runtime is a shared object, the linker keeps it as a NEEDED dependency
# (no --whole-archive games, no duplicate TypeID storage).
if(TL_PA_TRITON_RUNTIME_LIB)
  target_link_libraries(_triton_frontend_cxx PRIVATE "${TL_PA_TRITON_RUNTIME_LIB}")
  get_filename_component(_TRITON_SO_DIR "${TL_PA_TRITON_RUNTIME_LIB}" DIRECTORY)
  set_property(TARGET _triton_frontend_cxx APPEND PROPERTY
    BUILD_RPATH "${_TRITON_SO_DIR}")
  set_property(TARGET _triton_frontend_cxx APPEND PROPERTY
    INSTALL_RPATH "${_TRITON_SO_DIR}")
endif()

# The vendored tts dialect (TritonStructuredDialect) registers its own dialect /
# op / attr TypeIDs that MLIR only reaches through the runtime registry, so its
# archive members must be force-retained or the import dies with
# `undefined symbol: ..TypeIDResolver<..tts..>::id`. Force-load just this one
# archive (GNU ld / lld `--whole-archive` on ELF; `-force_load` on Mach-O).
if(NOT TRITON_FRONTEND_STUB_BUILD)
  if(APPLE)
    target_link_options(_triton_frontend_cxx PRIVATE
      "LINKER:-force_load,$<TARGET_FILE:TritonStructuredDialect>")
  else()
    target_link_options(_triton_frontend_cxx PRIVATE
      "LINKER:--whole-archive"
      "$<TARGET_FILE:TritonStructuredDialect>"
      "LINKER:--no-whole-archive")
  endif()
endif()

if(TARGET TritonStructuredVendoredTableGen)
  add_dependencies(_triton_frontend_cxx TritonStructuredVendoredTableGen)
endif()

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

# ---------------------------------------------------------------------
# Standalone pybind shim: register_triton_structured
#
# Removed in Integration #5: `register_dialects` is now exposed directly
# by `_triton_frontend_cxx` to avoid MLIR TypeID conflicts from linking
# statically against Triton dialects twice in the same python process.
# ---------------------------------------------------------------------
