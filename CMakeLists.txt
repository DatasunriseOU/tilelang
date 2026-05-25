# Learn a lot from the MLC - LLM Project
# https://github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.26)

# Detect CUDA toolkit: tries host installation first, then falls back to
# pip-installed packages (env WITH_PIP_CUDA_TOOLCHAIN or auto-detect).
# Must be included before project() so CMAKE_CUDA_COMPILER is set.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindPipCUDAToolkit.cmake)

project(TILE_LANG C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND "$ENV{CIBUILDWHEEL}")
  # Warning came from tvm submodule
  string(APPEND CMAKE_CXX_FLAGS " -Wno-dangling-reference")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# CPPMEGA local: skip submodule init — manually managed (apache/tvm latest, custom tvm-ffi pin)
if(FALSE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE TILELANG_GIT_SUBMODULE_RESULT
    )
    if(NOT TILELANG_GIT_SUBMODULE_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          "Failed to initialize git submodules. Please run "
          "`git submodule update --init --recursive` and re-run CMake."
      )
    endif()
  else()
    message(
      FATAL_ERROR
        "Git is required to initialize TileLang submodules. "
        "Please install git or fetch the submodules manually."
    )
  endif()
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM} with base_dir=${CMAKE_SOURCE_DIR}")
  if(APPLE)
    # Passing configs like `ccache base_dir=/xxx cc ...` is supported
    # (likely) since ccache 4.x, which has been provided by homebrew.
    # Our Linux builder image (manylinux2014 & manylinux_2_28) still
    # provides ccache 3.x and do not support this form.
    # `cibuildwheel` uses fixed folder on Linux (`/project`) as working directory,
    # so cache would work without setting `base_dir`.
    set(CCACHE_PROGRAM "${CCACHE_PROGRAM};base_dir=${CMAKE_SOURCE_DIR}")
  endif()
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
else()
  find_program(SCCACHE_PROGRAM sccache)
  if(SCCACHE_PROGRAM)
    message(STATUS "Using sccache: ${SCCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
  endif()
endif()

# Configs
set(TILELANG_BACKENDS CUDA ROCM METAL)

set(TILELANG_BACKEND_DOC_CUDA "Enable CUDA backend (ON/OFF/or CUDA SDK path)")
set(TILELANG_BACKEND_DOC_ROCM "Enable ROCm backend (ON/OFF/or ROCm SDK path)")
set(TILELANG_BACKEND_DOC_METAL "Enable Metal backend")

# TVM's config.cmake redefines USE_* options later, so we cache the user's choice
# (including explicit -DUSE_XXX arguments) before we include TVM and restore it
# afterwards.

macro(tilelang_define_backend_option BACKEND)
  set(_backend_var "USE_${BACKEND}")
  set(_doc "${TILELANG_BACKEND_DOC_${BACKEND}}")
  set(_user_override_var "TILELANG_USER_OVERRIDE_${_backend_var}")

  set(_user_override OFF)
  if(DEFINED ${_user_override_var})
    set(_user_override "${${_user_override_var}}")
  endif()

  if(DEFINED CACHE{${_backend_var}})
    get_property(_cache_type CACHE ${_backend_var} PROPERTY TYPE)
    if(_cache_type STREQUAL "UNINITIALIZED")
      set(_user_override ON)
    endif()
  endif()

  set(_default OFF)
  if(DEFINED ${_backend_var})
    set(_default "${${_backend_var}}")
  endif()

  option(${_backend_var} "${_doc}" "${_default}")
  # Remember if the user explicitly set this option so that later logic
  # won't auto-toggle backends they configured on the command line.
  set(${_user_override_var} ${_user_override} CACHE INTERNAL
    "User explicitly set ${_backend_var} during configuration" FORCE)
  set(TILELANG_OPTION_${_backend_var} "${${_backend_var}}")
endmacro()

foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  tilelang_define_backend_option(${BACKEND})
endforeach()

set(PREBUILD_CYTHON ON)

# CUDA stub libraries (cuda/cudart/nvrtc) are used to build wheels that can run
# across different CUDA Toolkit major versions and/or on CPU-only machines by
# avoiding hard DT_NEEDED dependencies on versioned CUDA SONAMEs.
#
# These stubs are currently POSIX-only (dlopen/dlsym via <dlfcn.h>).
if(WIN32 AND NOT CYGWIN)
  set(_TILELANG_USE_CUDA_STUBS_DEFAULT OFF)
else()
  set(_TILELANG_USE_CUDA_STUBS_DEFAULT ON)
endif()
option(TILELANG_USE_CUDA_STUBS
       "Use POSIX dlopen-based CUDA stub libraries (cuda/cudart/nvrtc) for portable wheels"
       ${_TILELANG_USE_CUDA_STUBS_DEFAULT})
unset(_TILELANG_USE_CUDA_STUBS_DEFAULT)

# HIP stub libraries (hip/hiprtc) are used to build wheels that can be imported
# on machines without ROCm installed by avoiding hard DT_NEEDED dependencies on
# libamdhip64.so / libhiprtc.so.
#
# These stubs are currently POSIX-only (dlopen/dlsym via <dlfcn.h>).
if(WIN32 AND NOT CYGWIN)
  set(_TILELANG_USE_HIP_STUBS_DEFAULT OFF)
else()
  # Only meaningful when USE_ROCM is enabled.
  set(_TILELANG_USE_HIP_STUBS_DEFAULT ON)
endif()
option(TILELANG_USE_HIP_STUBS
       "Use POSIX dlopen-based HIP stub libraries (hip/hiprtc) for portable wheels"
       ${_TILELANG_USE_HIP_STUBS_DEFAULT})
unset(_TILELANG_USE_HIP_STUBS_DEFAULT)
# Configs end

include(cmake/load_tvm.cmake)

if(EXISTS ${TVM_SOURCE}/cmake/config.cmake)
  include(${TVM_SOURCE}/cmake/config.cmake)
else()
  message(FATAL_ERROR "Nor tvm provided or submodule checkout-ed.")
endif()
# Re-apply TileLang's preferred backend settings after TVM's config may have
# overridden the USE_* cache entries.
foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  set(_backend_var "USE_${BACKEND}")
  set(_doc "${TILELANG_BACKEND_DOC_${BACKEND}}")
  set(${_backend_var} ${TILELANG_OPTION_${_backend_var}} CACHE STRING "${_doc}" FORCE)
  set(${_backend_var} ${TILELANG_OPTION_${_backend_var}})
endforeach()
# tvm tries to detect gtest by default, but may fail if its header is not installed.
set(USE_GTEST OFF)

# TileLang is built as a separate dylib but still derives from and calls Apache
# TVM internal C++ classes (tirx mutators, codegen, arith helpers, etc.).  Those
# classes are not all marked TVM_DLL, so hiding TVM private symbols leaves
# libtilelang with unresolved references on Darwin.
set(HIDE_PRIVATE_SYMBOLS OFF CACHE STRING
    "TileLang links against TVM internal C++ APIs; keep them visible." FORCE)

# Include directories for TileLang
set(TILE_LANG_INCLUDES ${TVM_INCLUDES})
list(APPEND TILE_LANG_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Collect source files
file(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/transform/common/*.cc
  src/transform/vendored/*.cc
  src/op/*.cc
  src/backend/cpu/op/*.cc
  src/backend/cuda/op/copy_analysis.cc
  src/backend/metal/op/*.cc
  src/backend/webgpu/op/*.cc
  # CuTeDSL codegen emits Python source and uses TVM's CUDA fallback module
  # only as a source container. Keep this compile-free builder available in
  # Metal/CPU dev builds so source-level CuTeDSL tests do not require CUDA.
  src/target/codegen_py.cc
  src/target/codegen_utils.cc
  src/target/codegen_cutedsl.cc
  src/target/ptx.cc
  src/target/rt_mod_cutedsl.cc
  src/target/utils.cc
  src/target/codegen_c_host.cc
  src/target/codegen_c.cc
  src/target/rt_mod_c.cc
  # intrin_rule doesn't have system dependency
  src/target/intrin_rule*.cc
)

# Always include CPU-safe runtime helpers
list(APPEND TILE_LANG_SRCS
  src/runtime/error_helpers.cc
)

set(TILELANG_OUTPUT_TARGETS tilelang tvm_runtime tvm_compiler)

# Track if the user explicitly selected a backend via cache options.
set(TILELANG_BACKEND_USER_SELECTED OFF)
foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  set(_backend_var "USE_${BACKEND}")
  set(_override_var "TILELANG_USER_OVERRIDE_${_backend_var}")
  if(${_backend_var} OR ${_override_var})
    set(TILELANG_BACKEND_USER_SELECTED ON)
  endif()
endforeach()

# Only auto-select a backend when the user didn't specify one explicitly.
if(NOT TILELANG_BACKEND_USER_SELECTED)
  if($ENV{USE_METAL})
    set(USE_METAL ON)
  elseif(APPLE)
    message(STATUS "Enable Metal support by default.")
    set(USE_METAL ON)
  elseif($ENV{USE_ROCM})
    set(USE_ROCM ON)
  else()
    if($ENV{USE_CUDA})
      set(USE_CUDA ON)
    elseif(DEFINED ENV{USE_CUDA} AND NOT $ENV{USE_CUDA})
      # Build CPU-only when we explicitly disable CUDA
      set(USE_CUDA OFF)
    else()
      message(STATUS "Enable CUDA support by default.")
      set(USE_CUDA ON)
    endif()
  endif()
endif()

# Backend-local CMake files own native source lists, stubs, include paths, and
# compile definitions. Top-level CMake only selects and delegates.
include("${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/src/backend/rocm/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/CMakeLists.txt")

set(USE_Z3      ON CACHE STRING "Use Z3 SMT solver for TileLang optimizations")
set(USE_PYPI_Z3 ON CACHE BOOL   "Use Z3 provided by PyPI z3-solver package")

if(USE_Z3 AND USE_PYPI_Z3)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/pypi-z3")
  find_package(Z3 REQUIRED)
endif()

# Enable custom logging so we control the output format (e.g. strip build paths
# from __FILE__ so wheel users don't see CI machine paths in warnings).
set(USE_CUSTOM_LOGGING ON CACHE BOOL "Use custom logging implementation" FORCE)

# TVM consumes USE_LLVM from its own subdirectory scope.  When TileLang is the
# top-level project, a command-line -DUSE_LLVM=/path/to/llvm-config can remain
# as an untyped top-level cache entry and TVM's option helper falls back to OFF.
# Re-materialize the cache value as a normal variable before add_subdirectory.
if(DEFINED CACHE{USE_LLVM})
  get_property(TILELANG_TVM_USE_LLVM CACHE USE_LLVM PROPERTY VALUE)
  if(NOT "${TILELANG_TVM_USE_LLVM}" STREQUAL "")
    set(USE_LLVM "${TILELANG_TVM_USE_LLVM}" CACHE STRING "Build TVM with LLVM" FORCE)
    set(USE_LLVM "${TILELANG_TVM_USE_LLVM}")
  endif()
endif()

# Detect release (wheel) builds: in CI (cibuildwheel) or scikit-build-core wheel builds,
# we strip source paths from LOG(WARNING) etc. for a cleaner user experience.
# Local dev builds keep full paths for debugging.
if(DEFINED ENV{CIBUILDWHEEL} OR "$ENV{SKBUILD_STATE}" STREQUAL "wheel")
  set(TILELANG_RELEASE_BUILD_DEFAULT ON)
else()
  set(TILELANG_RELEASE_BUILD_DEFAULT OFF)
endif()
option(TILELANG_RELEASE_BUILD "Strip source paths from log messages (for wheel releases)" ${TILELANG_RELEASE_BUILD_DEFAULT})

# CPPMEGA z3-stack fix-A5: gate test-only FFI surface behind a build flag
# so we don't ship `tl.z3.bv_can_prove` / `tl.z3.bv_scoped_round_trip`
# in release wheels (these are diagnostic helpers only used by tests).
# CPPMEGA fix-round-2 (HIGH design+sec): default flipped to OFF so a
# stock release wheel does NOT ship the test-only Z3 probe FFIs (they
# expose internal prover state and bypass scope discipline). CI/test
# runs must pass `-DTILELANG_BUILD_TESTS=ON` explicitly.
option(TILELANG_BUILD_TESTS "Build test-only FFI helpers (e.g. Z3 bv probes)" OFF)

# Include tvm after configs have been populated
add_subdirectory(${TVM_SOURCE} tvm EXCLUDE_FROM_ALL)

# Provide the custom LogMessageImpl / LogFatalImpl implementation to TVM,
# since TVM_LOG_CUSTOMIZE=1 requires them to be supplied by the user.
# target_sources(tvm_runtime_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/tilelang_logging_impl.cc")  # disabled with TVM_LOG_CUSTOMIZE
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tvm_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()

# Resolve compile warnings in tvm
add_compile_definitions(DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

# Set debug mode compile definitions
# Enable the TVM debug option, i.e., TVM_LOG_DEBUG
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building TileLang with DEBUG mode")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
endif()

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
# CPPMEGA: force-include compat shim so old ICHECK macros alias to TVM_FFI_ICHECK family
target_compile_options(tilelang_objs PRIVATE "-include" "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/vendored/tl_compat.h")
# CPPMEGA: disable custom logging (incompatible with apache/tvm latest tvm-ffi API)
# target_compile_definitions(tilelang_objs PRIVATE TVM_LOG_CUSTOMIZE=1)
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tilelang_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()
# CPPMEGA z3-stack fix-A5: surface TILELANG_BUILD_TESTS as a compile def
# so source files (currently `src/transform/vendored/z3_prover.cc`) can
# `#ifdef`-gate test-only FFI registrations. Default ON keeps existing
# tests working unchanged.
if(TILELANG_BUILD_TESTS)
  target_compile_definitions(tilelang_objs PRIVATE TILELANG_BUILD_TESTS=1)
endif()

add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm_compiler)

# CPPMEGA: vendored Z3 prover (src/transform/vendored/z3_prover.cc) needs
# z3++.h headers and libz3 at link time. The find_package(Z3) above defines
# z3::libz3 (the imported shared library) but did not previously feed it
# into TileLang because the original sync rewriter used a stub. Wire it in
# now so the partial-sync optimization paths actually solve.
if(USE_Z3 AND USE_PYPI_Z3)
  target_include_directories(tilelang_objs PRIVATE
    "$<TARGET_PROPERTY:z3::libz3,INTERFACE_INCLUDE_DIRECTORIES>")
  target_link_libraries(tilelang PRIVATE z3::libz3)
  if(APPLE)
    # PyPI z3-solver ships libz3.dylib with a bare install name
    # ("libz3.dylib"). dyld does not resolve bare install names through
    # LC_RPATH, so the installed wheel can fail to import even though the
    # z3 package path is present in INSTALL_RPATH. Normalize the dependency
    # edge to @rpath so the rpath below is actually used.
    add_custom_command(
      TARGET tilelang
      POST_BUILD
      COMMAND "${CMAKE_INSTALL_NAME_TOOL}" -change libz3.dylib
              @rpath/libz3.dylib "$<TARGET_FILE:tilelang>"
      COMMENT "Normalizing TileLang Z3 install name to @rpath/libz3.dylib"
      VERBATIM)
  endif()
endif()

# Place dev build outputs under build/lib for consistency
set_target_properties(tilelang PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
# Build cython extension
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp"
  COMMENT
    "Cythoning tilelang/jit/adapter/cython/cython_wrapper.pyx"
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
          --module-name tilelang_cython_wrapper
          --cplus --output-file "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
  VERBATIM)

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

python_add_library(tilelang_cython_wrapper MODULE "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp" ${USE_SABI} WITH_SOABI)

# Disable Cython's PEP-489 multi-phase init for the wrapper. The generated
# C++ depends on CPython's private `_xxsubinterpreters` module at import
# time, which is stripped from some distributor-built Python 3.12 builds
# (notably Red Hat's RHEL 9 system Python). Single-phase init avoids that
# dependency and matches Cython's own suggested workaround. See #2125.
target_compile_definitions(tilelang_cython_wrapper PRIVATE CYTHON_PEP489_MULTI_PHASE_INIT=0)

# Ensure dev builds drop the extension into build/lib alongside other shared libs
set_target_properties(tilelang_cython_wrapper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install the extension into tilelang/lib inside the wheel
install(TARGETS tilelang_cython_wrapper
        LIBRARY DESTINATION tilelang/lib
        RUNTIME DESTINATION tilelang/lib
        ARCHIVE DESTINATION tilelang/lib)

option(TILELANG_BUILD_MLX_TVM_FFI
       "Build optional MLX graph-safe TVM-FFI Metal bridge"
       ON)

if(APPLE AND USE_METAL AND TILELANG_BUILD_MLX_TVM_FFI)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE TILELANG_NANOBIND_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TILELANG_NANOBIND_RESULT)
  if(TILELANG_NANOBIND_RESULT EQUAL 0 AND EXISTS "${TILELANG_NANOBIND_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${TILELANG_NANOBIND_ROOT}")
    find_package(nanobind CONFIG QUIET)
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import mlx, mlx.core as mx, sys; root=next(iter(getattr(mlx, '__path__', [])), ''); print(root); print(getattr(mx, '__file__', '')); print(getattr(mx, '__version__', ''))"
    OUTPUT_VARIABLE TILELANG_MLX_PY_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TILELANG_MLX_PY_ERROR
    RESULT_VARIABLE TILELANG_MLX_PY_RESULT)
  if(TILELANG_MLX_PY_RESULT EQUAL 0)
    string(REPLACE "\r\n" "\n" TILELANG_MLX_PY_INFO "${TILELANG_MLX_PY_INFO}")
    string(REPLACE "\n" ";" TILELANG_MLX_PY_INFO_LINES "${TILELANG_MLX_PY_INFO}")
    list(LENGTH TILELANG_MLX_PY_INFO_LINES TILELANG_MLX_PY_INFO_COUNT)
    if(TILELANG_MLX_PY_INFO_COUNT LESS 3)
      message(FATAL_ERROR
        "MLX Python package introspection returned incomplete data from ${Python_EXECUTABLE}: "
        "${TILELANG_MLX_PY_INFO}")
    endif()
    list(GET TILELANG_MLX_PY_INFO_LINES 0 TILELANG_MLX_PY_ROOT)
    list(GET TILELANG_MLX_PY_INFO_LINES 1 TILELANG_MLX_PY_CORE)
    list(GET TILELANG_MLX_PY_INFO_LINES 2 TILELANG_MLX_PY_VERSION)
    if(TILELANG_MLX_PY_ROOT STREQUAL "" OR TILELANG_MLX_PY_VERSION STREQUAL "")
      message(FATAL_ERROR
        "MLX Python package did not expose a usable package root/version from "
        "${Python_EXECUTABLE}: root='${TILELANG_MLX_PY_ROOT}' version='${TILELANG_MLX_PY_VERSION}'")
    endif()
  else()
    message(STATUS
      "Skipping optional MLX TVM-FFI bridge: Python MLX import failed from "
      "${Python_EXECUTABLE}: ${TILELANG_MLX_PY_ERROR}")
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_VARIABLE TILELANG_MLX_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TILELANG_MLX_RESULT)

  if(TILELANG_MLX_PY_RESULT EQUAL 0)
    if(DEFINED MLX_DIR AND NOT "${MLX_DIR}" STREQUAL "")
      get_filename_component(TILELANG_MLX_ROOT "${MLX_DIR}/../../.." ABSOLUTE)
    elseif(NOT TILELANG_MLX_RESULT EQUAL 0 OR NOT EXISTS "${TILELANG_MLX_ROOT}")
      set(TILELANG_MLX_ROOT "${TILELANG_MLX_PY_ROOT}")
    endif()

    file(REAL_PATH "${TILELANG_MLX_ROOT}" TILELANG_MLX_ROOT_REAL)
    file(REAL_PATH "${TILELANG_MLX_PY_ROOT}" TILELANG_MLX_PY_ROOT_REAL)
    if(NOT TILELANG_MLX_ROOT_REAL STREQUAL TILELANG_MLX_PY_ROOT_REAL)
      message(FATAL_ERROR
        "MLX CMake root does not match the active Python MLX package.\n"
        "  Python executable: ${Python_EXECUTABLE}\n"
        "  Python MLX root:   ${TILELANG_MLX_PY_ROOT_REAL}\n"
        "  Requested MLX root: ${TILELANG_MLX_ROOT_REAL}\n"
        "Set MLX_DIR to the active Python package's share/cmake/MLX directory, "
        "or run CMake with the Python environment that owns the requested MLX build.")
    endif()

    set(TILELANG_MLX_CONFIG_DIR "${TILELANG_MLX_ROOT}/share/cmake/MLX")
    set(TILELANG_MLX_CONFIG_VERSION_FILE "${TILELANG_MLX_CONFIG_DIR}/MLXConfigVersion.cmake")
    if(NOT EXISTS "${TILELANG_MLX_CONFIG_DIR}/MLXConfig.cmake" OR
       NOT EXISTS "${TILELANG_MLX_CONFIG_VERSION_FILE}")
      message(FATAL_ERROR
        "Active Python MLX package is missing CMake package metadata: "
        "${TILELANG_MLX_CONFIG_DIR}")
    endif()

    file(READ "${TILELANG_MLX_CONFIG_VERSION_FILE}" TILELANG_MLX_CONFIG_VERSION_TEXT)
    string(REGEX MATCH "set\\(PACKAGE_VERSION \"([^\"]+)\"\\)"
           TILELANG_MLX_CONFIG_VERSION_MATCH
           "${TILELANG_MLX_CONFIG_VERSION_TEXT}")
    if(NOT CMAKE_MATCH_1)
      message(FATAL_ERROR
        "Could not parse MLX package version from ${TILELANG_MLX_CONFIG_VERSION_FILE}")
    endif()
    set(TILELANG_MLX_CONFIG_VERSION "${CMAKE_MATCH_1}")
    if(NOT TILELANG_MLX_CONFIG_VERSION STREQUAL TILELANG_MLX_PY_VERSION)
      message(FATAL_ERROR
        "MLX Python package version and CMake package version differ.\n"
        "  Python MLX version: ${TILELANG_MLX_PY_VERSION}\n"
        "  CMake MLX version:  ${TILELANG_MLX_CONFIG_VERSION}\n"
        "  MLX root:           ${TILELANG_MLX_ROOT_REAL}")
    endif()

    set(MLX_LIBRARY "${TILELANG_MLX_ROOT}/lib/libmlx.dylib" CACHE FILEPATH
        "libmlx.dylib used by TileLang's MLX TVM-FFI bridge" FORCE)
    set(MLX_DIR "${TILELANG_MLX_CONFIG_DIR}" CACHE PATH
        "MLX CMake package used by TileLang's MLX TVM-FFI bridge" FORCE)
    find_package(MLX CONFIG QUIET
                 PATHS "${TILELANG_MLX_ROOT}"
                 NO_DEFAULT_PATH)
    if(MLX_FOUND AND TARGET mlx)
      get_target_property(TILELANG_MLX_LIB mlx IMPORTED_LOCATION_RELEASE)
      if(NOT TILELANG_MLX_LIB)
        get_target_property(TILELANG_MLX_LIB mlx IMPORTED_LOCATION)
      endif()
      if(NOT TILELANG_MLX_LIB OR NOT EXISTS "${TILELANG_MLX_LIB}")
        message(FATAL_ERROR
          "MLX imported target does not point at an existing libmlx.dylib")
      endif()
      file(REAL_PATH "${TILELANG_MLX_LIB}" TILELANG_MLX_LIB_REAL)
      file(REAL_PATH "${MLX_LIBRARY}" TILELANG_MLX_LIBRARY_VAR_REAL)
      if(NOT TILELANG_MLX_LIBRARY_VAR_REAL STREQUAL TILELANG_MLX_LIB_REAL)
        message(FATAL_ERROR
          "MLX_LIBRARY cache entry and MLX imported target disagree.\n"
          "  MLX_LIBRARY: ${TILELANG_MLX_LIBRARY_VAR_REAL}\n"
          "  target mlx:  ${TILELANG_MLX_LIB_REAL}")
      endif()
      string(FIND "${TILELANG_MLX_LIB_REAL}" "${TILELANG_MLX_ROOT_REAL}/" TILELANG_MLX_LIB_ROOT_POS)
      if(NOT TILELANG_MLX_LIB_ROOT_POS EQUAL 0)
        message(FATAL_ERROR
          "MLX imported target points outside the active MLX package.\n"
          "  MLX root: ${TILELANG_MLX_ROOT_REAL}\n"
          "  libmlx:   ${TILELANG_MLX_LIB_REAL}")
      endif()
      file(SHA256 "${TILELANG_MLX_LIB_REAL}" TILELANG_MLX_LIB_SHA256)
      set(TILELANG_MLX_PY_BRIDGE "${TILELANG_MLX_ROOT}/lib/libmlx_python_bridge.dylib")
      if(NOT EXISTS "${TILELANG_MLX_PY_BRIDGE}")
        message(FATAL_ERROR
          "Active Python MLX package is missing libmlx_python_bridge.dylib, "
          "which TileLang needs to wrap native mx::array outputs without "
          "constructing dummy Python arrays.\n"
          "  Expected bridge: ${TILELANG_MLX_PY_BRIDGE}\n"
          "  MLX root:        ${TILELANG_MLX_ROOT_REAL}")
      endif()
      file(REAL_PATH "${TILELANG_MLX_PY_BRIDGE}" TILELANG_MLX_PY_BRIDGE_REAL)
      string(FIND "${TILELANG_MLX_PY_BRIDGE_REAL}" "${TILELANG_MLX_ROOT_REAL}/"
             TILELANG_MLX_PY_BRIDGE_ROOT_POS)
      if(NOT TILELANG_MLX_PY_BRIDGE_ROOT_POS EQUAL 0)
        message(FATAL_ERROR
          "MLX Python bridge points outside the active MLX package.\n"
          "  MLX root: ${TILELANG_MLX_ROOT_REAL}\n"
          "  bridge:   ${TILELANG_MLX_PY_BRIDGE_REAL}")
      endif()
      file(SHA256 "${TILELANG_MLX_PY_BRIDGE_REAL}" TILELANG_MLX_PY_BRIDGE_SHA256)
      message(STATUS "MLX bridge Python executable: ${Python_EXECUTABLE}")
      message(STATUS "MLX bridge Python package: root=${TILELANG_MLX_PY_ROOT_REAL}, core=${TILELANG_MLX_PY_CORE}, version=${TILELANG_MLX_PY_VERSION}")
      message(STATUS "MLX bridge CMake package: root=${TILELANG_MLX_ROOT_REAL}, config=${TILELANG_MLX_CONFIG_DIR}, version=${TILELANG_MLX_CONFIG_VERSION}")
      message(STATUS "MLX bridge libmlx: ${TILELANG_MLX_LIB_REAL} sha256=${TILELANG_MLX_LIB_SHA256}")
      message(STATUS "MLX bridge wrapper: ${TILELANG_MLX_PY_BRIDGE_REAL} sha256=${TILELANG_MLX_PY_BRIDGE_SHA256}")

      include(CheckCXXSourceCompiles)
      set(CMAKE_REQUIRED_LIBRARIES mlx)
      set(CMAKE_REQUIRED_QUIET TRUE)
      check_cxx_source_compiles([=[
        #include <mlx/backend/metal/device.h>
        #include <mlx/device.h>
        #include <mlx/stream.h>
        int main() {
          auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
          auto& encoder = mlx::core::metal::get_command_encoder(stream);
          encoder.prepare_external_dispatch();
          (void)encoder.raw_command_encoder();
          encoder.end_encoding();
          (void)encoder.get_command_buffer();
          (void)&encoder;
          return 0;
        }
      ]=] TILELANG_MLX_HAS_DIRECT_METAL_COMMAND_ENCODER_API)
      unset(CMAKE_REQUIRED_LIBRARIES)
      unset(CMAKE_REQUIRED_QUIET)
      if(NOT TILELANG_MLX_HAS_DIRECT_METAL_COMMAND_ENCODER_API)
        message(FATAL_ERROR
          "MLX lib/header pair does not expose the direct Metal CommandEncoder API required "
          "by TileLang's graph-safe TVM-FFI bridge, including active compute encoder interop. "
          "Checked libmlx: "
          "${TILELANG_MLX_LIB_REAL}")
      endif()
      set(CMAKE_REQUIRED_LIBRARIES mlx "${TILELANG_MLX_PY_BRIDGE_REAL}")
      set(CMAKE_REQUIRED_QUIET TRUE)
      check_cxx_source_compiles([=[
        #include <mlx/array.h>
        extern "C" void* mlx_core_wrap_mx_array_move(mlx::core::array*);
        int main() {
          auto* wrapper = &mlx_core_wrap_mx_array_move;
          return wrapper == nullptr;
        }
      ]=] TILELANG_MLX_HAS_ARRAY_WRAPPER)
      unset(CMAKE_REQUIRED_LIBRARIES)
      unset(CMAKE_REQUIRED_QUIET)
      if(NOT TILELANG_MLX_HAS_ARRAY_WRAPPER)
        message(FATAL_ERROR
          "MLX Python bridge does not export mlx_core_wrap_mx_array_move().\n"
          "  Checked bridge: ${TILELANG_MLX_PY_BRIDGE_REAL}\n"
          "  Bridge sha256:  ${TILELANG_MLX_PY_BRIDGE_SHA256}\n"
          "  MLX version:    ${TILELANG_MLX_CONFIG_VERSION}")
      endif()
    endif()
  endif()

  if(nanobind_FOUND AND MLX_FOUND AND TARGET tvm_ffi_shared)
    set(TILELANG_MLX_TVM_FFI_C_API_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h")
    file(SHA256 "${TILELANG_MLX_TVM_FFI_C_API_HEADER}"
         TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256)
    set(TILELANG_MLX_TVM_FFI_INCLUDE_DIRS
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${TVM_SOURCE}/3rdparty/tvm-ffi/include"
      "${TVM_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include")
    set(TILELANG_MLX_TVM_FFI_COMPILE_DEFS
      "TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256=\"${TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256}\""
      "TILELANG_MLX_TVM_FFI_BUILD_MLX_VERSION=\"${TILELANG_MLX_CONFIG_VERSION}\""
      "TILELANG_MLX_TVM_FFI_BUILD_MLX_LIB_SHA256=\"${TILELANG_MLX_LIB_SHA256}\""
      "TILELANG_MLX_TVM_FFI_BUILD_MLX_PY_BRIDGE_SHA256=\"${TILELANG_MLX_PY_BRIDGE_SHA256}\"")

    nanobind_build_library(nanobind-static)
    add_library(
      tilelang_mlx_tvm_ffi_c_api
      SHARED
      src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_ext.cpp)
    nanobind_compile_options(tilelang_mlx_tvm_ffi_c_api)
    nanobind_link_options(tilelang_mlx_tvm_ffi_c_api)
    target_include_directories(tilelang_mlx_tvm_ffi_c_api PRIVATE
      ${TILELANG_MLX_TVM_FFI_INCLUDE_DIRS})
    target_compile_definitions(tilelang_mlx_tvm_ffi_c_api PRIVATE
      TILELANG_MLX_TVM_FFI_EXPORTS
      TILELANG_MLX_TVM_FFI_WITH_PY_MODULE=0
      NB_DOMAIN=mlx
      ${TILELANG_MLX_TVM_FFI_COMPILE_DEFS})
    target_link_libraries(tilelang_mlx_tvm_ffi_c_api PRIVATE
      nanobind-static
      mlx
      tvm_runtime
      tvm_ffi_shared
      "${TILELANG_MLX_PY_BRIDGE_REAL}")
    set_target_properties(tilelang_mlx_tvm_ffi_c_api PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      BUILD_RPATH "${TILELANG_MLX_ROOT}/lib"
      INSTALL_RPATH "@loader_path;${TILELANG_MLX_ROOT}/lib")
    install(TARGETS tilelang_mlx_tvm_ffi_c_api
            LIBRARY DESTINATION tilelang/lib
            RUNTIME DESTINATION tilelang/lib
            ARCHIVE DESTINATION tilelang/lib)

    nanobind_add_module(
      _tilelang_mlx_tvm_ffi
      NB_STATIC
      STABLE_ABI
      LTO
      NOMINSIZE
      NB_DOMAIN
      mlx
      src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_ext.cpp)
    target_include_directories(_tilelang_mlx_tvm_ffi PRIVATE
      ${TILELANG_MLX_TVM_FFI_INCLUDE_DIRS})
    target_compile_definitions(_tilelang_mlx_tvm_ffi PRIVATE
      TILELANG_MLX_TVM_FFI_EXPORTS
      TILELANG_MLX_TVM_FFI_WITH_PY_MODULE=1
      ${TILELANG_MLX_TVM_FFI_COMPILE_DEFS})
    target_link_libraries(_tilelang_mlx_tvm_ffi PRIVATE
      mlx
      tvm_runtime
      tvm_ffi_shared
      "${TILELANG_MLX_PY_BRIDGE_REAL}")
    set_target_properties(_tilelang_mlx_tvm_ffi PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      BUILD_RPATH "${TILELANG_MLX_ROOT}/lib"
      INSTALL_RPATH "@loader_path;${TILELANG_MLX_ROOT}/lib")
    install(TARGETS _tilelang_mlx_tvm_ffi
            LIBRARY DESTINATION tilelang/lib
            RUNTIME DESTINATION tilelang/lib
            ARCHIVE DESTINATION tilelang/lib)
    install(FILES "${TILELANG_MLX_TVM_FFI_C_API_HEADER}"
            DESTINATION tilelang/include/tilelang/contrib)
  else()
    message(STATUS "Skipping optional MLX TVM-FFI bridge: nanobind/MLX/tvm_ffi_shared target not available")
  endif()
endif()

# Copy libz3.so to build folder to workaround isolated build env issue
if(USE_Z3 AND USE_PYPI_Z3)
  get_target_property(Z3_LIBRARY_PATH z3::libz3 IMPORTED_LOCATION)
  install(FILES "${Z3_LIBRARY_PATH}" DESTINATION "${CMAKE_BINARY_DIR}/lib")
  if(APPLE)
    set_target_properties(tvm_compiler PROPERTIES BUILD_RPATH "@loader_path")
  else()
    set_target_properties(tvm_compiler PROPERTIES BUILD_RPATH "\$ORIGIN")
  endif()
endif()

if(DEFINED TILELANG_ACTIVE_BACKEND_STUB_LINK)
  foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
    target_link_libraries(${target} PUBLIC ${TILELANG_ACTIVE_BACKEND_STUB_LINK})
  endforeach()
endif()

# Append stub targets after the linking loop so they don't link to themselves
if(DEFINED TILELANG_ACTIVE_BACKEND_STUB_TARGETS)
  list(APPEND TILELANG_OUTPUT_TARGETS ${TILELANG_ACTIVE_BACKEND_STUB_TARGETS})
endif()

unset(PATCHELF_EXECUTABLE CACHE)

if(APPLE)
  set(TILELANG_INSTALL_RPATH "@loader_path;@loader_path/../../tvm_ffi/lib")
  if(USE_Z3 AND USE_PYPI_Z3)
    # Some z3 is placed in lib/ and some in bin/, we add both in rpath
    string(APPEND TILELANG_INSTALL_RPATH ";@loader_path/../../z3/lib;@loader_path/../../z3/bin")
  endif()
elseif(UNIX)
  set(TILELANG_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../tvm_ffi/lib")
  if(USE_Z3 AND USE_PYPI_Z3)
    string(APPEND TILELANG_INSTALL_RPATH ":\$ORIGIN/../../z3/lib")
  endif()
  if(DEFINED TILELANG_ACTIVE_BACKEND_RPATH_EXTRA)
    string(APPEND TILELANG_INSTALL_RPATH "${TILELANG_ACTIVE_BACKEND_RPATH_EXTRA}")
  endif()
  find_program(PATCHELF_EXECUTABLE patchelf)
  if (NOT PATCHELF_EXECUTABLE)
    message(STATUS "`patchelf` not found.")
  endif()
endif()

# Let libtilelang search for tvm in the same directory
foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
  set_target_properties(${target} PROPERTIES INSTALL_RPATH "${TILELANG_INSTALL_RPATH}")
  set_target_properties(${target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
endforeach()

# Strip backend runtime dependencies for portable wheels
if(DEFINED TILELANG_ACTIVE_BACKEND_PATCHELF_REMOVE AND PATCHELF_EXECUTABLE)
  foreach(_needed IN LISTS TILELANG_ACTIVE_BACKEND_PATCHELF_REMOVE)
    set(_patchelf_remove_args "${_patchelf_remove_args} --remove-needed ${_needed}")
  endforeach()
  foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
    install(CODE "
      execute_process(
        COMMAND ${PATCHELF_EXECUTABLE}${_patchelf_remove_args}
          \"$<TARGET_FILE:${target}>\"
        WORKING_DIRECTORY \"${CMAKE_INSTALL_PREFIX}\"
        RESULT_VARIABLE patchelf_result
      )
      if(patchelf_result EQUAL 0)
        message(STATUS \"patchelf: removed dependencies from $<TARGET_FILE:${target}>\")
      else()
        message(WARNING \"patchelf failed for $<TARGET_FILE:${target}>\")
      endif()
    ")
  endforeach()
endif()

install(
  TARGETS ${TILELANG_OUTPUT_TARGETS}
  LIBRARY DESTINATION tilelang/lib
  RUNTIME DESTINATION tilelang/lib
  ARCHIVE DESTINATION tilelang/lib
)
