# Install script for directory: /Volumes/external/sources/tilelang

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/lib" TYPE MODULE FILES "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/tilelang_cython_wrapper.cpython-313-darwin.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/tilelang_cython_wrapper.cpython-313-darwin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/tilelang_cython_wrapper.cpython-313-darwin.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/tilelang_cython_wrapper.cpython-313-darwin.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/lib" TYPE SHARED_LIBRARY FILES "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/libtilelang_mlx_tvm_ffi_c_api.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang_mlx_tvm_ffi_c_api.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang_mlx_tvm_ffi_c_api.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib"
      -add_rpath "@loader_path"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang_mlx_tvm_ffi_c_api.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang_mlx_tvm_ffi_c_api.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/lib" TYPE MODULE FILES "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/_tilelang_mlx_tvm_ffi.cpython-313-darwin.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/_tilelang_mlx_tvm_ffi.cpython-313-darwin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/_tilelang_mlx_tvm_ffi.cpython-313-darwin.so")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib"
      -add_rpath "@loader_path"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/_tilelang_mlx_tvm_ffi.cpython-313-darwin.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/_tilelang_mlx_tvm_ffi.cpython-313-darwin.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/include/tilelang/contrib" TYPE FILE FILES "/Volumes/external/sources/tilelang/src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/libz3.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib" TYPE FILE FILES "/Volumes/external/sources/nanochat/.venv/lib/python3.13/site-packages/z3/lib/libz3.dylib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/lib" TYPE SHARED_LIBRARY FILES "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/libtilelang.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib"
      -add_rpath "@loader_path"
      -add_rpath "@loader_path/../../tvm_ffi/lib"
      -add_rpath "@loader_path/../../z3/lib"
      -add_rpath "@loader_path/../../z3/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtilelang.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/lib" TYPE SHARED_LIBRARY FILES "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/libtvm_runtime.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_runtime.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_runtime.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib"
      -add_rpath "@loader_path"
      -add_rpath "@loader_path/../../tvm_ffi/lib"
      -add_rpath "@loader_path/../../z3/lib"
      -add_rpath "@loader_path/../../z3/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_runtime.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_runtime.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tilelang/lib" TYPE SHARED_LIBRARY FILES "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib/libtvm_compiler.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_compiler.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_compiler.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/lib"
      -add_rpath "@loader_path/../../tvm_ffi/lib"
      -add_rpath "@loader_path/../../z3/lib"
      -add_rpath "@loader_path/../../z3/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_compiler.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/tilelang/lib/libtvm_compiler.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Volumes/external/sources/tilelang/build-appleclang-triton-llvm/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
