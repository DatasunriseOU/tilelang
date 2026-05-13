#ifndef TILELANG_CONTRIB_MLX_TVM_FFI_C_API_H_
#define TILELANG_CONTRIB_MLX_TVM_FFI_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifndef PyObject_HEAD
typedef struct _object PyObject;
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef TILELANG_MLX_TVM_FFI_EXPORTS
#define TILELANG_MLX_TVM_FFI_EXPORT __declspec(dllexport)
#else
#define TILELANG_MLX_TVM_FFI_EXPORT __declspec(dllimport)
#endif
#else
#define TILELANG_MLX_TVM_FFI_EXPORT __attribute__((visibility("default")))
#endif

#define TILELANG_MLX_TVM_FFI_C_API_VERSION 1u
#define TILELANG_MLX_TVM_FFI_C_API_ABI_HASH                                    \
  "tilelang.mlx_tvm_ffi.c_api.v1.pyobject_metal_call.20260514"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum TileLangMLXTVMFFICApiResult {
  kTileLangMLXTVMFFICApiOk = 0,
  kTileLangMLXTVMFFICApiNullOut = 1,
  kTileLangMLXTVMFFICApiStructTooSmall = 2,
  kTileLangMLXTVMFFICApiVersionMismatch = 3,
  kTileLangMLXTVMFFICApiHashMismatch = 4,
} TileLangMLXTVMFFICApiResult;

typedef struct TileLangMLXTVMFFIStatus {
  uint32_t version;
  uint32_t struct_size;
  int32_t code;
  const char *state;
  const char *reason;
  const char *abi_hash;
  const char *header_sha256;
  const char *mlx_version;
  const char *mlx_lib_sha256;
  const char *mlx_python_bridge_sha256;
} TileLangMLXTVMFFIStatus;

typedef PyObject *(*TileLangMLXTVMFFIMetalCallFn)(
    uint64_t func_handle, PyObject *inputs, PyObject *output_shapes,
    PyObject *output_dtypes, PyObject *result_indices, int64_t num_params,
    PyObject *zero_init_output_positions, PyObject *launch_sync_state,
    PyObject *wait_edges);

typedef PyObject *(*TileLangMLXTVMFFIOwnerOutputBufferFn)(
    PyObject *shape, const char *dtype_name);

typedef PyObject *(*TileLangMLXTVMFFIOwnerOutputBuffersFn)(
    PyObject *shapes, PyObject *dtype_names);

typedef PyObject *(*TileLangMLXTVMFFINoArgsPyObjectFn)(void);
typedef int (*TileLangMLXTVMFFIStatusFn)(TileLangMLXTVMFFIStatus *out,
                                         size_t out_size);

typedef struct TileLangMLXTVMFFICAPI {
  uint32_t version;
  uint32_t struct_size;
  const char *abi_hash;
  const char *header_sha256;
  const char *mlx_version;
  const char *mlx_lib_sha256;
  const char *mlx_python_bridge_sha256;
  TileLangMLXTVMFFIStatusFn status;
  TileLangMLXTVMFFIMetalCallFn metal_call;
  TileLangMLXTVMFFIOwnerOutputBufferFn owner_output_buffer;
  TileLangMLXTVMFFIOwnerOutputBuffersFn owner_output_buffers;
  TileLangMLXTVMFFINoArgsPyObjectFn make_launch_sync_state;
  TileLangMLXTVMFFINoArgsPyObjectFn make_sync_edge;
  TileLangMLXTVMFFINoArgsPyObjectFn debug_state;
  void (*reset_debug_state)(void);
} TileLangMLXTVMFFICAPI;

TILELANG_MLX_TVM_FFI_EXPORT int
tilelang_mlx_tvm_ffi_get_c_api(uint32_t requested_version,
                               const char *requested_abi_hash,
                               TileLangMLXTVMFFICAPI *out, size_t out_size);

TILELANG_MLX_TVM_FFI_EXPORT int
tilelang_mlx_tvm_ffi_status(TileLangMLXTVMFFIStatus *out, size_t out_size);

TILELANG_MLX_TVM_FFI_EXPORT const char *
tilelang_mlx_tvm_ffi_c_api_abi_hash(void);
TILELANG_MLX_TVM_FFI_EXPORT const char *
tilelang_mlx_tvm_ffi_c_api_header_sha256(void);
TILELANG_MLX_TVM_FFI_EXPORT const char *tilelang_mlx_tvm_ffi_mlx_version(void);
TILELANG_MLX_TVM_FFI_EXPORT const char *
tilelang_mlx_tvm_ffi_mlx_lib_sha256(void);
TILELANG_MLX_TVM_FFI_EXPORT const char *
tilelang_mlx_tvm_ffi_mlx_python_bridge_sha256(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TILELANG_CONTRIB_MLX_TVM_FFI_C_API_H_
