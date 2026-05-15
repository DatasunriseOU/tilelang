#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include "contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h"

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

extern "C" PyObject* mlx_core_wrap_mx_array_move(mx::array* array);
extern "C" void TVMMetalSetExternalCommandBufferDirect(void* command_buffer);
extern "C" void TVMMetalClearExternalCommandBufferDirect();
extern "C" void TVMMetalSetExternalComputeEncoderDirect(void* compute_encoder);
extern "C" void TVMMetalClearExternalComputeEncoderDirect();
extern "C" void* TVMMetalCreateDirectLaunchHandle(
    TVMFFIObjectHandle module_handle, const char* func_name);
extern "C" void TVMMetalReleaseDirectLaunchHandle(void* handle);
extern "C" int TVMMetalDirectLaunch(
    void* handle, void** buffers, int32_t num_buffers, const int64_t* launch_args,
    int32_t num_launch_args);
extern "C" const char* TVMMetalDirectLaunchLastError();

namespace tilelang::mlx_tvm_ffi {

#ifndef TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256
#define TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256 "unknown"
#endif

#ifndef TILELANG_MLX_TVM_FFI_BUILD_MLX_VERSION
#define TILELANG_MLX_TVM_FFI_BUILD_MLX_VERSION "unknown"
#endif

#ifndef TILELANG_MLX_TVM_FFI_BUILD_MLX_LIB_SHA256
#define TILELANG_MLX_TVM_FFI_BUILD_MLX_LIB_SHA256 "unknown"
#endif

#ifndef TILELANG_MLX_TVM_FFI_BUILD_MLX_PY_BRIDGE_SHA256
#define TILELANG_MLX_TVM_FFI_BUILD_MLX_PY_BRIDGE_SHA256 "unknown"
#endif

#ifndef TILELANG_MLX_TVM_FFI_WITH_PY_MODULE
#define TILELANG_MLX_TVM_FFI_WITH_PY_MODULE 1
#endif

constexpr int32_t kDLMetalDeviceType = 8;
constexpr const char* kDebugCompletionEnv = "TILELANG_MLX_TVM_FFI_DEBUG_COMPLETION";
constexpr const char* kForceCommandBufferBoundaryEnv =
    "TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY";
constexpr const char* kForceOutputBarrierEnv =
    "TILELANG_MLX_TVM_FFI_FORCE_OUTPUT_BARRIER";
constexpr const char* kUseActiveComputeEncoderEnv =
    "TILELANG_MLX_TVM_FFI_USE_ACTIVE_COMPUTE_ENCODER";

struct DebugCounters {
  std::atomic<uint64_t> launches{0};
  std::atomic<uint64_t> debug_completion_launches{0};
  std::atomic<uint64_t> input_buffers_checked{0};
  std::atomic<uint64_t> output_buffers_checked{0};
  std::atomic<uint64_t> null_input_buffers{0};
  std::atomic<uint64_t> null_output_buffers{0};
  std::atomic<uint64_t> command_buffers_checked{0};
  std::atomic<uint64_t> null_command_buffers{0};
  std::atomic<uint64_t> completion_handlers_installed{0};
  std::atomic<uint64_t> completed_command_buffers{0};
  std::atomic<uint64_t> errored_command_buffers{0};
  std::atomic<uint64_t> device_event_waits_encoded{0};
  std::atomic<uint64_t> device_event_signals_encoded{0};
  std::atomic<uint64_t> direct_device_launches{0};
  std::atomic<uint64_t> direct_pipeline_launches{0};
  std::atomic<uint64_t> direct_compute_encoder_launches{0};
};

DebugCounters& debug_counters() {
  static DebugCounters counters;
  return counters;
}

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  std::string flag(value);
  return flag != "0" && flag != "false" && flag != "False" && flag != "FALSE";
}

bool env_flag_enabled_by_default(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  std::string flag(value);
  return flag != "0" && flag != "false" && flag != "False" && flag != "FALSE";
}

void reset_debug_counters() {
  auto& counters = debug_counters();
  counters.launches.store(0, std::memory_order_relaxed);
  counters.debug_completion_launches.store(0, std::memory_order_relaxed);
  counters.input_buffers_checked.store(0, std::memory_order_relaxed);
  counters.output_buffers_checked.store(0, std::memory_order_relaxed);
  counters.null_input_buffers.store(0, std::memory_order_relaxed);
  counters.null_output_buffers.store(0, std::memory_order_relaxed);
  counters.command_buffers_checked.store(0, std::memory_order_relaxed);
  counters.null_command_buffers.store(0, std::memory_order_relaxed);
  counters.completion_handlers_installed.store(0, std::memory_order_relaxed);
  counters.completed_command_buffers.store(0, std::memory_order_relaxed);
  counters.errored_command_buffers.store(0, std::memory_order_relaxed);
  counters.device_event_waits_encoded.store(0, std::memory_order_relaxed);
  counters.device_event_signals_encoded.store(0, std::memory_order_relaxed);
  counters.direct_device_launches.store(0, std::memory_order_relaxed);
  counters.direct_pipeline_launches.store(0, std::memory_order_relaxed);
  counters.direct_compute_encoder_launches.store(0, std::memory_order_relaxed);
}

nb::dict debug_state() {
  const auto& counters = debug_counters();
  nb::dict state;
  state["launches"] = counters.launches.load(std::memory_order_relaxed);
  state["debug_completion_launches"] =
      counters.debug_completion_launches.load(std::memory_order_relaxed);
  state["input_buffers_checked"] =
      counters.input_buffers_checked.load(std::memory_order_relaxed);
  state["output_buffers_checked"] =
      counters.output_buffers_checked.load(std::memory_order_relaxed);
  state["null_input_buffers"] =
      counters.null_input_buffers.load(std::memory_order_relaxed);
  state["null_output_buffers"] =
      counters.null_output_buffers.load(std::memory_order_relaxed);
  state["command_buffers_checked"] =
      counters.command_buffers_checked.load(std::memory_order_relaxed);
  state["null_command_buffers"] =
      counters.null_command_buffers.load(std::memory_order_relaxed);
  state["completion_handlers_installed"] =
      counters.completion_handlers_installed.load(std::memory_order_relaxed);
  state["completed_command_buffers"] =
      counters.completed_command_buffers.load(std::memory_order_relaxed);
  state["errored_command_buffers"] =
      counters.errored_command_buffers.load(std::memory_order_relaxed);
  state["device_event_waits_encoded"] =
      counters.device_event_waits_encoded.load(std::memory_order_relaxed);
  state["device_event_signals_encoded"] =
      counters.device_event_signals_encoded.load(std::memory_order_relaxed);
  state["direct_device_launches"] =
      counters.direct_device_launches.load(std::memory_order_relaxed);
  state["direct_pipeline_launches"] =
      counters.direct_pipeline_launches.load(std::memory_order_relaxed);
  state["direct_compute_encoder_launches"] =
      counters.direct_compute_encoder_launches.load(std::memory_order_relaxed);
  state["debug_completion_enabled"] = env_flag_enabled(kDebugCompletionEnv);
  state["force_command_buffer_boundary_enabled"] =
      env_flag_enabled(kForceCommandBufferBoundaryEnv);
  state["force_output_barrier_enabled"] =
      env_flag_enabled(kForceOutputBarrierEnv);
  state["use_active_compute_encoder_enabled"] =
      env_flag_enabled_by_default(kUseActiveComputeEncoderEnv);
  return state;
}

void install_completion_debug_hook(MTL::CommandBuffer* command_buffer) {
  debug_counters().completion_handlers_installed.fetch_add(1, std::memory_order_relaxed);
  command_buffer->addCompletedHandler(MTL::HandlerFunction([](MTL::CommandBuffer* completed) {
    auto& counters = debug_counters();
    if (completed != nullptr && completed->status() == MTL::CommandBufferStatusError) {
      counters.errored_command_buffers.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    counters.completed_command_buffers.fetch_add(1, std::memory_order_relaxed);
  }));
}

struct MetalSyncEdge : public std::enable_shared_from_this<MetalSyncEdge> {
  ~MetalSyncEdge() {
    std::lock_guard<std::mutex> lock(mutex);
    if (event != nullptr) {
      event->release();
      event = nullptr;
    }
  }

  MTL::Event* ensure_event(mx::Stream stream) {
    std::lock_guard<std::mutex> lock(mutex);
    if (event == nullptr) {
      auto* device = mx::metal::device(stream.device).mtl_device();
      event = device->newSharedEvent();
      if (event == nullptr) {
        throw std::runtime_error("failed to allocate Metal shared event for TVM-FFI edge");
      }
    }
    return static_cast<MTL::Event*>(event);
  }

  uint64_t value() const {
    return value_;
  }

 private:
  std::mutex mutex;
  MTL::SharedEvent* event{nullptr};
  uint64_t value_{1};
};

struct MetalLaunchSyncState : public std::enable_shared_from_this<MetalLaunchSyncState> {
  void add_signal_edge(std::shared_ptr<MetalSyncEdge> edge) {
    if (edge == nullptr) {
      throw std::runtime_error("cannot add a null Metal sync edge");
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (std::find(signal_edges.begin(), signal_edges.end(), edge) == signal_edges.end()) {
      signal_edges.push_back(std::move(edge));
    }
  }

  std::vector<std::shared_ptr<MetalSyncEdge>> snapshot_signal_edges() const {
    std::lock_guard<std::mutex> lock(mutex);
    return signal_edges;
  }

  size_t signal_edge_count() const {
    std::lock_guard<std::mutex> lock(mutex);
    return signal_edges.size();
  }

 private:
  mutable std::mutex mutex;
  std::vector<std::shared_ptr<MetalSyncEdge>> signal_edges;
};

std::shared_ptr<MetalLaunchSyncState> make_launch_sync_state() {
  return std::make_shared<MetalLaunchSyncState>();
}

std::shared_ptr<MetalSyncEdge> make_sync_edge() {
  return std::make_shared<MetalSyncEdge>();
}

struct CachedDirectLaunchHandle {
  CachedDirectLaunchHandle(TVMFFIObjectHandle module_handle, const std::string& kernel_name)
      : handle(TVMMetalCreateDirectLaunchHandle(module_handle, kernel_name.c_str())) {}

  ~CachedDirectLaunchHandle() {
    if (handle != nullptr) {
      TVMMetalReleaseDirectLaunchHandle(handle);
      handle = nullptr;
    }
  }

  void* handle{nullptr};
};

std::string direct_launch_handle_cache_key(
    TVMFFIObjectHandle module_handle, const std::string& kernel_name) {
  std::ostringstream os;
  os << static_cast<const void*>(module_handle) << '\n' << kernel_name;
  return os.str();
}

std::shared_ptr<CachedDirectLaunchHandle> get_cached_direct_launch_handle(
    TVMFFIObjectHandle module_handle, const std::string& kernel_name) {
  if (module_handle == nullptr || kernel_name.empty()) {
    return nullptr;
  }
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::shared_ptr<CachedDirectLaunchHandle>> cache;
  const std::string key = direct_launch_handle_cache_key(module_handle, kernel_name);
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  auto entry = std::make_shared<CachedDirectLaunchHandle>(module_handle, kernel_name);
  if (entry->handle == nullptr) {
    return nullptr;
  }
  cache.emplace(key, entry);
  return entry;
}

void encode_device_event_waits(
    MTL::CommandBuffer* command_buffer,
    mx::Stream stream,
    const std::vector<std::shared_ptr<MetalSyncEdge>>& edges) {
  for (const auto& edge : edges) {
    if (edge == nullptr) {
      continue;
    }
    command_buffer->encodeWait(edge->ensure_event(stream), edge->value());
    debug_counters().device_event_waits_encoded.fetch_add(1, std::memory_order_relaxed);
  }
}

void encode_device_event_signals(
    MTL::CommandBuffer* command_buffer,
    mx::Stream stream,
    const std::vector<std::shared_ptr<MetalSyncEdge>>& edges) {
  for (const auto& edge : edges) {
    if (edge == nullptr) {
      continue;
    }
    command_buffer->encodeSignalEvent(edge->ensure_event(stream), edge->value());
    debug_counters().device_event_signals_encoded.fetch_add(1, std::memory_order_relaxed);
  }
}

DLDataType mlx_dtype_to_dlpack(mx::Dtype dtype) {
  switch (dtype.val()) {
    case mx::Dtype::Val::bool_:
      return DLDataType{kDLBool, 8, 1};
    case mx::Dtype::Val::uint8:
      return DLDataType{kDLUInt, 8, 1};
    case mx::Dtype::Val::uint16:
      return DLDataType{kDLUInt, 16, 1};
    case mx::Dtype::Val::uint32:
      return DLDataType{kDLUInt, 32, 1};
    case mx::Dtype::Val::uint64:
      return DLDataType{kDLUInt, 64, 1};
    case mx::Dtype::Val::int8:
      return DLDataType{kDLInt, 8, 1};
    case mx::Dtype::Val::int16:
      return DLDataType{kDLInt, 16, 1};
    case mx::Dtype::Val::int32:
      return DLDataType{kDLInt, 32, 1};
    case mx::Dtype::Val::int64:
      return DLDataType{kDLInt, 64, 1};
    case mx::Dtype::Val::float16:
      return DLDataType{kDLFloat, 16, 1};
    case mx::Dtype::Val::float32:
      return DLDataType{kDLFloat, 32, 1};
    case mx::Dtype::Val::float64:
      return DLDataType{kDLFloat, 64, 1};
    case mx::Dtype::Val::bfloat16:
      return DLDataType{kDLBfloat, 16, 1};
    case mx::Dtype::Val::complex64:
      return DLDataType{kDLComplex, 64, 1};
  }
  throw std::runtime_error("unsupported MLX dtype for TVM-FFI DLTensor view");
}

std::string normalize_dtype_name(std::string name) {
  constexpr const char* mlx_prefix = "mlx.core.";
  constexpr const char* torch_prefix = "torch.";
  if (name.rfind(mlx_prefix, 0) == 0) {
    name.erase(0, std::char_traits<char>::length(mlx_prefix));
  } else if (name.rfind(torch_prefix, 0) == 0) {
    name.erase(0, std::char_traits<char>::length(torch_prefix));
  }
  if (name == "bool") {
    return "bool_";
  }
  return name;
}

mx::Dtype dtype_from_name(const std::string& raw_name) {
  const auto name = normalize_dtype_name(raw_name);
  if (name == "bool_") return mx::bool_;
  if (name == "uint8") return mx::uint8;
  if (name == "uint16") return mx::uint16;
  if (name == "uint32") return mx::uint32;
  if (name == "uint64") return mx::uint64;
  if (name == "int8") return mx::int8;
  if (name == "int16") return mx::int16;
  if (name == "int32") return mx::int32;
  if (name == "int64") return mx::int64;
  if (name == "float16") return mx::float16;
  if (name == "float32") return mx::float32;
  if (name == "float64") return mx::float64;
  if (name == "bfloat16") return mx::bfloat16;
  if (name == "complex64") return mx::complex64;
  if (name == "float8_e3m4" || name == "float8_e4m3" ||
      name == "float8_e4m3b11fnuz" || name == "float8_e4m3fn" ||
      name == "float8_e4m3fnuz" || name == "float8_e5m2" ||
      name == "float8_e5m2fnuz" || name == "float8_e8m0fnu") {
    return mx::uint8;
  }
  throw std::runtime_error("unsupported MLX output dtype for TVM-FFI bridge: " + raw_name);
}

bool dl_dtype_equal(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

bool is_fp8_dlpack_dtype(DLDataType dtype) {
  return dtype.bits == 8 && dtype.lanes == 1 &&
         (dtype.code == kDLFloat8_e3m4 || dtype.code == kDLFloat8_e4m3 ||
          dtype.code == kDLFloat8_e4m3b11fnuz || dtype.code == kDLFloat8_e4m3fn ||
          dtype.code == kDLFloat8_e4m3fnuz || dtype.code == kDLFloat8_e5m2 ||
          dtype.code == kDLFloat8_e5m2fnuz || dtype.code == kDLFloat8_e8m0fnu);
}

DLDataType expected_dlpack_dtype_from_name(const std::string& raw_name) {
  const auto name = normalize_dtype_name(raw_name);
  if (name == "bool_") return DLDataType{kDLBool, 8, 1};
  if (name == "uint8") return DLDataType{kDLUInt, 8, 1};
  if (name == "uint16") return DLDataType{kDLUInt, 16, 1};
  if (name == "uint32") return DLDataType{kDLUInt, 32, 1};
  if (name == "uint64") return DLDataType{kDLUInt, 64, 1};
  if (name == "int8") return DLDataType{kDLInt, 8, 1};
  if (name == "int16") return DLDataType{kDLInt, 16, 1};
  if (name == "int32") return DLDataType{kDLInt, 32, 1};
  if (name == "int64") return DLDataType{kDLInt, 64, 1};
  if (name == "float16") return DLDataType{kDLFloat, 16, 1};
  if (name == "float32") return DLDataType{kDLFloat, 32, 1};
  if (name == "float64") return DLDataType{kDLFloat, 64, 1};
  if (name == "bfloat16") return DLDataType{kDLBfloat, 16, 1};
  if (name == "complex64") return DLDataType{kDLComplex, 64, 1};
  if (name == "float8_e3m4") return DLDataType{kDLFloat8_e3m4, 8, 1};
  if (name == "float8_e4m3") return DLDataType{kDLFloat8_e4m3, 8, 1};
  if (name == "float8_e4m3b11fnuz") return DLDataType{kDLFloat8_e4m3b11fnuz, 8, 1};
  if (name == "float8_e4m3fn") return DLDataType{kDLFloat8_e4m3fn, 8, 1};
  if (name == "float8_e4m3fnuz") return DLDataType{kDLFloat8_e4m3fnuz, 8, 1};
  if (name == "float8_e5m2") return DLDataType{kDLFloat8_e5m2, 8, 1};
  if (name == "float8_e5m2fnuz") return DLDataType{kDLFloat8_e5m2fnuz, 8, 1};
  if (name == "float8_e8m0fnu") return DLDataType{kDLFloat8_e8m0fnu, 8, 1};
  throw std::runtime_error("unsupported expected TVM dtype for MLX TVM-FFI bridge: " + raw_name);
}

DLDataType dlpack_dtype_for_tensor_view(
    mx::Dtype actual_mlx_dtype,
    const std::string* expected_dtype_name) {
  const DLDataType actual = mlx_dtype_to_dlpack(actual_mlx_dtype);
  if (expected_dtype_name == nullptr || expected_dtype_name->empty() ||
      *expected_dtype_name == "None" || *expected_dtype_name == "none" ||
      *expected_dtype_name == "null") {
    return actual;
  }
  const DLDataType expected = expected_dlpack_dtype_from_name(*expected_dtype_name);
  if (dl_dtype_equal(actual, expected)) {
    return actual;
  }
  if (is_fp8_dlpack_dtype(expected) &&
      (actual.code == kDLUInt || actual.code == kDLInt) &&
      actual.bits == 8 && actual.lanes == 1) {
    return expected;
  }
  throw std::runtime_error(
      "MLX DLTensor dtype does not match expected TileLang ABI dtype: expected " +
      *expected_dtype_name + " for native TVM-FFI tensor view");
}

mx::Shape shape_from_i64(const std::vector<int64_t>& shape) {
  mx::Shape out;
  out.reserve(shape.size());
  for (int64_t dim : shape) {
    if (dim < 0 || dim > static_cast<int64_t>(std::numeric_limits<mx::ShapeElem>::max())) {
      throw std::runtime_error("MLX output shape dimension is out of range");
    }
    out.push_back(static_cast<mx::ShapeElem>(dim));
  }
  return out;
}

std::vector<int64_t> shape_to_i64(const mx::Shape& shape) {
  std::vector<int64_t> out;
  out.reserve(shape.size());
  for (mx::ShapeElem dim : shape) {
    out.push_back(static_cast<int64_t>(dim));
  }
  return out;
}

std::string python_type_name(nb::handle item) {
  PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(item.ptr()));
  nb::object module = nb::steal(PyObject_GetAttrString(type, "__module__"));
  if (!module.is_valid()) {
    PyErr_Clear();
    module = nb::str("<unknown>");
  }
  nb::object qualname = nb::steal(PyObject_GetAttrString(type, "__qualname__"));
  if (!qualname.is_valid()) {
    PyErr_Clear();
    qualname = nb::steal(PyObject_GetAttrString(type, "__name__"));
  }
  if (!qualname.is_valid()) {
    PyErr_Clear();
    qualname = nb::str("<unknown>");
  }
  const char* module_c = PyUnicode_AsUTF8(module.ptr());
  if (module_c == nullptr) {
    PyErr_Clear();
    module_c = "<unknown>";
  }
  const char* qualname_c = PyUnicode_AsUTF8(qualname.ptr());
  if (qualname_c == nullptr) {
    PyErr_Clear();
    qualname_c = "<unknown>";
  }
  return std::string(module_c) + "." + qualname_c;
}

mx::array parse_mlx_array(nb::handle item) {
  // MLX arrays are nanobind class instances. In mixed dev environments the
  // normal cross-module nb::cast<mx::array>() path can fail if the active MLX
  // wheel and this bridge were built in separate CMake trees, even when they
  // use the same NB_DOMAIN. For the exact mlx.core.array type, copying the
  // C++ mx::array handle out of the nanobind instance is still zero-copy with
  // respect to tensor storage and avoids any DLPack/eval boundary.
  const std::string type_name = python_type_name(item);
  if (type_name == "mlx.core.array") {
    auto* array = nb::inst_ptr<mx::array>(item);
    if (array == nullptr) {
      throw std::runtime_error("mlx.core.array instance has null C++ storage");
    }
    return *array;
  }
  try {
    return nb::cast<mx::array>(item);
  } catch (const std::exception& exc) {
    throw std::runtime_error(
        "expected mlx.core.array input, got " + type_name + ": " + exc.what());
  }
}

std::vector<mx::array> parse_array_sequence(nb::handle values) {
  std::vector<mx::array> arrays;
  for (nb::handle item : nb::iter(values)) {
    arrays.push_back(parse_mlx_array(item));
  }
  return arrays;
}

std::vector<std::vector<int64_t>> parse_shape_sequence(nb::handle values) {
  std::vector<std::vector<int64_t>> shapes;
  for (nb::handle shape_obj : nb::iter(values)) {
    std::vector<int64_t> shape;
    for (nb::handle dim_obj : nb::iter(shape_obj)) {
      shape.push_back(nb::cast<int64_t>(dim_obj));
    }
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

std::vector<std::string> parse_string_sequence(nb::handle values) {
  std::vector<std::string> strings;
  for (nb::handle item : nb::iter(values)) {
    strings.push_back(nb::cast<std::string>(item));
  }
  return strings;
}

std::vector<int64_t> parse_i64_sequence(nb::handle values) {
  std::vector<int64_t> integers;
  for (nb::handle item : nb::iter(values)) {
    integers.push_back(nb::cast<int64_t>(item));
  }
  return integers;
}

std::shared_ptr<MetalLaunchSyncState> parse_launch_sync_state(nb::handle handle) {
  if (handle.is_none()) {
    return make_launch_sync_state();
  }
  return nb::cast<std::shared_ptr<MetalLaunchSyncState>>(handle);
}

std::vector<std::shared_ptr<MetalSyncEdge>> parse_sync_edge_sequence(nb::handle handle) {
  std::vector<std::shared_ptr<MetalSyncEdge>> edges;
  if (handle.is_none()) {
    return edges;
  }
  for (nb::handle item : nb::iter(handle)) {
    auto edge = nb::cast<std::shared_ptr<MetalSyncEdge>>(item);
    if (edge != nullptr) {
      edges.push_back(std::move(edge));
    }
  }
  return edges;
}

struct BorrowedTensorView {
  DLTensor tensor{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

int64_t product_i64(const std::vector<int64_t>& shape) {
  int64_t product = 1;
  for (int64_t dim : shape) {
    if (dim < 0 || (dim != 0 && product > std::numeric_limits<int64_t>::max() / dim)) {
      throw std::runtime_error("tensor shape product is out of range");
    }
    product *= dim;
  }
  return product;
}

std::vector<int64_t> compact_strides_for_shape(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= std::max<int64_t>(shape[static_cast<size_t>(i)], 1);
  }
  return strides;
}

mx::Strides compact_mlx_strides_for_shape(const std::vector<int64_t>& shape) {
  auto vector_strides = compact_strides_for_shape(shape);
  mx::Strides strides;
  strides.reserve(vector_strides.size());
  for (int64_t stride : vector_strides) {
    strides.push_back(stride);
  }
  return strides;
}

bool is_compact_array(const mx::array& array) {
  const auto& shape = array.shape();
  const auto& strides = array.strides();
  if (shape.size() != strides.size()) {
    return false;
  }
  int64_t expected_stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    const int dim = shape[static_cast<size_t>(i)];
    const int64_t stride = static_cast<int64_t>(strides[static_cast<size_t>(i)]);
    if (dim > 1 && stride != expected_stride) {
      return false;
    }
    expected_stride *= static_cast<int64_t>(std::max(dim, 1));
  }
  return true;
}

bool can_borrow_compact_input_array(const mx::array& array) {
  if (array.is_tracer() || array.has_primitive()) {
    return false;
  }
  return array.buffer().ptr() != nullptr && array.offset() == 0 && is_compact_array(array);
}

BorrowedTensorView make_tensor_view(
    const mx::array& array,
    mx::Stream stream,
    bool is_output,
    const std::string* expected_dtype_name = nullptr,
    const std::vector<int64_t>* shape_override = nullptr) {
  void* buffer = const_cast<void*>(array.buffer().ptr());
  auto& counters = debug_counters();
  if (is_output) {
    counters.output_buffers_checked.fetch_add(1, std::memory_order_relaxed);
  } else {
    counters.input_buffers_checked.fetch_add(1, std::memory_order_relaxed);
  }
  if (buffer == nullptr) {
    if (is_output) {
      counters.null_output_buffers.fetch_add(1, std::memory_order_relaxed);
    } else {
      counters.null_input_buffers.fetch_add(1, std::memory_order_relaxed);
    }
    throw std::runtime_error(
        is_output
            ? "MLX output array has no materialized Metal buffer at TVM-FFI launch time"
            : "MLX input array has no materialized Metal buffer at TVM-FFI launch time");
  }
  if (!is_output && !is_compact_array(array)) {
    throw std::runtime_error(
        "MLX input array is not compact at TVM-FFI launch time; "
        "TileLang MLX graph lowering must insert a compact-layout mapping");
  }

  BorrowedTensorView view;
  if (shape_override != nullptr) {
    view.shape = *shape_override;
    if (product_i64(view.shape) != static_cast<int64_t>(array.size())) {
      throw std::runtime_error(
          is_output
              ? "MLX owner output size does not match TileLang result ABI shape"
              : "MLX input size does not match TileLang parameter ABI shape");
    }
    view.strides = compact_strides_for_shape(view.shape);
  } else {
    view.shape.reserve(array.ndim());
    for (auto dim : array.shape()) {
      view.shape.push_back(static_cast<int64_t>(dim));
    }
    view.strides.assign(array.strides().begin(), array.strides().end());
  }

  view.tensor.data = buffer;
  view.tensor.device = DLDevice{
      static_cast<DLDeviceType>(kDLMetalDeviceType),
      stream.device.index};
  view.tensor.ndim = static_cast<int>(view.shape.size());
  view.tensor.dtype = dlpack_dtype_for_tensor_view(array.dtype(), expected_dtype_name);
  view.tensor.shape = view.shape.data();
  view.tensor.strides = view.strides.data();
  view.tensor.byte_offset = static_cast<uint64_t>(array.offset());
  return view;
}

void* make_direct_opaque_ptr(
    const mx::array& array,
    bool is_output,
    int64_t param_idx,
    const std::string* expected_dtype_name = nullptr,
    const std::vector<int64_t>* shape_override = nullptr) {
  void* buffer = const_cast<void*>(array.buffer().ptr());
  auto& counters = debug_counters();
  if (is_output) {
    counters.output_buffers_checked.fetch_add(1, std::memory_order_relaxed);
  } else {
    counters.input_buffers_checked.fetch_add(1, std::memory_order_relaxed);
  }
  if (buffer == nullptr) {
    if (is_output) {
      counters.null_output_buffers.fetch_add(1, std::memory_order_relaxed);
    } else {
      counters.null_input_buffers.fetch_add(1, std::memory_order_relaxed);
    }
    std::ostringstream os;
    os << "TVM-FFI parameter " << param_idx << (is_output ? " output" : " input")
       << " has no materialized Metal buffer at direct launch time";
    throw std::runtime_error(os.str());
  }
  if (!is_output && !is_compact_array(array)) {
    std::ostringstream os;
    os << "TVM-FFI parameter " << param_idx
       << " input is not compact at direct launch time; "
       << "TileLang MLX graph lowering must insert a compact-layout mapping";
    throw std::runtime_error(os.str());
  }
  if (shape_override != nullptr &&
      product_i64(*shape_override) != static_cast<int64_t>(array.size())) {
    std::ostringstream os;
    os << "TVM-FFI parameter " << param_idx
       << (is_output ? " output" : " input")
       << " size does not match TileLang direct-launch ABI shape";
    throw std::runtime_error(os.str());
  }
  (void)dlpack_dtype_for_tensor_view(array.dtype(), expected_dtype_name);
  if (array.offset() != 0) {
    std::ostringstream os;
    os << "TVM-FFI parameter " << param_idx
       << " has non-zero MLX buffer offset; direct Metal device launch "
       << "requires compact zero-offset buffers";
    throw std::runtime_error(os.str());
  }
  return buffer;
}

mx::array compact_input_array(const mx::array& array) {
  return mx::contiguous(array, false, mx::default_stream(mx::Device::gpu));
}

struct ExternalCommandBufferScope {
  explicit ExternalCommandBufferScope(void* command_buffer) {
    auto& counters = debug_counters();
    counters.command_buffers_checked.fetch_add(1, std::memory_order_relaxed);
    if (command_buffer == nullptr) {
      counters.null_command_buffers.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("MLX returned a null Metal command buffer for TVM-FFI launch");
    }
    TVMMetalSetExternalCommandBufferDirect(command_buffer);
  }

  ~ExternalCommandBufferScope() {
    TVMMetalClearExternalCommandBufferDirect();
  }
};

struct ExternalComputeEncoderScope {
  explicit ExternalComputeEncoderScope(void* compute_encoder) {
    if (compute_encoder == nullptr) {
      throw std::runtime_error("MLX returned a null Metal compute encoder for TVM-FFI launch");
    }
    TVMMetalSetExternalComputeEncoderDirect(compute_encoder);
  }

  ~ExternalComputeEncoderScope() {
    TVMMetalClearExternalComputeEncoderDirect();
  }
};

MTL::CommandBuffer* finish_encoding_and_get_command_buffer(mx::Stream stream) {
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.end_encoding();
  return encoder.get_command_buffer();
}

void zero_output_buffer(MTL::CommandBuffer* command_buffer, mx::array& out) {
  if (out.nbytes() == 0) {
    return;
  }
  auto* buffer = static_cast<MTL::Buffer*>(const_cast<void*>(out.buffer().ptr()));
  auto* blit_encoder = command_buffer->blitCommandEncoder();
  blit_encoder->fillBuffer(buffer, NS::Range::Make(out.offset(), out.nbytes()), 0);
  blit_encoder->endEncoding();
}

void encode_command_buffer_boundary(MTL::CommandBuffer* command_buffer, mx::Stream stream) {
  auto* device = mx::metal::device(stream.device).mtl_device();
  auto* event = device->newSharedEvent();
  if (event == nullptr) {
    throw std::runtime_error("failed to allocate Metal shared event for TVM-FFI boundary");
  }
  constexpr uint64_t kBoundaryValue = 1;
  command_buffer->encodeSignalEvent(event, kBoundaryValue);
  command_buffer->encodeWait(event, kBoundaryValue);
  command_buffer->addCompletedHandler([event](MTL::CommandBuffer*) { event->release(); });
}

void maybe_encode_command_buffer_boundary(MTL::CommandBuffer* command_buffer, mx::Stream stream) {
  if (!env_flag_enabled(kForceCommandBufferBoundaryEnv)) {
    return;
  }
  encode_command_buffer_boundary(command_buffer, stream);
}

mx::metal::CommandEncoder* get_command_encoder(mx::Stream stream) {
  return &mx::metal::get_command_encoder(stream);
}

void register_external_inputs(
    mx::metal::CommandEncoder* encoder,
    const std::vector<mx::array>& inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    encoder->set_input_array(inputs[i], static_cast<int>(i));
  }
}

void register_external_inputs_ptr(
    mx::metal::CommandEncoder* encoder,
    const std::vector<const mx::array*>& inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    encoder->set_input_array(*inputs[i], static_cast<int>(i));
  }
}

void publish_external_outputs(
    mx::Stream stream,
    const std::vector<mx::array>& outputs) {
  if (outputs.empty()) {
    return;
  }
  auto* encoder = get_command_encoder(stream);
  for (const auto& out : outputs) {
    encoder->register_output_array(out);
  }
  if (env_flag_enabled(kForceOutputBarrierEnv)) {
    encoder->barrier();
    encoder->end_encoding();
  }
}

struct PreparedMetalCall : public std::enable_shared_from_this<PreparedMetalCall> {
  ~PreparedMetalCall() {
    if (func_handle_ptr != nullptr) {
      (void)TVMFFIObjectDecRef(func_handle_ptr);
      func_handle_ptr = nullptr;
    }
  }

  uint64_t func_handle{0};
  TVMFFIObjectHandle func_handle_ptr{nullptr};
  int64_t num_params{0};
  std::vector<int64_t> result_indices;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<mx::Shape> output_mlx_shapes;
  std::vector<std::string> output_dtypes;
  std::vector<mx::Dtype> output_mlx_dtypes;
  std::vector<std::string> expected_param_dtypes;
  std::vector<std::vector<int64_t>> param_shapes;
  std::vector<int64_t> direct_launch_args;
  std::vector<int64_t> direct_param_indices;
  uint64_t direct_module_handle{0};
  std::string direct_kernel_name;
  std::shared_ptr<CachedDirectLaunchHandle> direct_launch_handle;
  std::vector<int64_t> zero_init_output_positions;
};

class TVMFFIMetalCall : public mx::Primitive {
 public:
  TVMFFIMetalCall(
      mx::Stream stream,
      uint64_t func_handle,
      int64_t num_params,
      std::vector<int64_t> result_indices,
      std::vector<std::vector<int64_t>> output_shapes,
      std::vector<std::string> expected_param_dtypes,
      std::vector<std::vector<int64_t>> param_shapes,
      std::vector<int64_t> direct_launch_args,
      std::vector<int64_t> direct_param_indices,
      uint64_t direct_module_handle,
      std::string direct_kernel_name,
      std::vector<int64_t> zero_init_output_positions,
      bool owner_outputs_are_inputs,
      std::shared_ptr<MetalLaunchSyncState> launch_sync_state,
      std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges)
      : mx::Primitive(stream),
        func_handle_(reinterpret_cast<TVMFFIObjectHandle>(func_handle)),
        num_params_(num_params),
        result_indices_(std::move(result_indices)),
        output_shapes_(std::move(output_shapes)),
        expected_param_dtypes_(std::move(expected_param_dtypes)),
        param_shapes_(std::move(param_shapes)),
        direct_launch_args_(std::move(direct_launch_args)),
        direct_param_indices_(std::move(direct_param_indices)),
        direct_module_handle_(reinterpret_cast<TVMFFIObjectHandle>(direct_module_handle)),
        direct_kernel_name_(std::move(direct_kernel_name)),
        zero_init_output_positions_(std::move(zero_init_output_positions)),
        owner_outputs_are_inputs_(owner_outputs_are_inputs),
        launch_sync_state_(std::move(launch_sync_state)),
        wait_edges_(std::move(wait_edges)) {
    if (func_handle_ == nullptr) {
      throw std::runtime_error("TVM-FFI function handle is null");
    }
    if (launch_sync_state_ == nullptr) {
      launch_sync_state_ = make_launch_sync_state();
    }
    if (num_params_ <= 0) {
      throw std::runtime_error("TVM-FFI bridge requires a positive parameter count");
    }
    if (!expected_param_dtypes_.empty() &&
        static_cast<int64_t>(expected_param_dtypes_.size()) != num_params_) {
      throw std::runtime_error("TVM-FFI expected param dtype metadata length mismatch");
    }
    if (!param_shapes_.empty() &&
        static_cast<int64_t>(param_shapes_.size()) != num_params_) {
      throw std::runtime_error("TVM-FFI parameter shape metadata length mismatch");
    }
    std::sort(result_indices_.begin(), result_indices_.end());
    result_indices_.erase(
        std::unique(result_indices_.begin(), result_indices_.end()),
        result_indices_.end());
    std::sort(zero_init_output_positions_.begin(), zero_init_output_positions_.end());
    zero_init_output_positions_.erase(
        std::unique(zero_init_output_positions_.begin(), zero_init_output_positions_.end()),
        zero_init_output_positions_.end());
    for (int64_t idx : result_indices_) {
      if (idx < 0 || idx >= num_params_) {
        throw std::runtime_error("TVM-FFI result index is outside the parameter list");
      }
    }
    if (output_shapes_.size() != result_indices_.size()) {
      throw std::runtime_error("TVM-FFI output shape metadata/result_idx length mismatch");
    }
    for (int64_t idx : zero_init_output_positions_) {
      if (idx < 0 || idx >= static_cast<int64_t>(result_indices_.size())) {
        throw std::runtime_error("TVM-FFI zero-init output position is outside the output list");
      }
    }
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectIncRef(func_handle_));
    if (direct_module_handle_ != nullptr && !direct_kernel_name_.empty() &&
        !direct_launch_args_.empty()) {
      direct_launch_handle_ =
          get_cached_direct_launch_handle(direct_module_handle_, direct_kernel_name_);
    }
  }

  TVMFFIMetalCall(
      mx::Stream stream,
      std::shared_ptr<PreparedMetalCall> prepared,
      bool owner_outputs_are_inputs,
      std::shared_ptr<MetalLaunchSyncState> launch_sync_state,
      std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges)
      : mx::Primitive(stream),
        prepared_(std::move(prepared)),
        owner_outputs_are_inputs_(owner_outputs_are_inputs),
        launch_sync_state_(std::move(launch_sync_state)),
        wait_edges_(std::move(wait_edges)),
        owns_func_handle_ref_(false) {
    if (prepared_ == nullptr || prepared_->func_handle_ptr == nullptr) {
      throw std::runtime_error("prepared TVM-FFI Metal call is null");
    }
    func_handle_ = prepared_->func_handle_ptr;
    num_params_ = prepared_->num_params;
    direct_launch_handle_ = prepared_->direct_launch_handle;
    if (launch_sync_state_ == nullptr) {
      launch_sync_state_ = make_launch_sync_state();
    }
  }

  ~TVMFFIMetalCall() override {
    if (owns_func_handle_ref_ && func_handle_ != nullptr) {
      (void)TVMFFIObjectDecRef(func_handle_);
    }
  }

  const char* name() const override {
    return "TVMFFIMetalCall";
  }

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("TVM-FFI Metal bridge cannot run on CPU");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& result_indices = prepared_ ? prepared_->result_indices : result_indices_;
    const auto& output_shapes = prepared_ ? prepared_->output_shapes : output_shapes_;
    const auto& expected_param_dtypes =
        prepared_ ? prepared_->expected_param_dtypes : expected_param_dtypes_;
    const auto& param_shapes = prepared_ ? prepared_->param_shapes : param_shapes_;
    const auto& direct_launch_args =
        prepared_ ? prepared_->direct_launch_args : direct_launch_args_;
    const auto& direct_param_indices =
        prepared_ ? prepared_->direct_param_indices : direct_param_indices_;
    const auto& zero_init_output_positions =
        prepared_ ? prepared_->zero_init_output_positions : zero_init_output_positions_;
    debug_counters().launches.fetch_add(1, std::memory_order_relaxed);
    if (outputs.size() != result_indices.size()) {
      throw std::runtime_error("TVM-FFI bridge output count mismatch");
    }
    const int64_t expected_inputs = num_params_ - static_cast<int64_t>(result_indices.size());
    const int64_t expected_primitive_inputs =
        expected_inputs +
        (owner_outputs_are_inputs_ ? static_cast<int64_t>(result_indices.size()) : 0);
    if (static_cast<int64_t>(inputs.size()) != expected_primitive_inputs) {
      throw std::runtime_error("TVM-FFI bridge input count mismatch");
    }

    auto* encoder = get_command_encoder(stream());
    // Compact-input contract: callers must pass row-contiguous inputs. The
    // Python adapter wraps non-compact inputs with mx.contiguous() at graph
    // construction time so MLX scheduler materializes copies before this
    // primitive's eval_gpu runs. We can't call mx::eval here -- we are
    // already on the scheduler thread and would deadlock.
    if (owner_outputs_are_inputs_) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& owner = inputs.at(static_cast<size_t>(expected_inputs) + i);
        if (owner.buffer().ptr() == nullptr) {
          throw std::runtime_error(
              "MLX owner output array has no materialized Metal buffer at TVM-FFI launch time");
        }
        if (!is_compact_array(owner)) {
          throw std::runtime_error(
              "MLX owner output array is not compact at TVM-FFI launch time");
        }
        const std::vector<int64_t> owner_shape = shape_to_i64(owner.shape());
        outputs.at(i).copy_shared_buffer(
            owner,
            compact_mlx_strides_for_shape(owner_shape),
            mx::array::Flags{true, true, true},
            static_cast<size_t>(owner.size()),
            owner.offset());
      }
    } else {
      for (auto& out : outputs) {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        if (out.buffer().ptr() == nullptr) {
          throw std::runtime_error("MLX failed to allocate TVM-FFI output buffer");
        }
      }
    }

    const bool direct_device_launch = !direct_launch_args.empty();
    const bool direct_pipeline_launch =
        direct_device_launch &&
        direct_launch_handle_ != nullptr &&
        direct_launch_handle_->handle != nullptr;
    if (direct_pipeline_launch) {
      std::vector<void*> direct_buffers;
      direct_buffers.reserve(static_cast<size_t>(num_params_));

      auto result_position_for_param = [&](int64_t param_idx) -> int64_t {
        auto it = std::lower_bound(result_indices.begin(), result_indices.end(), param_idx);
        if (it == result_indices.end() || *it != param_idx) {
          return -1;
        }
        return static_cast<int64_t>(std::distance(result_indices.begin(), it));
      };
      auto input_position_for_param = [&](int64_t param_idx) -> int64_t {
        auto it = std::lower_bound(result_indices.begin(), result_indices.end(), param_idx);
        return param_idx - static_cast<int64_t>(std::distance(result_indices.begin(), it));
      };
      auto bind_direct_param = [&](int64_t param_idx) -> void* {
        const int64_t output_pos = result_position_for_param(param_idx);
        const bool is_output = output_pos >= 0;
        const mx::array* array = nullptr;
        const std::vector<int64_t>* shape_override = nullptr;
        if (is_output) {
          array = &outputs.at(static_cast<size_t>(output_pos));
          if (!param_shapes.empty()) {
            shape_override = &param_shapes.at(static_cast<size_t>(param_idx));
          } else {
            shape_override = &output_shapes.at(static_cast<size_t>(output_pos));
          }
        } else {
          const int64_t input_pos = input_position_for_param(param_idx);
          if (input_pos < 0 || input_pos >= expected_inputs) {
            throw std::runtime_error("TVM-FFI direct Metal input position is out of range");
          }
          array = &inputs.at(static_cast<size_t>(input_pos));
          if (!param_shapes.empty()) {
            shape_override = &param_shapes.at(static_cast<size_t>(param_idx));
          }
        }
        const std::string* expected_dtype = expected_param_dtypes.empty()
            ? nullptr
            : &expected_param_dtypes.at(static_cast<size_t>(param_idx));
        try {
          return make_direct_opaque_ptr(
              *array,
              is_output,
              param_idx,
              expected_dtype,
              shape_override);
        } catch (const std::exception& exc) {
          std::ostringstream os;
          os << "TVM-FFI parameter " << param_idx << (is_output ? " output" : " input")
             << " failed direct Metal pointer binding: " << exc.what();
          throw std::runtime_error(os.str());
        }
      };

      if (!direct_param_indices.empty()) {
        for (int64_t param_idx : direct_param_indices) {
          direct_buffers.push_back(bind_direct_param(param_idx));
        }
      } else {
        for (int64_t param_idx = 0; param_idx < num_params_; ++param_idx) {
          direct_buffers.push_back(bind_direct_param(param_idx));
        }
      }

      register_external_inputs(encoder, inputs);
      auto signal_edges = launch_sync_state_->snapshot_signal_edges();
      auto call_direct = [&]() {
        debug_counters().direct_device_launches.fetch_add(1, std::memory_order_relaxed);
        debug_counters().direct_pipeline_launches.fetch_add(1, std::memory_order_relaxed);
        int rc = TVMMetalDirectLaunch(
            direct_launch_handle_->handle,
            direct_buffers.data(),
            static_cast<int32_t>(direct_buffers.size()),
            direct_launch_args.data(),
            static_cast<int32_t>(direct_launch_args.size()));
        if (rc != 0) {
          const char* error = TVMMetalDirectLaunchLastError();
          throw std::runtime_error(
              std::string("TVM Metal direct pipeline launch failed inside MLX graph eval: ") +
              (error == nullptr ? "unknown error" : error));
        }
      };
      const bool can_launch_on_active_compute_encoder =
          env_flag_enabled_by_default(kUseActiveComputeEncoderEnv) &&
          zero_init_output_positions.empty() &&
          wait_edges_.empty() &&
          signal_edges.empty() &&
          !env_flag_enabled(kDebugCompletionEnv) &&
          !env_flag_enabled(kForceCommandBufferBoundaryEnv);
      if (can_launch_on_active_compute_encoder) {
        encoder->prepare_external_dispatch();
        {
          ExternalComputeEncoderScope external(encoder->raw_command_encoder());
          debug_counters().direct_compute_encoder_launches.fetch_add(1, std::memory_order_relaxed);
          call_direct();
        }
        publish_external_outputs(stream(), outputs);
        return;
      }

      auto* command_buffer = finish_encoding_and_get_command_buffer(stream());
      maybe_encode_command_buffer_boundary(command_buffer, stream());
      encode_device_event_waits(command_buffer, stream(), wait_edges_);
      for (int64_t output_pos : zero_init_output_positions) {
        zero_output_buffer(command_buffer, outputs.at(static_cast<size_t>(output_pos)));
      }
      maybe_encode_command_buffer_boundary(command_buffer, stream());
      {
        ExternalCommandBufferScope external(command_buffer);
        call_direct();
      }
      encode_device_event_signals(command_buffer, stream(), signal_edges);
      maybe_encode_command_buffer_boundary(command_buffer, stream());
      publish_external_outputs(stream(), outputs);
      if (env_flag_enabled(kDebugCompletionEnv)) {
        debug_counters().debug_completion_launches.fetch_add(1, std::memory_order_relaxed);
        install_completion_debug_hook(command_buffer);
        for (const auto& out : outputs) {
          if (out.buffer().ptr() == nullptr) {
            debug_counters().null_output_buffers.fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error(
                "MLX output buffer became null after TVM-FFI Metal debug hook install");
          }
        }
      }
      return;
    }
    std::vector<int64_t> direct_param_order;
    std::vector<int64_t> direct_arg_position_by_param;
    if (direct_device_launch) {
      if (!direct_param_indices.empty()) {
        if (static_cast<int64_t>(direct_param_indices.size()) != num_params_) {
          throw std::runtime_error(
              "TVM-FFI direct Metal parameter permutation length mismatch");
        }
        direct_param_order = direct_param_indices;
      } else {
        direct_param_order.reserve(static_cast<size_t>(num_params_));
        for (int64_t i = 0; i < num_params_; ++i) {
          direct_param_order.push_back(i);
        }
      }
      direct_arg_position_by_param.assign(static_cast<size_t>(num_params_), -1);
      for (size_t pos = 0; pos < direct_param_order.size(); ++pos) {
        const int64_t param_idx = direct_param_order[pos];
        if (param_idx < 0 || param_idx >= num_params_) {
          throw std::runtime_error("TVM-FFI direct Metal parameter index out of range");
        }
        auto& mapped_pos = direct_arg_position_by_param[static_cast<size_t>(param_idx)];
        if (mapped_pos >= 0) {
          throw std::runtime_error("TVM-FFI direct Metal parameter permutation has duplicates");
        }
        mapped_pos = static_cast<int64_t>(pos);
      }
      for (int64_t pos : direct_arg_position_by_param) {
        if (pos < 0) {
          throw std::runtime_error("TVM-FFI direct Metal parameter permutation is incomplete");
        }
      }
    }
    const size_t arg_count = static_cast<size_t>(num_params_) +
                             (direct_device_launch ? direct_launch_args.size() : 0);
    std::vector<void*> direct_buffers;
    std::vector<void*> direct_buffers_by_param;
    if (direct_device_launch) {
      direct_buffers.reserve(static_cast<size_t>(num_params_));
      direct_buffers_by_param.assign(static_cast<size_t>(num_params_), nullptr);
    }
    std::vector<TVMFFIAny> args;
    if (!direct_pipeline_launch) {
      args.resize(arg_count);
    }
    std::vector<BorrowedTensorView> views;
    if (!direct_device_launch) {
      views.reserve(num_params_);
    }
    size_t input_pos = 0;
    size_t output_pos = 0;
    for (int64_t param_idx = 0; param_idx < num_params_; ++param_idx) {
      const bool is_output = std::binary_search(
          result_indices.begin(),
          result_indices.end(),
          param_idx);
      const std::vector<int64_t>* shape_override = nullptr;
      const mx::array& array =
          is_output ? outputs.at(output_pos)
                    : inputs.at(input_pos++);
      if (is_output) {
        if (!param_shapes.empty()) {
          shape_override = &param_shapes.at(static_cast<size_t>(param_idx));
        } else {
          shape_override = &output_shapes.at(output_pos);
        }
        ++output_pos;
      } else if (!param_shapes.empty()) {
        shape_override = &param_shapes.at(static_cast<size_t>(param_idx));
      }
      const std::string* expected_dtype = expected_param_dtypes.empty()
          ? nullptr
          : &expected_param_dtypes.at(static_cast<size_t>(param_idx));
      if (direct_device_launch) {
        TVMFFIAny* arg = direct_pipeline_launch
            ? nullptr
            : &args[static_cast<size_t>(
                  direct_arg_position_by_param[static_cast<size_t>(param_idx)])];
        if (arg != nullptr) {
          arg->type_index = kTVMFFIOpaquePtr;
          arg->zero_padding = 0;
        }
        try {
          void* ptr = make_direct_opaque_ptr(
              array,
              is_output,
              param_idx,
              expected_dtype,
              shape_override);
          if (arg != nullptr) {
            arg->v_ptr = ptr;
          }
          direct_buffers_by_param[static_cast<size_t>(param_idx)] = ptr;
        } catch (const std::exception& exc) {
          std::ostringstream os;
          os << "TVM-FFI parameter " << param_idx << (is_output ? " output" : " input")
             << " failed direct Metal pointer binding: " << exc.what();
          throw std::runtime_error(os.str());
        }
      } else {
        TVMFFIAny& arg = args[static_cast<size_t>(param_idx)];
        arg.zero_padding = 0;
        try {
          views.push_back(
              make_tensor_view(array, stream(), is_output, expected_dtype, shape_override));
        } catch (const std::exception& exc) {
          std::ostringstream os;
          os << "TVM-FFI parameter " << param_idx << (is_output ? " output" : " input")
             << " failed DLTensor binding: " << exc.what();
          throw std::runtime_error(os.str());
        }
        arg.type_index = kTVMFFIDLTensorPtr;
        arg.v_ptr = &views.back().tensor;
      }
    }
    if (direct_device_launch) {
      for (int64_t param_idx : direct_param_order) {
        void* ptr = direct_buffers_by_param[static_cast<size_t>(param_idx)];
        if (ptr == nullptr) {
          throw std::runtime_error("TVM-FFI direct Metal parameter pointer is null");
        }
        direct_buffers.push_back(ptr);
      }
    }
    if (direct_device_launch && !direct_pipeline_launch) {
      for (size_t i = 0; i < direct_launch_args.size(); ++i) {
        TVMFFIAny& arg = args[static_cast<size_t>(num_params_) + i];
        arg.type_index = kTVMFFIInt;
        arg.zero_padding = 0;
        arg.v_int64 = direct_launch_args[i];
      }
    }

    register_external_inputs(encoder, inputs);
    auto signal_edges = launch_sync_state_->snapshot_signal_edges();
    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;
    auto call_tvm = [&]() {
      try {
        if (direct_device_launch) {
          debug_counters().direct_device_launches.fetch_add(1, std::memory_order_relaxed);
        }
        if (direct_pipeline_launch) {
          debug_counters().direct_pipeline_launches.fetch_add(1, std::memory_order_relaxed);
          int rc = TVMMetalDirectLaunch(
              direct_launch_handle_->handle,
              direct_buffers.data(),
              static_cast<int32_t>(direct_buffers.size()),
              direct_launch_args.data(),
              static_cast<int32_t>(direct_launch_args.size()));
          if (rc != 0) {
            const char* error = TVMMetalDirectLaunchLastError();
            throw std::runtime_error(
                std::string("TVM Metal direct pipeline launch failed inside MLX graph eval: ") +
                (error == nullptr ? "unknown error" : error));
          }
        } else {
          TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionCall(
              func_handle_,
              args.data(),
              static_cast<int32_t>(args.size()),
              &result));
        }
      } catch (const tvm::ffi::Error& exc) {
        throw std::runtime_error(
            std::string("TVM-FFI function call failed inside MLX graph eval: ") +
            exc.FullMessage());
      }
    };
    auto release_result = [&]() {
      if (result.type_index >= kTVMFFIStaticObjectBegin && result.v_obj != nullptr) {
        TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectDecRef(result.v_obj));
        result.type_index = kTVMFFINone;
        result.v_obj = nullptr;
      }
    };
    const bool can_launch_on_active_compute_encoder =
        direct_device_launch &&
        env_flag_enabled_by_default(kUseActiveComputeEncoderEnv) &&
        zero_init_output_positions.empty() &&
        wait_edges_.empty() &&
        signal_edges.empty() &&
        !env_flag_enabled(kDebugCompletionEnv) &&
        !env_flag_enabled(kForceCommandBufferBoundaryEnv);
    if (can_launch_on_active_compute_encoder) {
      encoder->prepare_external_dispatch();
      {
        ExternalComputeEncoderScope external(encoder->raw_command_encoder());
        debug_counters().direct_compute_encoder_launches.fetch_add(1, std::memory_order_relaxed);
        call_tvm();
      }
      publish_external_outputs(stream(), outputs);
      release_result();
      return;
    }
    auto* command_buffer = finish_encoding_and_get_command_buffer(stream());
    maybe_encode_command_buffer_boundary(command_buffer, stream());
    encode_device_event_waits(command_buffer, stream(), wait_edges_);
    for (int64_t output_pos : zero_init_output_positions) {
      zero_output_buffer(command_buffer, outputs.at(static_cast<size_t>(output_pos)));
    }
    maybe_encode_command_buffer_boundary(command_buffer, stream());

    {
      ExternalCommandBufferScope external(command_buffer);
      call_tvm();
    }
    encode_device_event_signals(command_buffer, stream(), signal_edges);
    maybe_encode_command_buffer_boundary(command_buffer, stream());
    publish_external_outputs(stream(), outputs);
    release_result();
    if (env_flag_enabled(kDebugCompletionEnv)) {
      debug_counters().debug_completion_launches.fetch_add(1, std::memory_order_relaxed);
      install_completion_debug_hook(command_buffer);
      for (const auto& out : outputs) {
        if (out.buffer().ptr() == nullptr) {
          debug_counters().null_output_buffers.fetch_add(1, std::memory_order_relaxed);
          throw std::runtime_error(
              "MLX output buffer became null after TVM-FFI Metal debug hook install");
        }
      }
    }
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    // The underlying TVM call is mathematically pure but operationally opaque:
    // it borrows MLX's current command buffer and writes caller-owned Metal
    // output buffers. Do not let MLX CSE/merge distinct launch nodes.
    return this == &other;
  }

 private:
  TVMFFIObjectHandle func_handle_;
  int64_t num_params_;
  std::shared_ptr<PreparedMetalCall> prepared_;
  std::vector<int64_t> result_indices_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<std::string> expected_param_dtypes_;
  std::vector<std::vector<int64_t>> param_shapes_;
  std::vector<int64_t> direct_launch_args_;
  std::vector<int64_t> direct_param_indices_;
  TVMFFIObjectHandle direct_module_handle_{nullptr};
  std::string direct_kernel_name_;
  std::shared_ptr<CachedDirectLaunchHandle> direct_launch_handle_;
  std::vector<int64_t> zero_init_output_positions_;
  bool owner_outputs_are_inputs_;
  std::shared_ptr<MetalLaunchSyncState> launch_sync_state_;
  std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges_;
  bool owns_func_handle_ref_{true};
};

std::vector<mx::array> tvm_ffi_metal_call(
    uint64_t func_handle,
    const std::vector<mx::array>& inputs,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<std::string>& output_dtypes,
    const std::vector<int64_t>& result_indices,
    int64_t num_params,
    const std::vector<std::string>& expected_param_dtypes,
    const std::vector<std::vector<int64_t>>& param_shapes,
    const std::vector<int64_t>& direct_launch_args,
    const std::vector<int64_t>& direct_param_indices,
    uint64_t direct_module_handle,
    const std::string& direct_kernel_name,
    const std::vector<int64_t>& zero_init_output_positions,
    std::shared_ptr<MetalLaunchSyncState> launch_sync_state,
    std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges,
    const std::vector<mx::array>* owner_outputs = nullptr) {
  if (output_shapes.size() != output_dtypes.size()) {
    throw std::runtime_error("TVM-FFI output_shapes/output_dtypes length mismatch");
  }
  if (output_shapes.size() != result_indices.size()) {
    throw std::runtime_error("TVM-FFI output metadata/result_idx length mismatch");
  }

  std::vector<mx::Shape> shapes;
  shapes.reserve(output_shapes.size());
  std::vector<mx::Dtype> dtypes;
  dtypes.reserve(output_dtypes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    shapes.push_back(shape_from_i64(output_shapes[i]));
    dtypes.push_back(dtype_from_name(output_dtypes[i]));
  }

  auto primitive = std::make_shared<TVMFFIMetalCall>(
      mx::default_stream(mx::Device::gpu),
      func_handle,
      num_params,
      result_indices,
      output_shapes,
      expected_param_dtypes,
      param_shapes,
      direct_launch_args,
      direct_param_indices,
      direct_module_handle,
      direct_kernel_name,
      zero_init_output_positions,
      owner_outputs != nullptr,
      std::move(launch_sync_state),
      std::move(wait_edges));
  std::vector<mx::array> primitive_inputs = inputs;
  if (owner_outputs != nullptr) {
    primitive_inputs.insert(
        primitive_inputs.end(), owner_outputs->begin(), owner_outputs->end());
  }
  return mx::array::make_arrays(std::move(shapes), dtypes, primitive, primitive_inputs);
}

std::shared_ptr<PreparedMetalCall> prepare_metal_call(
    uint64_t func_handle,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<std::string>& output_dtypes,
    const std::vector<int64_t>& result_indices,
    int64_t num_params,
    const std::vector<std::string>& expected_param_dtypes,
    const std::vector<std::vector<int64_t>>& param_shapes,
    const std::vector<int64_t>& direct_launch_args,
    const std::vector<int64_t>& direct_param_indices,
    uint64_t direct_module_handle,
    const std::string& direct_kernel_name,
    const std::vector<int64_t>& zero_init_output_positions) {
  if (output_shapes.size() != output_dtypes.size()) {
    throw std::runtime_error("TVM-FFI output_shapes/output_dtypes length mismatch");
  }
  if (output_shapes.size() != result_indices.size()) {
    throw std::runtime_error("TVM-FFI output metadata/result_idx length mismatch");
  }
  if (func_handle == 0) {
    throw std::runtime_error("TVM-FFI function handle is null");
  }
  if (num_params <= 0) {
    throw std::runtime_error("TVM-FFI bridge requires a positive parameter count");
  }
  if (!expected_param_dtypes.empty() &&
      static_cast<int64_t>(expected_param_dtypes.size()) != num_params) {
    throw std::runtime_error("TVM-FFI expected param dtype metadata length mismatch");
  }
  if (!param_shapes.empty() &&
      static_cast<int64_t>(param_shapes.size()) != num_params) {
    throw std::runtime_error("TVM-FFI parameter shape metadata length mismatch");
  }
  if (!direct_param_indices.empty()) {
    if (static_cast<int64_t>(direct_param_indices.size()) != num_params) {
      throw std::runtime_error("TVM-FFI direct Metal parameter permutation length mismatch");
    }
    std::vector<uint8_t> seen(static_cast<size_t>(num_params), 0);
    for (int64_t param_idx : direct_param_indices) {
      if (param_idx < 0 || param_idx >= num_params) {
        throw std::runtime_error("TVM-FFI direct Metal parameter index out of range");
      }
      auto& mapped = seen[static_cast<size_t>(param_idx)];
      if (mapped) {
        throw std::runtime_error("TVM-FFI direct Metal parameter permutation has duplicates");
      }
      mapped = 1;
    }
    for (uint8_t mapped : seen) {
      if (!mapped) {
        throw std::runtime_error("TVM-FFI direct Metal parameter permutation is incomplete");
      }
    }
  }
  auto prepared = std::make_shared<PreparedMetalCall>();
  prepared->func_handle = func_handle;
  prepared->func_handle_ptr = reinterpret_cast<TVMFFIObjectHandle>(func_handle);
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectIncRef(prepared->func_handle_ptr));
  prepared->num_params = num_params;
  prepared->result_indices = result_indices;
  prepared->output_shapes = output_shapes;
  prepared->output_dtypes = output_dtypes;
  prepared->expected_param_dtypes = expected_param_dtypes;
  prepared->param_shapes = param_shapes;
  prepared->direct_launch_args = direct_launch_args;
  prepared->direct_param_indices = direct_param_indices;
  prepared->direct_module_handle = direct_module_handle;
  prepared->direct_kernel_name = direct_kernel_name;
  prepared->zero_init_output_positions = zero_init_output_positions;
  std::sort(prepared->result_indices.begin(), prepared->result_indices.end());
  prepared->result_indices.erase(
      std::unique(prepared->result_indices.begin(), prepared->result_indices.end()),
      prepared->result_indices.end());
  std::sort(prepared->zero_init_output_positions.begin(),
            prepared->zero_init_output_positions.end());
  prepared->zero_init_output_positions.erase(
      std::unique(prepared->zero_init_output_positions.begin(),
                  prepared->zero_init_output_positions.end()),
      prepared->zero_init_output_positions.end());
  for (int64_t idx : prepared->result_indices) {
    if (idx < 0 || idx >= num_params) {
      throw std::runtime_error("TVM-FFI result index is outside the parameter list");
    }
  }
  for (int64_t idx : prepared->zero_init_output_positions) {
    if (idx < 0 || idx >= static_cast<int64_t>(prepared->result_indices.size())) {
      throw std::runtime_error("TVM-FFI zero-init output position is outside the output list");
    }
  }
  if (reinterpret_cast<TVMFFIObjectHandle>(direct_module_handle) != nullptr &&
      !prepared->direct_kernel_name.empty() && !prepared->direct_launch_args.empty()) {
    prepared->direct_launch_handle = get_cached_direct_launch_handle(
        reinterpret_cast<TVMFFIObjectHandle>(direct_module_handle),
        prepared->direct_kernel_name);
  }
  prepared->output_mlx_shapes.reserve(output_shapes.size());
  prepared->output_mlx_dtypes.reserve(output_dtypes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    prepared->output_mlx_shapes.push_back(shape_from_i64(output_shapes[i]));
    prepared->output_mlx_dtypes.push_back(dtype_from_name(output_dtypes[i]));
  }
  return prepared;
}

std::vector<mx::array> tvm_ffi_metal_call_prepared(
    const std::shared_ptr<PreparedMetalCall>& prepared,
    const std::vector<mx::array>& inputs,
    std::shared_ptr<MetalLaunchSyncState> launch_sync_state,
    std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges,
    const std::vector<mx::array>* owner_outputs = nullptr) {
  if (prepared == nullptr) {
    throw std::runtime_error("prepared TVM-FFI Metal call is null");
  }
  auto primitive = std::make_shared<TVMFFIMetalCall>(
      mx::default_stream(mx::Device::gpu),
      prepared,
      owner_outputs != nullptr,
      std::move(launch_sync_state),
      std::move(wait_edges));
  std::vector<mx::array> primitive_inputs;
  primitive_inputs.reserve(
      inputs.size() + (owner_outputs == nullptr ? 0 : owner_outputs->size()));
  primitive_inputs.insert(primitive_inputs.end(), inputs.begin(), inputs.end());
  if (owner_outputs != nullptr) {
    primitive_inputs.insert(
        primitive_inputs.end(), owner_outputs->begin(), owner_outputs->end());
    bool owners_match_prepared_outputs =
        owner_outputs->size() == prepared->output_mlx_shapes.size() &&
        owner_outputs->size() == prepared->output_mlx_dtypes.size();
    if (owners_match_prepared_outputs) {
      for (size_t i = 0; i < owner_outputs->size(); ++i) {
        const auto& owner = owner_outputs->at(i);
        if (owner.shape() != prepared->output_mlx_shapes[i] ||
            owner.dtype() != prepared->output_mlx_dtypes[i]) {
          owners_match_prepared_outputs = false;
          break;
        }
      }
    }
    if (owners_match_prepared_outputs) {
      return mx::array::make_arrays(
          prepared->output_mlx_shapes,
          prepared->output_mlx_dtypes,
          primitive,
          primitive_inputs);
    }
    std::vector<mx::Shape> owner_shapes;
    owner_shapes.reserve(owner_outputs->size());
    std::vector<mx::Dtype> owner_dtypes;
    owner_dtypes.reserve(owner_outputs->size());
    for (const auto& owner : *owner_outputs) {
      owner_shapes.push_back(owner.shape());
      owner_dtypes.push_back(owner.dtype());
    }
    return mx::array::make_arrays(
        std::move(owner_shapes),
        std::move(owner_dtypes),
        primitive,
        primitive_inputs);
  }
  return mx::array::make_arrays(
      prepared->output_mlx_shapes,
      prepared->output_mlx_dtypes,
      primitive,
      primitive_inputs);
}

nb::object wrap_mlx_array(mx::array&& array) {
  auto payload = std::make_unique<mx::array>(std::move(array));
  PyObject* py_array = mlx_core_wrap_mx_array_move(payload.get());
  if (py_array == nullptr) {
    if (PyErr_Occurred() != nullptr) {
      throw nb::python_error();
    }
    throw std::runtime_error("MLX array wrapper returned null without a Python exception");
  }
  payload.release();
  return nb::steal(py_array);
}

nb::list wrap_mlx_arrays(std::vector<mx::array>&& arrays) {
  nb::list result;
  for (auto& array : arrays) {
    result.append(wrap_mlx_array(std::move(array)));
  }
  return result;
}

nb::object tvm_ffi_metal_call_prepared_borrowed_no_wait(
    const std::shared_ptr<PreparedMetalCall>& prepared,
    const std::vector<mx::array>& inputs,
    const std::vector<mx::array>* owner_outputs = nullptr) {
  for (const auto& input : inputs) {
    if (!can_borrow_compact_input_array(input)) {
      return nb::none();
    }
  }
  auto launch_sync_state = make_launch_sync_state();
  auto outputs = wrap_mlx_arrays(tvm_ffi_metal_call_prepared(
      prepared,
      inputs,
      launch_sync_state,
      {},
      owner_outputs));
  return nb::make_tuple(std::move(outputs), launch_sync_state);
}

mx::array make_owner_output_buffer(
    const std::vector<int64_t>& shape,
    const std::string& dtype_name) {
  return mx::empty(
      shape_from_i64(shape),
      dtype_from_name(dtype_name),
      mx::default_stream(mx::Device::gpu));
}

std::vector<mx::array> make_owner_output_buffers(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<std::string>& dtype_names) {
  if (shapes.size() != dtype_names.size()) {
    throw std::runtime_error("owner output shapes/dtypes length mismatch");
  }
  std::vector<mx::array> outputs;
  outputs.reserve(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    outputs.push_back(make_owner_output_buffer(shapes[i], dtype_names[i]));
  }
  return outputs;
}

int fill_c_api_status(TileLangMLXTVMFFIStatus* out, size_t out_size) {
  if (out == nullptr) {
    return kTileLangMLXTVMFFICApiNullOut;
  }
  if (out_size < sizeof(TileLangMLXTVMFFIStatus)) {
    return kTileLangMLXTVMFFICApiStructTooSmall;
  }
  *out = TileLangMLXTVMFFIStatus{};
  out->version = TILELANG_MLX_TVM_FFI_C_API_VERSION;
  out->struct_size = sizeof(TileLangMLXTVMFFIStatus);
  out->code = kTileLangMLXTVMFFICApiOk;
  out->state = "available";
  out->reason = "TileLang MLX TVM-FFI C API is linked";
  out->abi_hash = TILELANG_MLX_TVM_FFI_C_API_ABI_HASH;
  out->header_sha256 = TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256;
  out->mlx_version = TILELANG_MLX_TVM_FFI_BUILD_MLX_VERSION;
  out->mlx_lib_sha256 = TILELANG_MLX_TVM_FFI_BUILD_MLX_LIB_SHA256;
  out->mlx_python_bridge_sha256 = TILELANG_MLX_TVM_FFI_BUILD_MLX_PY_BRIDGE_SHA256;
  return kTileLangMLXTVMFFICApiOk;
}

nb::dict c_api_status_dict() {
  TileLangMLXTVMFFIStatus status{};
  int rc = fill_c_api_status(&status, sizeof(status));
  nb::dict result;
  result["code"] = rc;
  result["state"] = status.state == nullptr ? "unavailable" : status.state;
  result["reason"] = status.reason == nullptr ? "" : status.reason;
  result["version"] = status.version;
  result["struct_size"] = status.struct_size;
  result["abi_hash"] = status.abi_hash == nullptr ? "" : status.abi_hash;
  result["header_sha256"] = status.header_sha256 == nullptr ? "" : status.header_sha256;
  result["mlx_version"] = status.mlx_version == nullptr ? "" : status.mlx_version;
  result["mlx_lib_sha256"] = status.mlx_lib_sha256 == nullptr ? "" : status.mlx_lib_sha256;
  result["mlx_python_bridge_sha256"] =
      status.mlx_python_bridge_sha256 == nullptr ? "" : status.mlx_python_bridge_sha256;
  return result;
}

PyObject* release_py_object(nb::object&& obj) {
  return obj.release().ptr();
}

PyObject* c_api_metal_call(
    uint64_t func_handle,
    PyObject* inputs,
    PyObject* output_shapes,
    PyObject* output_dtypes,
    PyObject* result_indices,
    int64_t num_params,
    PyObject* zero_init_output_positions,
    PyObject* launch_sync_state,
    PyObject* wait_edges) {
  nb::gil_scoped_acquire gil;
  if (inputs == nullptr || output_shapes == nullptr || output_dtypes == nullptr ||
      result_indices == nullptr) {
    PyErr_SetString(PyExc_TypeError, "TileLang MLX TVM-FFI C API received a null PyObject");
    return nullptr;
  }
  PyObject* zero_init = zero_init_output_positions == nullptr ? Py_None : zero_init_output_positions;
  PyObject* launch_state = launch_sync_state == nullptr ? Py_None : launch_sync_state;
  PyObject* waits = wait_edges == nullptr ? Py_None : wait_edges;
  try {
    return release_py_object(wrap_mlx_arrays(tvm_ffi_metal_call(
        func_handle,
        parse_array_sequence(nb::handle(inputs)),
        parse_shape_sequence(nb::handle(output_shapes)),
        parse_string_sequence(nb::handle(output_dtypes)),
        parse_i64_sequence(nb::handle(result_indices)),
        num_params,
        std::vector<std::string>{},
        std::vector<std::vector<int64_t>>{},
        std::vector<int64_t>{},
        std::vector<int64_t>{},
        0,
        std::string{},
        parse_i64_sequence(nb::handle(zero_init)),
        parse_launch_sync_state(nb::handle(launch_state)),
        parse_sync_edge_sequence(nb::handle(waits)))));
  } catch (nb::python_error& exc) {
    exc.restore();
    return nullptr;
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
  }
}

PyObject* c_api_owner_output_buffer(PyObject* shape, const char* dtype_name) {
  nb::gil_scoped_acquire gil;
  if (shape == nullptr || dtype_name == nullptr) {
    PyErr_SetString(PyExc_TypeError, "owner_output_buffer received a null argument");
    return nullptr;
  }
  try {
    return release_py_object(wrap_mlx_array(
        make_owner_output_buffer(parse_i64_sequence(nb::handle(shape)), dtype_name)));
  } catch (nb::python_error& exc) {
    exc.restore();
    return nullptr;
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
  }
}

PyObject* c_api_owner_output_buffers(PyObject* shapes, PyObject* dtype_names) {
  nb::gil_scoped_acquire gil;
  if (shapes == nullptr || dtype_names == nullptr) {
    PyErr_SetString(PyExc_TypeError, "owner_output_buffers received a null argument");
    return nullptr;
  }
  try {
    return release_py_object(wrap_mlx_arrays(make_owner_output_buffers(
        parse_shape_sequence(nb::handle(shapes)),
        parse_string_sequence(nb::handle(dtype_names)))));
  } catch (nb::python_error& exc) {
    exc.restore();
    return nullptr;
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
  }
}

PyObject* c_api_make_launch_sync_state() {
  nb::gil_scoped_acquire gil;
  try {
    return release_py_object(nb::cast(make_launch_sync_state()));
  } catch (nb::python_error& exc) {
    exc.restore();
    return nullptr;
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
  }
}

PyObject* c_api_make_sync_edge() {
  nb::gil_scoped_acquire gil;
  try {
    return release_py_object(nb::cast(make_sync_edge()));
  } catch (nb::python_error& exc) {
    exc.restore();
    return nullptr;
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
  }
}

PyObject* c_api_debug_state() {
  nb::gil_scoped_acquire gil;
  try {
    return release_py_object(debug_state());
  } catch (nb::python_error& exc) {
    exc.restore();
    return nullptr;
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    return nullptr;
  }
}

}  // namespace tilelang::mlx_tvm_ffi

extern "C" TILELANG_MLX_TVM_FFI_EXPORT int tilelang_mlx_tvm_ffi_status(
    TileLangMLXTVMFFIStatus* out,
    size_t out_size) {
  return tilelang::mlx_tvm_ffi::fill_c_api_status(out, out_size);
}

extern "C" TILELANG_MLX_TVM_FFI_EXPORT const char* tilelang_mlx_tvm_ffi_c_api_abi_hash(void) {
  return TILELANG_MLX_TVM_FFI_C_API_ABI_HASH;
}

extern "C" TILELANG_MLX_TVM_FFI_EXPORT const char*
tilelang_mlx_tvm_ffi_c_api_header_sha256(void) {
  return TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256;
}

extern "C" TILELANG_MLX_TVM_FFI_EXPORT const char* tilelang_mlx_tvm_ffi_mlx_version(void) {
  return TILELANG_MLX_TVM_FFI_BUILD_MLX_VERSION;
}

extern "C" TILELANG_MLX_TVM_FFI_EXPORT const char* tilelang_mlx_tvm_ffi_mlx_lib_sha256(void) {
  return TILELANG_MLX_TVM_FFI_BUILD_MLX_LIB_SHA256;
}

extern "C" TILELANG_MLX_TVM_FFI_EXPORT const char*
tilelang_mlx_tvm_ffi_mlx_python_bridge_sha256(void) {
  return TILELANG_MLX_TVM_FFI_BUILD_MLX_PY_BRIDGE_SHA256;
}

extern "C" TILELANG_MLX_TVM_FFI_EXPORT int tilelang_mlx_tvm_ffi_get_c_api(
    uint32_t requested_version,
    const char* requested_abi_hash,
    TileLangMLXTVMFFICAPI* out,
    size_t out_size) {
  if (out == nullptr) {
    return kTileLangMLXTVMFFICApiNullOut;
  }
  if (out_size < sizeof(TileLangMLXTVMFFICAPI)) {
    return kTileLangMLXTVMFFICApiStructTooSmall;
  }
  if (requested_version != TILELANG_MLX_TVM_FFI_C_API_VERSION) {
    return kTileLangMLXTVMFFICApiVersionMismatch;
  }
  if (requested_abi_hash != nullptr && requested_abi_hash[0] != '\0' &&
      std::strcmp(requested_abi_hash, TILELANG_MLX_TVM_FFI_C_API_ABI_HASH) != 0) {
    return kTileLangMLXTVMFFICApiHashMismatch;
  }
  *out = TileLangMLXTVMFFICAPI{};
  out->version = TILELANG_MLX_TVM_FFI_C_API_VERSION;
  out->struct_size = sizeof(TileLangMLXTVMFFICAPI);
  out->abi_hash = TILELANG_MLX_TVM_FFI_C_API_ABI_HASH;
  out->header_sha256 = TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256;
  out->mlx_version = TILELANG_MLX_TVM_FFI_BUILD_MLX_VERSION;
  out->mlx_lib_sha256 = TILELANG_MLX_TVM_FFI_BUILD_MLX_LIB_SHA256;
  out->mlx_python_bridge_sha256 = TILELANG_MLX_TVM_FFI_BUILD_MLX_PY_BRIDGE_SHA256;
  out->status = tilelang_mlx_tvm_ffi_status;
  out->metal_call = tilelang::mlx_tvm_ffi::c_api_metal_call;
  out->owner_output_buffer = tilelang::mlx_tvm_ffi::c_api_owner_output_buffer;
  out->owner_output_buffers = tilelang::mlx_tvm_ffi::c_api_owner_output_buffers;
  out->make_launch_sync_state = tilelang::mlx_tvm_ffi::c_api_make_launch_sync_state;
  out->make_sync_edge = tilelang::mlx_tvm_ffi::c_api_make_sync_edge;
  out->debug_state = tilelang::mlx_tvm_ffi::c_api_debug_state;
  out->reset_debug_state = tilelang::mlx_tvm_ffi::reset_debug_counters;
  return kTileLangMLXTVMFFICApiOk;
}

#if TILELANG_MLX_TVM_FFI_WITH_PY_MODULE
NB_MODULE(_tilelang_mlx_tvm_ffi, m) {
  m.doc() = "Native graph-safe MLX primitive for TileLang TVM-FFI Metal kernels";
  nb::class_<tilelang::mlx_tvm_ffi::MetalSyncEdge>(m, "MetalSyncEdge");
  nb::class_<tilelang::mlx_tvm_ffi::MetalLaunchSyncState>(m, "MetalLaunchSyncState")
      .def("add_signal_edge", &tilelang::mlx_tvm_ffi::MetalLaunchSyncState::add_signal_edge)
      .def("signal_edge_count", &tilelang::mlx_tvm_ffi::MetalLaunchSyncState::signal_edge_count);
  nb::class_<tilelang::mlx_tvm_ffi::PreparedMetalCall>(m, "PreparedMetalCall");
  m.def(
      "prepare_metal_call",
      [](uint64_t func_handle,
         nb::handle output_shapes,
         nb::handle output_dtypes,
         nb::handle result_indices,
         int64_t num_params,
         nb::handle param_dtypes,
         nb::handle param_shapes,
         nb::handle direct_launch_args,
         nb::handle direct_param_indices,
         uint64_t direct_module_handle,
         const std::string& direct_kernel_name,
         nb::handle zero_init_output_positions) {
        std::vector<std::vector<int64_t>> parsed_output_shapes;
        std::vector<std::string> parsed_output_dtypes;
        std::vector<int64_t> parsed_result_indices;
        std::vector<std::string> parsed_param_dtypes;
        std::vector<std::vector<int64_t>> parsed_param_shapes;
        std::vector<int64_t> parsed_direct_launch_args;
        std::vector<int64_t> parsed_direct_param_indices;
        std::vector<int64_t> parsed_zero_init_output_positions;
        try {
          parsed_output_shapes = tilelang::mlx_tvm_ffi::parse_shape_sequence(output_shapes);
          parsed_output_dtypes = tilelang::mlx_tvm_ffi::parse_string_sequence(output_dtypes);
          parsed_result_indices = tilelang::mlx_tvm_ffi::parse_i64_sequence(result_indices);
          if (!param_dtypes.is_none()) {
            parsed_param_dtypes = tilelang::mlx_tvm_ffi::parse_string_sequence(param_dtypes);
          }
          if (!param_shapes.is_none()) {
            parsed_param_shapes = tilelang::mlx_tvm_ffi::parse_shape_sequence(param_shapes);
          }
          if (!direct_launch_args.is_none()) {
            parsed_direct_launch_args =
                tilelang::mlx_tvm_ffi::parse_i64_sequence(direct_launch_args);
          }
          if (!direct_param_indices.is_none()) {
            parsed_direct_param_indices =
                tilelang::mlx_tvm_ffi::parse_i64_sequence(direct_param_indices);
          }
          parsed_zero_init_output_positions =
              tilelang::mlx_tvm_ffi::parse_i64_sequence(zero_init_output_positions);
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to prepare native Metal call: ") + exc.what());
        }
        return tilelang::mlx_tvm_ffi::prepare_metal_call(
            func_handle,
            parsed_output_shapes,
            parsed_output_dtypes,
            parsed_result_indices,
            num_params,
            parsed_param_dtypes,
            parsed_param_shapes,
            parsed_direct_launch_args,
            parsed_direct_param_indices,
            direct_module_handle,
            direct_kernel_name,
            parsed_zero_init_output_positions);
      },
      "func_handle"_a,
      "output_shapes"_a,
      "output_dtypes"_a,
      "result_indices"_a,
      "num_params"_a,
      "param_dtypes"_a = nb::none(),
      "param_shapes"_a = nb::none(),
      "direct_launch_args"_a = nb::none(),
      "direct_param_indices"_a = nb::none(),
      "direct_module_handle"_a = 0,
      "direct_kernel_name"_a = "",
      "zero_init_output_positions"_a = nb::make_tuple());
  m.def(
      "metal_call",
      [](uint64_t func_handle,
         nb::handle inputs,
         nb::handle output_shapes,
         nb::handle output_dtypes,
         nb::handle result_indices,
         int64_t num_params,
         nb::handle zero_init_output_positions,
         nb::handle launch_sync_state,
         nb::handle wait_edges,
         nb::handle param_dtypes,
         nb::handle param_shapes,
         nb::handle direct_launch_args,
         nb::handle direct_param_indices,
         uint64_t direct_module_handle,
         const std::string& direct_kernel_name) {
        std::vector<mx::array> parsed_inputs;
        std::vector<std::vector<int64_t>> parsed_output_shapes;
        std::vector<std::string> parsed_output_dtypes;
        std::vector<int64_t> parsed_result_indices;
        std::vector<std::string> parsed_param_dtypes;
        std::vector<std::vector<int64_t>> parsed_param_shapes;
        std::vector<int64_t> parsed_direct_launch_args;
        std::vector<int64_t> parsed_direct_param_indices;
        std::vector<int64_t> parsed_zero_init_output_positions;
        std::shared_ptr<tilelang::mlx_tvm_ffi::MetalLaunchSyncState> parsed_launch_sync_state;
        std::vector<std::shared_ptr<tilelang::mlx_tvm_ffi::MetalSyncEdge>> parsed_wait_edges;
        try {
          parsed_inputs = tilelang::mlx_tvm_ffi::parse_array_sequence(inputs);
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse MLX input arrays: ") + exc.what());
        }
        try {
          parsed_output_shapes = tilelang::mlx_tvm_ffi::parse_shape_sequence(output_shapes);
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse output shapes: ") + exc.what());
        }
        try {
          parsed_output_dtypes = tilelang::mlx_tvm_ffi::parse_string_sequence(output_dtypes);
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse output dtypes: ") + exc.what());
        }
        try {
          parsed_result_indices = tilelang::mlx_tvm_ffi::parse_i64_sequence(result_indices);
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse result indices: ") + exc.what());
        }
        try {
          if (!param_dtypes.is_none()) {
            parsed_param_dtypes = tilelang::mlx_tvm_ffi::parse_string_sequence(param_dtypes);
          }
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse param dtypes: ") + exc.what());
        }
        try {
          if (!param_shapes.is_none()) {
            parsed_param_shapes = tilelang::mlx_tvm_ffi::parse_shape_sequence(param_shapes);
          }
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse param shapes: ") + exc.what());
        }
        try {
          if (!direct_launch_args.is_none()) {
            parsed_direct_launch_args =
                tilelang::mlx_tvm_ffi::parse_i64_sequence(direct_launch_args);
          }
          if (!direct_param_indices.is_none()) {
            parsed_direct_param_indices =
                tilelang::mlx_tvm_ffi::parse_i64_sequence(direct_param_indices);
          }
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to parse direct device launch args: ") + exc.what());
        }
        try {
          parsed_zero_init_output_positions =
              tilelang::mlx_tvm_ffi::parse_i64_sequence(zero_init_output_positions);
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to parse zero-init output positions: ") + exc.what());
        }
        try {
          parsed_launch_sync_state =
              tilelang::mlx_tvm_ffi::parse_launch_sync_state(launch_sync_state);
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse launch sync state: ") + exc.what());
        }
        try {
          parsed_wait_edges = tilelang::mlx_tvm_ffi::parse_sync_edge_sequence(wait_edges);
        } catch (const std::exception& exc) {
          throw std::runtime_error(std::string("failed to parse wait sync edges: ") + exc.what());
        }
        return tilelang::mlx_tvm_ffi::wrap_mlx_arrays(
            tilelang::mlx_tvm_ffi::tvm_ffi_metal_call(
                func_handle,
                parsed_inputs,
                parsed_output_shapes,
                parsed_output_dtypes,
                parsed_result_indices,
                num_params,
                parsed_param_dtypes,
                parsed_param_shapes,
                parsed_direct_launch_args,
                parsed_direct_param_indices,
                direct_module_handle,
                direct_kernel_name,
                parsed_zero_init_output_positions,
                parsed_launch_sync_state,
                parsed_wait_edges));
      },
      "func_handle"_a,
      "inputs"_a,
      "output_shapes"_a,
      "output_dtypes"_a,
      "result_indices"_a,
      "num_params"_a,
      "zero_init_output_positions"_a = nb::make_tuple(),
      "launch_sync_state"_a = nb::none(),
      "wait_edges"_a = nb::none(),
      "param_dtypes"_a = nb::none(),
      "param_shapes"_a = nb::none(),
      "direct_launch_args"_a = nb::none(),
      "direct_param_indices"_a = nb::none(),
      "direct_module_handle"_a = 0,
      "direct_kernel_name"_a = "");
  m.def(
      "metal_call_owner_outputs",
      [](uint64_t func_handle,
         nb::handle inputs,
         nb::handle owner_outputs,
         nb::handle output_shapes,
         nb::handle output_dtypes,
         nb::handle result_indices,
         int64_t num_params,
         nb::handle zero_init_output_positions,
         nb::handle launch_sync_state,
         nb::handle wait_edges,
         nb::handle param_dtypes,
         nb::handle param_shapes,
         nb::handle direct_launch_args,
         nb::handle direct_param_indices,
         uint64_t direct_module_handle,
         const std::string& direct_kernel_name) {
        std::vector<mx::array> parsed_inputs;
        std::vector<mx::array> parsed_owner_outputs;
        std::vector<std::vector<int64_t>> parsed_output_shapes;
        std::vector<std::string> parsed_output_dtypes;
        std::vector<int64_t> parsed_result_indices;
        std::vector<std::string> parsed_param_dtypes;
        std::vector<std::vector<int64_t>> parsed_param_shapes;
        std::vector<int64_t> parsed_direct_launch_args;
        std::vector<int64_t> parsed_direct_param_indices;
        std::vector<int64_t> parsed_zero_init_output_positions;
        std::shared_ptr<tilelang::mlx_tvm_ffi::MetalLaunchSyncState> parsed_launch_sync_state;
        std::vector<std::shared_ptr<tilelang::mlx_tvm_ffi::MetalSyncEdge>> parsed_wait_edges;
        try {
          parsed_inputs = tilelang::mlx_tvm_ffi::parse_array_sequence(inputs);
          parsed_owner_outputs = tilelang::mlx_tvm_ffi::parse_array_sequence(owner_outputs);
          parsed_output_shapes = tilelang::mlx_tvm_ffi::parse_shape_sequence(output_shapes);
          parsed_output_dtypes = tilelang::mlx_tvm_ffi::parse_string_sequence(output_dtypes);
          parsed_result_indices = tilelang::mlx_tvm_ffi::parse_i64_sequence(result_indices);
          if (!param_dtypes.is_none()) {
            parsed_param_dtypes = tilelang::mlx_tvm_ffi::parse_string_sequence(param_dtypes);
          }
          if (!param_shapes.is_none()) {
            parsed_param_shapes = tilelang::mlx_tvm_ffi::parse_shape_sequence(param_shapes);
          }
          if (!direct_launch_args.is_none()) {
            parsed_direct_launch_args =
                tilelang::mlx_tvm_ffi::parse_i64_sequence(direct_launch_args);
          }
          if (!direct_param_indices.is_none()) {
            parsed_direct_param_indices =
                tilelang::mlx_tvm_ffi::parse_i64_sequence(direct_param_indices);
          }
          parsed_zero_init_output_positions =
              tilelang::mlx_tvm_ffi::parse_i64_sequence(zero_init_output_positions);
          parsed_launch_sync_state =
              tilelang::mlx_tvm_ffi::parse_launch_sync_state(launch_sync_state);
          parsed_wait_edges = tilelang::mlx_tvm_ffi::parse_sync_edge_sequence(wait_edges);
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to parse native owner-output call: ") + exc.what());
        }
        return tilelang::mlx_tvm_ffi::wrap_mlx_arrays(
            tilelang::mlx_tvm_ffi::tvm_ffi_metal_call(
                func_handle,
                parsed_inputs,
                parsed_output_shapes,
                parsed_output_dtypes,
                parsed_result_indices,
                num_params,
                parsed_param_dtypes,
                parsed_param_shapes,
                parsed_direct_launch_args,
                parsed_direct_param_indices,
                direct_module_handle,
                direct_kernel_name,
                parsed_zero_init_output_positions,
                parsed_launch_sync_state,
                parsed_wait_edges,
                &parsed_owner_outputs));
      },
      "func_handle"_a,
      "inputs"_a,
      "owner_outputs"_a,
      "output_shapes"_a,
      "output_dtypes"_a,
      "result_indices"_a,
      "num_params"_a,
      "zero_init_output_positions"_a = nb::make_tuple(),
      "launch_sync_state"_a = nb::none(),
      "wait_edges"_a = nb::none(),
      "param_dtypes"_a = nb::none(),
      "param_shapes"_a = nb::none(),
      "direct_launch_args"_a = nb::none(),
      "direct_param_indices"_a = nb::none(),
      "direct_module_handle"_a = 0,
      "direct_kernel_name"_a = "");
  m.def(
      "prepared_metal_call",
      [](const std::shared_ptr<tilelang::mlx_tvm_ffi::PreparedMetalCall>& prepared,
         nb::handle inputs,
         nb::handle launch_sync_state,
         nb::handle wait_edges) {
        std::vector<mx::array> parsed_inputs;
        std::shared_ptr<tilelang::mlx_tvm_ffi::MetalLaunchSyncState> parsed_launch_sync_state;
        std::vector<std::shared_ptr<tilelang::mlx_tvm_ffi::MetalSyncEdge>> parsed_wait_edges;
        try {
          parsed_inputs = tilelang::mlx_tvm_ffi::parse_array_sequence(inputs);
          parsed_launch_sync_state =
              tilelang::mlx_tvm_ffi::parse_launch_sync_state(launch_sync_state);
          parsed_wait_edges = tilelang::mlx_tvm_ffi::parse_sync_edge_sequence(wait_edges);
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to parse prepared native Metal call: ") + exc.what());
        }
        return tilelang::mlx_tvm_ffi::wrap_mlx_arrays(
            tilelang::mlx_tvm_ffi::tvm_ffi_metal_call_prepared(
                prepared,
                parsed_inputs,
                parsed_launch_sync_state,
                parsed_wait_edges));
      },
      "prepared"_a,
      "inputs"_a,
      "launch_sync_state"_a = nb::none(),
      "wait_edges"_a = nb::none());
  m.def(
      "prepared_metal_call_owner_outputs",
      [](const std::shared_ptr<tilelang::mlx_tvm_ffi::PreparedMetalCall>& prepared,
         nb::handle inputs,
         nb::handle owner_outputs,
         nb::handle launch_sync_state,
         nb::handle wait_edges) {
        std::vector<mx::array> parsed_inputs;
        std::vector<mx::array> parsed_owner_outputs;
        std::shared_ptr<tilelang::mlx_tvm_ffi::MetalLaunchSyncState> parsed_launch_sync_state;
        std::vector<std::shared_ptr<tilelang::mlx_tvm_ffi::MetalSyncEdge>> parsed_wait_edges;
        try {
          parsed_inputs = tilelang::mlx_tvm_ffi::parse_array_sequence(inputs);
          parsed_owner_outputs = tilelang::mlx_tvm_ffi::parse_array_sequence(owner_outputs);
          parsed_launch_sync_state =
              tilelang::mlx_tvm_ffi::parse_launch_sync_state(launch_sync_state);
          parsed_wait_edges = tilelang::mlx_tvm_ffi::parse_sync_edge_sequence(wait_edges);
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to parse prepared native owner-output call: ") + exc.what());
        }
        return tilelang::mlx_tvm_ffi::wrap_mlx_arrays(
            tilelang::mlx_tvm_ffi::tvm_ffi_metal_call_prepared(
                prepared,
                parsed_inputs,
                parsed_launch_sync_state,
                parsed_wait_edges,
                &parsed_owner_outputs));
      },
      "prepared"_a,
      "inputs"_a,
      "owner_outputs"_a,
      "launch_sync_state"_a = nb::none(),
      "wait_edges"_a = nb::none());
  m.def(
      "prepared_metal_call_borrowed_no_wait",
      [](const std::shared_ptr<tilelang::mlx_tvm_ffi::PreparedMetalCall>& prepared,
         nb::handle inputs,
         nb::handle owner_outputs) {
        std::vector<mx::array> parsed_inputs;
        std::vector<mx::array> parsed_owner_outputs;
        try {
          parsed_inputs = tilelang::mlx_tvm_ffi::parse_array_sequence(inputs);
          if (!owner_outputs.is_none()) {
            parsed_owner_outputs =
                tilelang::mlx_tvm_ffi::parse_array_sequence(owner_outputs);
          }
        } catch (const std::exception& exc) {
          throw std::runtime_error(
              std::string("failed to parse prepared borrowed native call: ") + exc.what());
        }
        return tilelang::mlx_tvm_ffi::tvm_ffi_metal_call_prepared_borrowed_no_wait(
            prepared,
            parsed_inputs,
            owner_outputs.is_none() ? nullptr : &parsed_owner_outputs);
      },
      "prepared"_a,
      "inputs"_a,
      "owner_outputs"_a = nb::none());
  m.def(
      "is_compact",
      [](nb::handle array) {
        return tilelang::mlx_tvm_ffi::is_compact_array(
            tilelang::mlx_tvm_ffi::parse_mlx_array(array));
      },
      "array"_a);
  m.def(
      "can_borrow_compact_input",
      [](nb::handle array) {
        return tilelang::mlx_tvm_ffi::can_borrow_compact_input_array(
            tilelang::mlx_tvm_ffi::parse_mlx_array(array));
      },
      "array"_a);
  m.def(
      "compact_input",
      [](nb::handle array) {
        return tilelang::mlx_tvm_ffi::wrap_mlx_array(
            tilelang::mlx_tvm_ffi::compact_input_array(
                tilelang::mlx_tvm_ffi::parse_mlx_array(array)));
      },
      "array"_a);
  m.def("make_launch_sync_state", &tilelang::mlx_tvm_ffi::make_launch_sync_state);
  m.def("make_sync_edge", &tilelang::mlx_tvm_ffi::make_sync_edge);
  m.def("debug_state", &tilelang::mlx_tvm_ffi::debug_state);
  m.def("reset_debug_state", &tilelang::mlx_tvm_ffi::reset_debug_counters);
  m.def("c_api_status", &tilelang::mlx_tvm_ffi::c_api_status_dict);
  m.def("c_api_abi_hash", []() { return TILELANG_MLX_TVM_FFI_C_API_ABI_HASH; });
  m.def("c_api_header_sha256", []() { return TILELANG_MLX_TVM_FFI_C_API_HEADER_SHA256; });
  m.def(
      "owner_output_buffer",
      [](nb::handle shape, const std::string& dtype_name) {
        return tilelang::mlx_tvm_ffi::wrap_mlx_array(
            tilelang::mlx_tvm_ffi::make_owner_output_buffer(
                tilelang::mlx_tvm_ffi::parse_i64_sequence(shape),
                dtype_name));
      },
      "shape"_a,
      "dtype"_a);
  m.def(
      "owner_output_buffers",
      [](nb::handle shapes, nb::handle dtypes) {
        return tilelang::mlx_tvm_ffi::wrap_mlx_arrays(
            tilelang::mlx_tvm_ffi::make_owner_output_buffers(
                tilelang::mlx_tvm_ffi::parse_shape_sequence(shapes),
                tilelang::mlx_tvm_ffi::parse_string_sequence(dtypes)));
      },
      "shapes"_a,
      "dtypes"_a);
}
#endif  // TILELANG_MLX_TVM_FFI_WITH_PY_MODULE
