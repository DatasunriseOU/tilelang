#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>

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

namespace tilelang::mlx_tvm_ffi {

constexpr int32_t kDLMetalDeviceType = 8;
constexpr const char* kDebugCompletionEnv = "TILELANG_MLX_TVM_FFI_DEBUG_COMPLETION";

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
  state["debug_completion_enabled"] = env_flag_enabled(kDebugCompletionEnv);
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

struct MetalSyncEdge {
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

struct MetalLaunchSyncState {
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
  throw std::runtime_error("unsupported MLX output dtype for TVM-FFI bridge: " + raw_name);
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

BorrowedTensorView make_tensor_view(
    const mx::array& array,
    mx::Stream stream,
    bool is_output) {
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
  view.shape.reserve(array.ndim());
  for (auto dim : array.shape()) {
    view.shape.push_back(static_cast<int64_t>(dim));
  }
  view.strides.assign(array.strides().begin(), array.strides().end());

  view.tensor.data = buffer;
  view.tensor.device = DLDevice{
      static_cast<DLDeviceType>(kDLMetalDeviceType),
      stream.device.index};
  view.tensor.ndim = static_cast<int>(view.shape.size());
  view.tensor.dtype = mlx_dtype_to_dlpack(array.dtype());
  view.tensor.shape = view.shape.data();
  view.tensor.strides = view.strides.data();
  view.tensor.byte_offset = static_cast<uint64_t>(array.offset());
  return view;
}

struct ExternalCommandBufferScope {
  explicit ExternalCommandBufferScope(void* command_buffer) {
    auto& counters = debug_counters();
    counters.command_buffers_checked.fetch_add(1, std::memory_order_relaxed);
    if (command_buffer == nullptr) {
      counters.null_command_buffers.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("MLX returned a null Metal command buffer for TVM-FFI launch");
    }
    TVMFFIAny arg;
    arg.type_index = kTVMFFIOpaquePtr;
    arg.zero_padding = 0;
    arg.v_ptr = command_buffer;
    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionCall(
        set_external_function(),
        &arg,
        1,
        &result));
  }

  ~ExternalCommandBufferScope() {
    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;
    // Destructors cannot throw.  If clearing fails, the original TVM call
    // already failed or the runtime is unloading.
    (void)TVMFFIFunctionCall(clear_external_function(), nullptr, 0, &result);
  }

  static TVMFFIObjectHandle lookup_global_function(const char* name) {
    TVMFFIObjectHandle handle = nullptr;
    TVMFFIByteArray name_arr{name, std::char_traits<char>::length(name)};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionGetGlobal(&name_arr, &handle));
    if (handle == nullptr) {
      throw std::runtime_error(std::string("missing TVM global function: ") + name);
    }
    return handle;
  }

  static TVMFFIObjectHandle set_external_function() {
    static TVMFFIObjectHandle handle = lookup_global_function("metal.SetExternalCommandBuffer");
    return handle;
  }

  static TVMFFIObjectHandle clear_external_function() {
    static TVMFFIObjectHandle handle = lookup_global_function("metal.ClearExternalCommandBuffer");
    return handle;
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
  // Create a post-launch encoder so MLX records an output fence after the
  // external TVM-FFI commands. Without this, same-graph MLX consumers can race
  // a TileLang producer even though the output array is a normal graph value.
  encoder->barrier();
  encoder->end_encoding();
}

class TVMFFIMetalCall : public mx::Primitive {
 public:
  TVMFFIMetalCall(
      mx::Stream stream,
      uint64_t func_handle,
      int64_t num_params,
      std::vector<int64_t> result_indices,
      std::vector<int64_t> zero_init_output_positions,
      std::shared_ptr<MetalLaunchSyncState> launch_sync_state,
      std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges)
      : mx::Primitive(stream),
        func_handle_(reinterpret_cast<TVMFFIObjectHandle>(func_handle)),
        num_params_(num_params),
        result_indices_(std::move(result_indices)),
        zero_init_output_positions_(std::move(zero_init_output_positions)),
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
    for (int64_t idx : zero_init_output_positions_) {
      if (idx < 0 || idx >= static_cast<int64_t>(result_indices_.size())) {
        throw std::runtime_error("TVM-FFI zero-init output position is outside the output list");
      }
    }
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectIncRef(func_handle_));
  }

  ~TVMFFIMetalCall() override {
    if (func_handle_ != nullptr) {
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
    debug_counters().launches.fetch_add(1, std::memory_order_relaxed);
    if (outputs.size() != result_indices_.size()) {
      throw std::runtime_error("TVM-FFI bridge output count mismatch");
    }
    const int64_t expected_inputs = num_params_ - static_cast<int64_t>(result_indices_.size());
    if (static_cast<int64_t>(inputs.size()) != expected_inputs) {
      throw std::runtime_error("TVM-FFI bridge input count mismatch");
    }

    auto* encoder = get_command_encoder(stream());
    for (auto& out : outputs) {
      out.set_data(mx::allocator::malloc(out.nbytes()));
      if (out.buffer().ptr() == nullptr) {
        throw std::runtime_error("MLX failed to allocate TVM-FFI output buffer");
      }
    }

    std::vector<BorrowedTensorView> views;
    views.reserve(num_params_);
    std::vector<TVMFFIAny> args(static_cast<size_t>(num_params_));
    size_t input_pos = 0;
    size_t output_pos = 0;
    for (int64_t param_idx = 0; param_idx < num_params_; ++param_idx) {
      const bool is_output = std::binary_search(
          result_indices_.begin(),
          result_indices_.end(),
          param_idx);
      const mx::array& array = is_output ? outputs.at(output_pos++) : inputs.at(input_pos++);
      views.push_back(make_tensor_view(array, stream(), is_output));
      TVMFFIAny& arg = args[static_cast<size_t>(param_idx)];
      arg.type_index = kTVMFFIDLTensorPtr;
      arg.zero_padding = 0;
      arg.v_ptr = &views.back().tensor;
    }

    register_external_inputs(encoder, inputs);
    auto* command_buffer = finish_encoding_and_get_command_buffer(stream());
    encode_command_buffer_boundary(command_buffer, stream());
    encode_device_event_waits(command_buffer, stream(), wait_edges_);
    for (int64_t output_pos : zero_init_output_positions_) {
      zero_output_buffer(command_buffer, outputs.at(static_cast<size_t>(output_pos)));
    }
    encode_command_buffer_boundary(command_buffer, stream());

    auto signal_edges = launch_sync_state_->snapshot_signal_edges();
    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;
    {
      ExternalCommandBufferScope external(command_buffer);
      TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionCall(
          func_handle_,
          args.data(),
          static_cast<int32_t>(args.size()),
          &result));
    }
    encode_device_event_signals(command_buffer, stream(), signal_edges);
    encode_command_buffer_boundary(command_buffer, stream());
    publish_external_outputs(stream(), outputs);
    if (result.type_index >= kTVMFFIStaticObjectBegin && result.v_obj != nullptr) {
      TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectDecRef(result.v_obj));
    }
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
  std::vector<int64_t> result_indices_;
  std::vector<int64_t> zero_init_output_positions_;
  std::shared_ptr<MetalLaunchSyncState> launch_sync_state_;
  std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges_;
};

std::vector<mx::array> tvm_ffi_metal_call(
    uint64_t func_handle,
    const std::vector<mx::array>& inputs,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<std::string>& output_dtypes,
    const std::vector<int64_t>& result_indices,
    int64_t num_params,
    const std::vector<int64_t>& zero_init_output_positions,
    std::shared_ptr<MetalLaunchSyncState> launch_sync_state,
    std::vector<std::shared_ptr<MetalSyncEdge>> wait_edges) {
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
      zero_init_output_positions,
      std::move(launch_sync_state),
      std::move(wait_edges));
  return mx::array::make_arrays(std::move(shapes), dtypes, primitive, inputs);
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

}  // namespace tilelang::mlx_tvm_ffi

NB_MODULE(_tilelang_mlx_tvm_ffi, m) {
  m.doc() = "Native graph-safe MLX primitive for TileLang TVM-FFI Metal kernels";
  nb::class_<tilelang::mlx_tvm_ffi::MetalSyncEdge>(m, "MetalSyncEdge");
  nb::class_<tilelang::mlx_tvm_ffi::MetalLaunchSyncState>(m, "MetalLaunchSyncState")
      .def("add_signal_edge", &tilelang::mlx_tvm_ffi::MetalLaunchSyncState::add_signal_edge)
      .def("signal_edge_count", &tilelang::mlx_tvm_ffi::MetalLaunchSyncState::signal_edge_count);
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
         nb::handle wait_edges) {
        std::vector<mx::array> parsed_inputs;
        std::vector<std::vector<int64_t>> parsed_output_shapes;
        std::vector<std::string> parsed_output_dtypes;
        std::vector<int64_t> parsed_result_indices;
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
      "wait_edges"_a = nb::none());
  m.def("make_launch_sync_state", &tilelang::mlx_tvm_ffi::make_launch_sync_state);
  m.def("make_sync_edge", &tilelang::mlx_tvm_ffi::make_sync_edge);
  m.def("debug_state", &tilelang::mlx_tvm_ffi::debug_state);
  m.def("reset_debug_state", &tilelang::mlx_tvm_ffi::reset_debug_counters);
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
