#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
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
#include "mlx/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
namespace mx = mlx::core;

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

std::vector<mx::array> parse_array_sequence(nb::handle values) {
  std::vector<mx::array> arrays;
  for (nb::handle item : nb::iter(values)) {
    arrays.push_back(nb::cast<mx::array>(item));
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

struct BorrowedTensorView {
  DLTensor tensor{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

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

void zero_output_buffer(MTL::CommandBuffer* command_buffer, mx::array& out) {
  if (out.nbytes() == 0) {
    return;
  }
  auto* buffer = static_cast<MTL::Buffer*>(const_cast<void*>(out.buffer().ptr()));
  auto* blit_encoder = command_buffer->blitCommandEncoder();
  blit_encoder->fillBuffer(buffer, NS::Range::Make(out.offset(), out.nbytes()), 0);
  blit_encoder->endEncoding();
}

class TVMFFIMetalCall : public mx::Primitive {
 public:
  TVMFFIMetalCall(
      mx::Stream stream,
      uint64_t func_handle,
      int64_t num_params,
      std::vector<int64_t> result_indices)
      : mx::Primitive(stream),
        func_handle_(reinterpret_cast<TVMFFIObjectHandle>(func_handle)),
        num_params_(num_params),
        result_indices_(std::move(result_indices)) {
    if (func_handle_ == nullptr) {
      throw std::runtime_error("TVM-FFI function handle is null");
    }
    if (num_params_ <= 0) {
      throw std::runtime_error("TVM-FFI bridge requires a positive parameter count");
    }
    std::sort(result_indices_.begin(), result_indices_.end());
    result_indices_.erase(
        std::unique(result_indices_.begin(), result_indices_.end()),
        result_indices_.end());
    for (int64_t idx : result_indices_) {
      if (idx < 0 || idx >= num_params_) {
        throw std::runtime_error("TVM-FFI result index is outside the parameter list");
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

    auto& encoder = mx::metal::get_command_encoder(stream());
    for (auto& out : outputs) {
      out.set_data(mx::allocator::malloc(out.nbytes()));
      if (out.buffer().ptr() == nullptr) {
        throw std::runtime_error("MLX failed to allocate TVM-FFI output buffer");
      }
      encoder.register_output_array(out);
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

    auto* command_buffer = encoder.finish_encoding_and_get_command_buffer();
    for (auto& out : outputs) {
      zero_output_buffer(command_buffer, out);
    }

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
};

std::vector<mx::array> tvm_ffi_metal_call(
    uint64_t func_handle,
    const std::vector<mx::array>& inputs,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<std::string>& output_dtypes,
    const std::vector<int64_t>& result_indices,
    int64_t num_params,
    mx::StreamOrDevice s = {}) {
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
      mx::to_stream(s),
      func_handle,
      num_params,
      result_indices);
  return mx::array::make_arrays(std::move(shapes), dtypes, primitive, inputs);
}

}  // namespace tilelang::mlx_tvm_ffi

NB_MODULE(_tilelang_mlx_tvm_ffi, m) {
  m.doc() = "Native graph-safe MLX primitive for TileLang TVM-FFI Metal kernels";
  m.def(
      "metal_call",
      [](uint64_t func_handle,
         nb::handle inputs,
         nb::handle output_shapes,
         nb::handle output_dtypes,
         nb::handle result_indices,
         int64_t num_params) {
        return tilelang::mlx_tvm_ffi::tvm_ffi_metal_call(
            func_handle,
            tilelang::mlx_tvm_ffi::parse_array_sequence(inputs),
            tilelang::mlx_tvm_ffi::parse_shape_sequence(output_shapes),
            tilelang::mlx_tvm_ffi::parse_string_sequence(output_dtypes),
            tilelang::mlx_tvm_ffi::parse_i64_sequence(result_indices),
            num_params);
      },
      "func_handle"_a,
      "inputs"_a,
      "output_shapes"_a,
      "output_dtypes"_a,
      "result_indices"_a,
      "num_params"_a);
  m.def("debug_state", &tilelang::mlx_tvm_ffi::debug_state);
  m.def("reset_debug_state", &tilelang::mlx_tvm_ffi::reset_debug_counters);
}
