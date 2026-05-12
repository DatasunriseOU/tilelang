"""Utilities to adapt TVM FFI kernels to Torch tensors.

This adapter intentionally captures PyTorch's current CUDA stream and device
via light-weight callables so that, when the wrapped function is invoked,
the execution observes the same stream context as the active Torch code.
On non-CUDA builds, the stream/device fall back to 0/CPU semantics.
"""

from __future__ import annotations

from typing import Callable, Any
import sys

import torch
from tilelang import tvm
from tvm import runtime, tir
from tvm.target import Target
from tvm.relax import TensorType
from tilelang.utils.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils.language import retrieve_func_from_module
from tilelang.engine.param import KernelParam
from tilelang.language.dtypes import dtype
from tilelang.contrib.mlx_interop import (
    DLPackDeviceError,
    MLX_OUTPUT_WRITE_ONLY,
    first_mlx_array_device,
    has_mlx_arrays,
    is_mlx_array,
    maybe_mlx_metal_external_command_buffer,
    mlx_arrays_to_tvm_tensors,
    mlx_metal_output,
    validate_dlpack_inputs_for_target,
)


COMPILE_ARGS = {}

if sys.platform == "darwin":
    import os

    from torch.utils import cpp_extension

    include_paths = list(cpp_extension.include_paths())
    tvm_ffi_include = os.environ.get("TVM_FFI_INCLUDE_PATH")
    if not tvm_ffi_include:
        tvm_home = os.environ.get("TVM_HOME")
        if tvm_home:
            candidate = os.path.join(tvm_home, "3rdparty", "tvm-ffi", "include")
            if os.path.isdir(candidate):
                tvm_ffi_include = candidate
    if tvm_ffi_include and os.path.isdir(tvm_ffi_include):
        include_paths.append(tvm_ffi_include)
        tvm_ffi_root = os.path.dirname(tvm_ffi_include)
        dlpack_include = os.environ.get("TVM_FFI_DLPACK_INCLUDE_PATH")
        if not dlpack_include:
            candidate = os.path.join(tvm_ffi_root, "3rdparty", "dlpack", "include")
            if os.path.isdir(candidate):
                dlpack_include = candidate
        if dlpack_include and os.path.isdir(dlpack_include):
            include_paths.append(dlpack_include)

    COMPILE_ARGS["options"] = ["-x", "objective-c++", "-g", "-std=gnu++17"] + ["-I" + i for i in include_paths]


class TVMFFIKernelAdapter(BaseKernelAdapter):
    """Adapter that runs a TVM runtime.Executable with Torch tensors.

    Notes
    - We capture the "current" PyTorch CUDA stream/device as thunks (callables)
      rather than materializing them at construction time. This ensures the
      actual stream/device is read just-in-time when the function runs, matching
      the user's current Torch context (e.g., after a stream guard/switch).
    - The stream pointer returned is a raw CUDA stream handle compatible with
      TVM's device API; on CPU or when CUDA is unavailable, we return 0.
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    executable: tvm.runtime.Executable | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None
    # host_mod
    host_mod: tvm.IRModule | None = None
    # device_mod
    device_mod: tvm.IRModule | None = None
    # rt_mod
    rt_mod: tvm.runtime.Module | None = None
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: dict[tir.Var, tuple[int, int, int]] | None = None

    # Stream/device functors are inherited from BaseKernelAdapter
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        rt_mod: tvm.runtime.Module | None = None,
        host_kernel_source: str | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """Initialize the adapter with the given TIR function or module.

        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target.canon_target(determine_target(target))

        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.kernel_global_source = self.device_kernel_source

        self._post_init()

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int, int, int]]:
        """Extract information about dynamic shapes from the TIR function.

        Maps symbolic variables to their corresponding (id, buffer_index, dimension, stride_scale)
        for runtime shape resolution.
        id represents shape or stride, 0 represents shape, 1 represents stride, 2 represents scalar param.
        stride_scale compensates for sub-byte dtypes (e.g. float4_e2m1fn) where torch strides
        are in storage units but the kernel expects logical element strides.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            if isinstance(param, tir.Var) and (param not in dynamic_symbolic_map):
                dynamic_symbolic_map[param] = (2, i, -1, 1)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (0, i, j, 1)
        for i, param in enumerate(params):
            if param in buffer_map:
                buffer = buffer_map[param]
                element_bits = buffer.dtype.bits * buffer.dtype.lanes
                stride_scale = 8 // element_bits if element_bits < 8 else 1
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tir.Var) and (stride not in dynamic_symbolic_map) and (stride not in params):
                        dynamic_symbolic_map[stride] = (1, i, j, stride_scale)
        return dynamic_symbolic_map

    def _convert_torch_func(self) -> Callable[..., Any]:
        # Capture thunks that reflect Torch's current stream and device.
        # These are evaluated at call time to align TVM execution with the
        # caller's active PyTorch stream/device.
        # current_stream_functor = self.get_current_stream_functor()
        current_device_functor = self.get_current_device_functor()

        # Convert TVM types to native Python types during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        param_dtypes = [param.torch_dtype() for param in self.params]
        # Convert TVM shape arrays to native Python lists
        param_shapes = []

        for param in self.params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            tl_dtype = param.dtype
            if tl_dtype.bits < 8:
                stroage_dtype: dtype = dtype(param.torch_dtype())
                # last dim divide by bits to get the actual shape
                native_shape[-1] = native_shape[-1] * tl_dtype.bits * tl_dtype.lanes // (stroage_dtype.bits * stroage_dtype.lanes)
            param_shapes.append(native_shape)

        if self.executable is None:
            self.executable = runtime.Executable(self.rt_mod)
            if COMPILE_ARGS:
                # Precompile jit module with extra arguments
                self.executable.jit(**COMPILE_ARGS)

        dynamic_symbolic_map = self._process_dynamic_symbolic()
        executable = self.executable

        # Prepare helpers for friendly dtype error messages
        prim_func = self.prim_func
        buffer_map = prim_func.buffer_map
        params = prim_func.params
        # Expected dtype string per parameter index (for buffers only)
        expected_dtype_strs: list[str | None] = []
        # Track whether each param is a buffer (has dtype) vs scalar
        is_buffer_param: list[bool] = []
        for p in params:
            if p in buffer_map:
                expected_dtype_strs.append(str(buffer_map[p].dtype))
                is_buffer_param.append(True)
            else:
                expected_dtype_strs.append(None)
                is_buffer_param.append(False)

        def normalize_out_argument(out_arg: Any) -> list[Any] | None:
            if out_arg is None:
                return None
            if not self.result_idx:
                raise ValueError("Output buffers can only be provided when out_idx/result_idx is set.")
            if len(self.result_idx) == 1:
                return [out_arg]
            if not isinstance(out_arg, (list, tuple)):
                raise ValueError(f"Kernel expected {len(self.result_idx)} output buffers, but out= is not a sequence.")
            if len(out_arg) != len(self.result_idx):
                raise ValueError(f"Kernel expected {len(self.result_idx)} output buffers, but {len(out_arg)} are provided.")
            return list(out_arg)

        def _shape_dim(tensor: Any, dim: int) -> int:
            return int(tensor.shape[dim])

        def _compact_stride(shape: tuple[int, ...], dim: int) -> int:
            stride = 1
            for extent in reversed(shape[dim + 1:]):
                stride *= int(extent)
            return stride

        def _stride_dim(tensor: Any, dim: int) -> int:
            stride = getattr(tensor, "stride", None)
            if callable(stride):
                return int(stride()[dim])
            strides = getattr(tensor, "strides", None)
            if strides is not None:
                return int(strides[dim])
            if is_mlx_array(tensor):
                return _compact_stride(tuple(int(s) for s in tensor.shape), dim)
            raise ValueError(
                f"Cannot resolve dynamic stride from {type(tensor).__name__}; "
                "pass a DLPack tensor with exposed strides or specialize the stride."
            )

        def func(*inputs: torch.Tensor | Any, out: Any | None = None):
            # Validate input count.  The compact calling convention omits
            # result_idx outputs so the adapter allocates them; the full ABI
            # convention supplies every PrimFunc parameter and reuses caller
            # owned result buffers.
            expected_inputs = len(self.params) - len(self.result_idx)
            output_overrides = normalize_out_argument(out)
            using_full_abi_args = (
                output_overrides is None and bool(self.result_idx) and len(inputs) == len(self.params)
            )
            if output_overrides is not None and len(inputs) == len(self.params):
                raise ValueError("Output buffers were provided both positionally and via out=.")
            if output_overrides is not None:
                if len(inputs) != expected_inputs:
                    raise ValueError(f"Kernel expected {expected_inputs} inputs with out=, but {len(inputs)} are provided.")
                provided_outputs = dict(zip(self.result_idx, output_overrides))
            elif using_full_abi_args:
                provided_outputs = {idx: inputs[idx] for idx in self.result_idx}
            else:
                provided_outputs = {}
                if len(inputs) != expected_inputs:
                    if self.result_idx:
                        raise ValueError(
                            f"Kernel expected {expected_inputs} inputs, or {len(self.params)} full ABI arguments "
                            f"including output buffers, but {len(inputs)} are provided."
                        )
                    raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")

            dlpack_args = inputs
            if output_overrides is not None:
                dlpack_args = inputs + tuple(output_overrides)
            target_kind = self.target.kind.name
            uses_mlx_runtime = has_mlx_arrays(dlpack_args)
            if uses_mlx_runtime and target_kind != "metal":
                raise DLPackDeviceError(
                    f"MLX arrays export Metal DLPack buffers, but this kernel targets {target_kind!r}."
                )
            validate_dlpack_inputs_for_target(dlpack_args, target_kind)
            if uses_mlx_runtime:
                first_mlx_array_device(dlpack_args)

            # Resolve the device used for outputs. Prefer the first tensor input's device
            # if available, otherwise use PyTorch's current device.
            out_device: torch.device | None = None

            # Stitch the full positional argument list expected by the TVM executable
            ins_idx: int = 0
            tensor_list: list[Any] = []

            # Prepare input and output tensors
            for i in range(len(self.params)):
                if using_full_abi_args:
                    tensor = inputs[i]
                elif i in provided_outputs:
                    tensor = provided_outputs[i]
                elif i in self.result_idx:
                    shape = []
                    # Now working with native Python list, no FFI calls needed
                    for s in param_shapes[i]:
                        if isinstance(s, tir.Var):
                            for key in dynamic_symbolic_map:
                                if str(s) == str(key):
                                    ref_id, ref_tensor_idx, ref_shape_idx, stride_scale = dynamic_symbolic_map[key]
                                    if ref_id == 2:
                                        shape.append(inputs[ref_tensor_idx])
                                    elif ref_id == 0:
                                        shape.append(_shape_dim(tensor_list[ref_tensor_idx], ref_shape_idx))
                                    elif ref_id == 1:
                                        shape.append(
                                            _stride_dim(tensor_list[ref_tensor_idx], ref_shape_idx)
                                            * stride_scale
                                        )
                        else:  # Already converted to Python int during initialization
                            shape.append(s)

                    if len(shape) == 0:
                        param_name = self.params[i].name if hasattr(self.params[i], "name") else f"parameter_{i}"
                        raise ValueError(
                            f"Cannot create output tensor (name={param_name}) - 0-dimensional tensors are not supported. "
                            f"Expected shape: {shape}"
                        )
                    dtype = param_dtypes[i]
                    if uses_mlx_runtime:
                        tensor = mlx_metal_output(
                            shape,
                            expected_dtype_strs[i],
                            policy=MLX_OUTPUT_WRITE_ONLY,
                        )
                    else:
                        if out_device is None:
                            out_device = current_device_functor()
                        tensor = torch.empty(*shape, dtype=dtype, device=out_device)
                else:
                    tensor = inputs[ins_idx]
                    ins_idx += 1
                tensor_list.append(tensor)

            exec_tensor_list = (
                mlx_arrays_to_tvm_tensors(
                    tensor_list,
                    expected_dtypes=expected_dtype_strs,
                )
                if uses_mlx_runtime
                else tensor_list
            )
            with maybe_mlx_metal_external_command_buffer(tensor_list):
                executable(*exec_tensor_list)

            # Return outputs in the requested form
            if len(self.result_idx) == 1:
                return tensor_list[self.result_idx[0]]
            return [tensor_list[i] for i in self.result_idx]

        return func

    @classmethod
    def from_database(
        cls,
        params: list[TensorType],
        result_idx: list[int],
        target: str,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        host_kernel_source: str,
        device_kernel_source: str,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.host_kernel_source = host_kernel_source
        adapter.device_kernel_source = device_kernel_source
        adapter.wrapped_source = device_kernel_source + "\n\n" + host_kernel_source
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        target = determine_target(target, return_object=True)
        adapter.target = Target.canon_target(determine_target(target))

        adapter.verbose = verbose
        adapter.libpath = kernel_lib_path
        adapter.kernel_global_source = device_kernel_source
        adapter.executable = runtime.load_module(kernel_lib_path)
        adapter._post_init()
        return adapter

    def get_host_source(self):
        """Returns the source code of the host module."""
        if self.host_kernel_source is not None:
            return self.host_kernel_source
        return self.rt_mod.inspect_source()

    def get_device_source(self):
        """Returns the source code of the device module."""
        if self.device_kernel_source is not None:
            return self.device_kernel_source
        return self.rt_mod.imports[0].inspect_source()

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.get_device_source()
        else:
            return self.get_device_source() + "\n\n" + self.get_host_source()

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)
