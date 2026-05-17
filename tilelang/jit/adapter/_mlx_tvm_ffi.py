"""Private MLX graph primitive bridge for the TVM-FFI adapter.

This module is an implementation detail of ``tilelang.jit.adapter.tvm_ffi``.
Application/runtime code should call compiled TileLang kernels; it should not
import or call this bridge directly.
"""

from __future__ import annotations

from typing import Any, Iterable

from tilelang.analysis.metal_graph_sync import (
    MetalLaunchDependencyMetadata,
    has_mlx_tvm_ffi_producer,
    make_tvm_ffi_metal_dependency_metadata,
    plan_mlx_tvm_ffi_launch,
    register_mlx_tvm_ffi_output,
    register_mlx_tvm_ffi_outputs,
)


class MLXTVMFFIBridgeUnavailable(RuntimeError):
    """Raised when the optional native MLX/TVM-FFI bridge is unavailable."""


_NATIVE_MODULE: Any | None = None
_MLX_ARRAY_TYPE: type[Any] | None | bool = None


def _load_native_module():
    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE
    try:
        import _tilelang_mlx_tvm_ffi  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional native build
        raise MLXTVMFFIBridgeUnavailable(
            "native MLX TVM-FFI bridge is not built or cannot be loaded"
        ) from exc
    _NATIVE_MODULE = _tilelang_mlx_tvm_ffi
    return _tilelang_mlx_tvm_ffi


def is_available() -> bool:
    try:
        _load_native_module()
    except MLXTVMFFIBridgeUnavailable:
        return False
    return True


def debug_state() -> dict[str, Any]:
    native = _load_native_module()
    return dict(native.debug_state())


def reset_debug_state() -> None:
    from tilelang.analysis.metal_graph_sync import clear_metal_graph_sync_state_for_tests

    native = _load_native_module()
    native.reset_debug_state()
    clear_metal_graph_sync_state_for_tests()


def owner_output_buffer(shape: Iterable[int], dtype: Any):
    """Allocate a materialized MLX owner-output buffer in the native bridge."""

    native = _load_native_module()
    return native.owner_output_buffer([int(dim) for dim in shape], _dtype_name(dtype))


def owner_output_buffers(
    shapes: Iterable[Iterable[int]],
    dtypes: Iterable[Any],
):
    """Allocate materialized MLX owner-output buffers in the native bridge."""

    native = _load_native_module()
    return list(
        native.owner_output_buffers(
            [[int(dim) for dim in shape] for shape in shapes],
            [_dtype_name(dtype) for dtype in dtypes],
        )
    )


def _function_handle(func: Any) -> int:
    chandle = getattr(func, "__chandle__", None)
    if callable(chandle):
        handle = int(chandle())
    else:
        handle = int(func)
    if handle == 0:
        raise ValueError("TVM-FFI function handle is null")
    return handle


def _dtype_name(dtype: Any) -> str:
    if dtype is None:
        return ""
    name = str(dtype)
    if name.startswith("mlx.core."):
        return name.removeprefix("mlx.core.")
    if name.startswith("torch."):
        return name.removeprefix("torch.")
    if name == "bool":
        return "bool_"
    return name


def _is_mlx_array(value: Any) -> bool:
    global _MLX_ARRAY_TYPE
    if _MLX_ARRAY_TYPE is None:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except Exception:
            _MLX_ARRAY_TYPE = False
            return False
        _MLX_ARRAY_TYPE = mx.array
    if _MLX_ARRAY_TYPE is False:
        return False
    return isinstance(value, _MLX_ARRAY_TYPE)


def _contiguous_mlx_input(value: Any) -> Any:
    """Preserve MLX graph inputs for the native TVM-FFI boundary.

    TVM-FFI graph outputs are already compact by construction and must keep
    their producer registry identity so the dependency planner can see
    producer->consumer hazards. For other inputs, trust only already
    materialized compact zero-offset arrays. Lazy MLX graph nodes and views
    still go through ``native.compact_input`` because wrap-time shape/stride
    metadata can change when MLX evaluates the graph.
    """

    if not _is_mlx_array(value):
        return value
    native = _load_native_module()
    if native.can_borrow_compact_input(value):
        return value
    if has_mlx_tvm_ffi_producer(value):
        return value
    return native.compact_input(value)


def _can_use_borrowed_no_wait(values: Iterable[Any]) -> bool:
    native = _load_native_module()
    for value in values:
        if not _is_mlx_array(value):
            continue
        if has_mlx_tvm_ffi_producer(value):
            return False
        if not native.can_borrow_compact_input(value):
            return False
    return True


def prepare_metal_call(
    func: Any,
    *,
    output_shapes: Iterable[Iterable[int]],
    output_dtypes: Iterable[Any],
    result_indices: Iterable[int],
    num_params: int,
    param_dtypes: Iterable[Any | None] | None = None,
    param_shapes: Iterable[Iterable[int]] | None = None,
    direct_func: Any | None = None,
    direct_launch_args: Iterable[int] | None = None,
    direct_param_indices: Iterable[int] | None = None,
    direct_module: Any | None = None,
    direct_kernel_name: str | None = None,
    zero_init_output_positions: Iterable[int] = (),
):
    """Pre-parse static metadata for repeated native MLX graph launches."""

    native = _load_native_module()
    func_for_native_call = direct_func if direct_func is not None else func
    param_dtype_names = None
    if param_dtypes is not None:
        param_dtype_names = [_dtype_name(dtype) for dtype in param_dtypes]
    param_shape_list = None
    if param_shapes is not None:
        param_shape_list = [[int(dim) for dim in shape] for shape in param_shapes]
    direct_launch_arg_list = None
    if direct_launch_args is not None:
        direct_launch_arg_list = [int(value) for value in direct_launch_args]
    direct_param_index_list = None
    if direct_param_indices is not None:
        direct_param_index_list = [int(value) for value in direct_param_indices]
    direct_module_handle = 0
    if direct_module is not None:
        direct_module_handle = _function_handle(direct_module)
    direct_kernel_name_str = "" if direct_kernel_name is None else str(direct_kernel_name)
    return native.prepare_metal_call(
        _function_handle(func_for_native_call),
        [[int(dim) for dim in shape] for shape in output_shapes],
        [_dtype_name(dtype) for dtype in output_dtypes],
        [int(idx) for idx in result_indices],
        int(num_params),
        param_dtype_names,
        param_shape_list,
        direct_launch_arg_list,
        direct_param_index_list,
        direct_module_handle,
        direct_kernel_name_str,
        [int(idx) for idx in zero_init_output_positions],
    )


def prepared_metal_call(
    prepared: Any,
    *,
    inputs: Iterable[Any],
    owner_outputs: Iterable[Any] | None = None,
    command_buffer_domain: Any | None = None,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
):
    """Create MLX graph outputs from a pre-parsed native Metal call."""

    native = _load_native_module()
    raw_input_list = list(inputs)
    owner_output_list = None if owner_outputs is None else list(owner_outputs)
    if _can_use_borrowed_no_wait(raw_input_list):
        fast_result = native.prepared_metal_call_borrowed_no_wait(
            prepared,
            raw_input_list,
            owner_output_list,
        )
        if fast_result is not None:
            outputs, launch_sync_state = fast_result
            output_list = list(outputs)
            register_mlx_tvm_ffi_outputs(
                output_list,
                launch_sync_state,
                dependency_metadata=dependency_metadata,
                command_buffer_domain=command_buffer_domain,
            )
            return output_list

    input_list = [_contiguous_mlx_input(value) for value in raw_input_list]
    launch_sync_state, wait_edges = plan_mlx_tvm_ffi_launch(
        native,
        input_list,
        dependency_metadata=dependency_metadata,
        command_buffer_domain=command_buffer_domain,
    )
    if owner_outputs is None:
        outputs = native.prepared_metal_call(
            prepared,
            input_list,
            launch_sync_state,
            wait_edges,
        )
    else:
        outputs = native.prepared_metal_call_owner_outputs(
            prepared,
            input_list,
            owner_output_list,
            launch_sync_state,
            wait_edges,
        )
    output_list = list(outputs)
    register_mlx_tvm_ffi_outputs(
        output_list,
        launch_sync_state,
        dependency_metadata=dependency_metadata,
        command_buffer_domain=command_buffer_domain,
    )
    return output_list


def prepared_metal_call_args(
    prepared: Any,
    *args: Any,
    owner_output_count: int = 0,
    command_buffer_domain: Any | None = None,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
):
    """Create MLX graph outputs from positional prepared-call arrays.

    The common full-ABI owner-output path already has arrays in argument
    order. Passing them directly avoids allocating Python input/output lists
    and lets the native bridge parse nanobind varargs instead of iterating a
    Python sequence on every tiny launch.
    """

    native = _load_native_module()
    owner_output_count_int = int(owner_output_count)
    input_args = args[: len(args) - owner_output_count_int] if owner_output_count_int else args
    if _can_use_borrowed_no_wait(input_args):
        fast_result = native.prepared_metal_call_borrowed_no_wait_args(
            prepared,
            owner_output_count_int,
            *args,
        )
        if fast_result is not None:
            outputs, launch_sync_state = fast_result
            output_list = list(outputs)
            register_mlx_tvm_ffi_outputs(
                output_list,
                launch_sync_state,
                dependency_metadata=dependency_metadata,
                command_buffer_domain=command_buffer_domain,
            )
            return output_list

    if owner_output_count_int:
        split = len(args) - owner_output_count_int
        return prepared_metal_call(
            prepared,
            inputs=args[:split],
            owner_outputs=args[split:],
            command_buffer_domain=command_buffer_domain,
            dependency_metadata=dependency_metadata,
        )
    return prepared_metal_call(
        prepared,
        inputs=args,
        command_buffer_domain=command_buffer_domain,
        dependency_metadata=dependency_metadata,
    )


def prepared_metal_call_single_arg(
    prepared: Any,
    *args: Any,
    owner_output_count: int = 0,
    command_buffer_domain: Any | None = None,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
):
    """Create one MLX graph output from positional prepared-call arrays."""

    native = _load_native_module()
    owner_output_count_int = int(owner_output_count)
    input_args = args[: len(args) - owner_output_count_int] if owner_output_count_int else args
    if _can_use_borrowed_no_wait(input_args):
        if owner_output_count_int == 1 and len(args) == 5:
            fast_result = native.prepared_metal_call_borrowed_no_wait_4in1out_single(
                prepared,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
            )
        else:
            fast_result = native.prepared_metal_call_borrowed_no_wait_args_single(
                prepared,
                owner_output_count_int,
                *args,
            )
        if fast_result is not None:
            output, launch_sync_state = fast_result
            register_mlx_tvm_ffi_output(
                output,
                launch_sync_state,
                dependency_metadata=dependency_metadata,
                command_buffer_domain=command_buffer_domain,
            )
            return output

    outputs = prepared_metal_call_args(
        prepared,
        *args,
        owner_output_count=owner_output_count_int,
        command_buffer_domain=command_buffer_domain,
        dependency_metadata=dependency_metadata,
    )
    if len(outputs) != 1:
        raise RuntimeError("single-output prepared native call returned multiple outputs")
    return outputs[0]


def prepared_metal_call_4in1out_single(
    prepared: Any,
    a0: Any,
    a1: Any,
    a2: Any,
    a3: Any,
    owner_output: Any,
    *,
    command_buffer_domain: Any | None = None,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
):
    """Create one MLX graph output for the hot 4-input/1-owner-output ABI."""

    native = _load_native_module()
    if _can_use_borrowed_no_wait((a0, a1, a2, a3)):
        fast_result = native.prepared_metal_call_borrowed_no_wait_4in1out_single(
            prepared,
            a0,
            a1,
            a2,
            a3,
            owner_output,
        )
        if fast_result is not None:
            output, launch_sync_state = fast_result
            register_mlx_tvm_ffi_output(
                output,
                launch_sync_state,
                dependency_metadata=dependency_metadata,
                command_buffer_domain=command_buffer_domain,
            )
            return output

    return prepared_metal_call_single_arg(
        prepared,
        a0,
        a1,
        a2,
        a3,
        owner_output,
        owner_output_count=1,
        command_buffer_domain=command_buffer_domain,
        dependency_metadata=dependency_metadata,
    )


def metal_call(
    func: Any,
    *,
    inputs: Iterable[Any],
    owner_outputs: Iterable[Any] | None = None,
    output_shapes: Iterable[Iterable[int]],
    output_dtypes: Iterable[Any],
    result_indices: Iterable[int],
    num_params: int,
    param_dtypes: Iterable[Any | None] | None = None,
    param_shapes: Iterable[Iterable[int]] | None = None,
    direct_func: Any | None = None,
    direct_launch_args: Iterable[int] | None = None,
    direct_param_indices: Iterable[int] | None = None,
    direct_module: Any | None = None,
    direct_kernel_name: str | None = None,
    command_buffer_domain: Any | None = None,
    dependency_metadata: MetalLaunchDependencyMetadata | None = None,
    zero_init_output_positions: Iterable[int] = (),
):
    """Create MLX graph outputs that call a TVM-FFI Metal function at eval time.

    This function intentionally does not ask MLX arrays for DLPack capsules.
    The returned arrays are backed by a native MLX primitive; real MTLBuffer
    pointers are read only inside the primitive's ``eval_gpu``.
    """

    native = _load_native_module()
    input_list = [_contiguous_mlx_input(value) for value in inputs]
    owner_output_list = None
    if owner_outputs is not None:
        owner_output_list = list(owner_outputs)
    result_index_list = [int(idx) for idx in result_indices]
    param_dtype_names = None
    if param_dtypes is not None:
        param_dtype_names = [_dtype_name(dtype) for dtype in param_dtypes]
    param_shape_list = None
    if param_shapes is not None:
        param_shape_list = [[int(dim) for dim in shape] for shape in param_shapes]
    direct_launch_arg_list = None
    if direct_launch_args is not None:
        direct_launch_arg_list = [int(value) for value in direct_launch_args]
    direct_param_index_list = None
    if direct_param_indices is not None:
        direct_param_index_list = [int(value) for value in direct_param_indices]
    direct_module_handle = 0
    if direct_module is not None:
        direct_module_handle = _function_handle(direct_module)
    direct_kernel_name_str = "" if direct_kernel_name is None else str(direct_kernel_name)
    func_for_native_call = direct_func if direct_func is not None else func
    if dependency_metadata is None:
        result_index_set = set(result_index_list)
        dependency_metadata = make_tvm_ffi_metal_dependency_metadata(
            kernel_symbol="<tvm_ffi_metal>",
            input_param_indices=[i for i in range(int(num_params)) if i not in result_index_set],
            output_param_indices=result_index_list,
            command_buffer_domain=command_buffer_domain,
        )
    launch_sync_state, wait_edges = plan_mlx_tvm_ffi_launch(
        native,
        input_list,
        dependency_metadata=dependency_metadata,
    )
    output_shape_list = [[int(dim) for dim in shape] for shape in output_shapes]
    output_dtype_list = [_dtype_name(dtype) for dtype in output_dtypes]
    if owner_output_list is None:
        outputs = native.metal_call(
            _function_handle(func_for_native_call),
            input_list,
            output_shape_list,
            output_dtype_list,
            result_index_list,
            int(num_params),
            [int(idx) for idx in zero_init_output_positions],
            launch_sync_state,
            wait_edges,
            param_dtypes=param_dtype_names,
            param_shapes=param_shape_list,
            direct_launch_args=direct_launch_arg_list,
            direct_param_indices=direct_param_index_list,
            direct_module_handle=direct_module_handle,
            direct_kernel_name=direct_kernel_name_str,
        )
    else:
        outputs = native.metal_call_owner_outputs(
            _function_handle(func_for_native_call),
            input_list,
            owner_output_list,
            output_shape_list,
            output_dtype_list,
            result_index_list,
            int(num_params),
            [int(idx) for idx in zero_init_output_positions],
            launch_sync_state,
            wait_edges,
            param_dtypes=param_dtype_names,
            param_shapes=param_shape_list,
            direct_launch_args=direct_launch_arg_list,
            direct_param_indices=direct_param_index_list,
            direct_module_handle=direct_module_handle,
            direct_kernel_name=direct_kernel_name_str,
        )
    output_list = list(outputs)
    register_mlx_tvm_ffi_outputs(
        output_list,
        launch_sync_state,
        dependency_metadata=dependency_metadata,
    )
    return output_list
