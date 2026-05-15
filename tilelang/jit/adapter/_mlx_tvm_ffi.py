"""Private MLX graph primitive bridge for the TVM-FFI adapter.

This module is an implementation detail of ``tilelang.jit.adapter.tvm_ffi``.
Application/runtime code should call compiled TileLang kernels; it should not
import or call this bridge directly.
"""

from __future__ import annotations

import weakref
from typing import Any, Iterable

from tilelang.analysis.metal_graph_sync import (
    MetalLaunchDependencyMetadata,
    make_tvm_ffi_metal_dependency_metadata,
    plan_mlx_tvm_ffi_launch,
    register_mlx_tvm_ffi_outputs,
)


class MLXTVMFFIBridgeUnavailable(RuntimeError):
    """Raised when the optional native MLX/TVM-FFI bridge is unavailable."""


_NATIVE_MODULE: Any | None = None
_COMPACT_INPUT_CACHE: dict[int, weakref.ref[Any]] = {}
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


def _known_compact_mlx_input(value: Any) -> bool:
    entry = _COMPACT_INPUT_CACHE.get(id(value))
    if entry is None:
        return False
    if entry() is value:
        return True
    _COMPACT_INPUT_CACHE.pop(id(value), None)
    return False


def _remember_compact_mlx_input(value: Any) -> None:
    key = id(value)

    def remove(_ref: weakref.ref[Any], *, stale_key: int = key) -> None:
        _COMPACT_INPUT_CACHE.pop(stale_key, None)

    try:
        _COMPACT_INPUT_CACHE[key] = weakref.ref(value, remove)
    except TypeError:
        pass


def _contiguous_mlx_input(value: Any) -> Any:
    """Preserve MLX graph inputs for the native TVM-FFI boundary.

    Always route inputs through ``native.compact_input`` (mx.contiguous on
    the GPU stream). The wrap-time ``is_compact`` check is unreliable
    because MLX rewrites strides during eval: a slice node looks compact
    at graph-construction time but becomes a non-compact view of the
    parent buffer after evaluation. ``mx.contiguous`` is a near-no-op
    for already-row-contiguous inputs and an explicit row-contiguous
    copy otherwise, matching what ``mx.fast.metal_kernel`` does via its
    ``ensure_row_contiguous`` path.
    """

    if not _is_mlx_array(value):
        return value
    native = _load_native_module()
    return native.compact_input(value)


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
    input_list = [_contiguous_mlx_input(value) for value in inputs]
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
            list(owner_outputs),
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
