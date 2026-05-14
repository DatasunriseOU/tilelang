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
    register_mlx_tvm_ffi_outputs,
)


class MLXTVMFFIBridgeUnavailable(RuntimeError):
    """Raised when the optional native MLX/TVM-FFI bridge is unavailable."""


def _load_native_module():
    try:
        import _tilelang_mlx_tvm_ffi  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional native build
        raise MLXTVMFFIBridgeUnavailable(
            "native MLX TVM-FFI bridge is not built or cannot be loaded"
        ) from exc
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
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception:
        return False
    return isinstance(value, mx.array)


def _contiguous_mlx_input(value: Any) -> Any:
    """Preserve MLX graph inputs for the native TVM-FFI boundary.

    Compact inputs keep object identity so producer/consumer dependency
    metadata remains exact. Non-compact views get an explicit native MLX
    compact-layout graph node; that is the framework lowering step required by
    the flat TileLang TVM-FFI ABI, not a Python-side ``mx.contiguous`` repair.
    """

    if has_mlx_tvm_ffi_producer(value):
        if not _is_mlx_array(value):
            return value
        native = _load_native_module()
        if native.is_compact(value):
            return value
        return native.compact_input(value)
    if _is_mlx_array(value):
        native = _load_native_module()
        return native.compact_input(value)
    return value


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
            _function_handle(func),
            input_list,
            output_shape_list,
            output_dtype_list,
            result_index_list,
            int(num_params),
            [int(idx) for idx in zero_init_output_positions],
            launch_sync_state,
            wait_edges,
            param_dtypes=param_dtype_names,
        )
    else:
        outputs = native.metal_call_owner_outputs(
            _function_handle(func),
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
        )
    output_list = list(outputs)
    register_mlx_tvm_ffi_outputs(
        output_list,
        launch_sync_state,
        dependency_metadata=dependency_metadata,
    )
    return output_list
