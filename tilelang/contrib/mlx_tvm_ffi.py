"""Native MLX graph primitive for TileLang TVM-FFI Metal kernels."""

from __future__ import annotations

from typing import Any, Iterable


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
    native = _load_native_module()
    native.reset_debug_state()


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
    name = str(dtype)
    if name.startswith("mlx.core."):
        return name.removeprefix("mlx.core.")
    if name.startswith("torch."):
        return name.removeprefix("torch.")
    if name == "bool":
        return "bool_"
    return name


def metal_call(
    func: Any,
    *,
    inputs: Iterable[Any],
    output_shapes: Iterable[Iterable[int]],
    output_dtypes: Iterable[Any],
    result_indices: Iterable[int],
    num_params: int,
):
    """Create MLX graph outputs that call a TVM-FFI Metal function at eval time.

    This function intentionally does not ask MLX arrays for DLPack capsules.
    The returned arrays are backed by a native MLX primitive; real MTLBuffer
    pointers are read only inside the primitive's ``eval_gpu``.
    """

    native = _load_native_module()
    return native.metal_call(
        _function_handle(func),
        list(inputs),
        [[int(dim) for dim in shape] for shape in output_shapes],
        [_dtype_name(dtype) for dtype in output_dtypes],
        [int(idx) for idx in result_indices],
        int(num_params),
    )
