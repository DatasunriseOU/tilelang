# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""MLX/Metal interop helpers for TVM FFI execution."""

from __future__ import annotations

import ctypes
import functools
import hashlib
import re
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

from tilelang import tvm

DLPACK_DEVICE_CPU = 1
DLPACK_DEVICE_CUDA = 2
DLPACK_DEVICE_METAL = 8
DLPACK_DEVICE_ROCM = 10

_DLPACK_DEVICE_NAMES = {
    DLPACK_DEVICE_CPU: "kDLCPU",
    DLPACK_DEVICE_CUDA: "kDLCUDA",
    DLPACK_DEVICE_METAL: "kDLMetal",
    DLPACK_DEVICE_ROCM: "kDLROCM",
}


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class DLPackInteropError(RuntimeError):
    """Base class for Python-side DLPack interop failures."""


class DLPackDeviceError(DLPackInteropError):
    """Raised when a DLPack producer reports an incompatible device."""


class DLPackOwnershipError(DLPackInteropError):
    """Raised when a DLPack capsule cannot transfer ownership safely."""


class DLPackConversionError(DLPackInteropError):
    """Raised when a DLPack producer cannot be imported without a copy."""


class MLXGraphInteropError(DLPackInteropError):
    """Raised when TileLang Metal source cannot be mapped to an MLX graph op."""


MLX_OUTPUT_WRITE_ONLY = "write_only"
MLX_OUTPUT_ZEROED = "zeroed"

_MSL_COMMENT_OR_STRING_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)
_KERNEL_DEF_RE = re.compile(r"\bkernel\s+void\s+(?P<name>[A-Za-z_]\w*)\s*\(")
_PARAM_NAME_RE = re.compile(r"\b([A-Za-z_]\w*)\s*$")
_METAL_BUILTIN_PARAM_NAMES = frozenset(
    {
        "blockDim",
        "blockIdx",
        "gridDim",
        "grid_size",
        "simdgroup_index_in_threadgroup",
        "threadIdx",
        "thread_execution_width",
        "thread_index_in_simdgroup",
        "thread_index_in_threadgroup",
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "threadgroups_per_grid",
        "threads_per_threadgroup",
    }
)


def _mlx_core():
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception:
        return None
    return mx


def _mask_msl_comments_and_strings(msl: str) -> str:
    return _MSL_COMMENT_OR_STRING_RE.sub(lambda match: " " * len(match.group(0)), msl)


def _rewrite_msl_code_segments(msl: str, rewrite) -> str:
    chunks: list[str] = []
    start = 0
    for match in _MSL_COMMENT_OR_STRING_RE.finditer(msl):
        chunks.append(rewrite(msl[start : match.start()]))
        chunks.append(match.group(0))
        start = match.end()
    chunks.append(rewrite(msl[start:]))
    return "".join(chunks)


def _split_kernel_msl(msl: str) -> tuple[str, str, str, str]:
    """Split TileLang-emitted MSL into prelude, kernel name, signature, body."""

    masked = _mask_msl_comments_and_strings(msl)
    match = _KERNEL_DEF_RE.search(masked)
    if match is None:
        raise MLXGraphInteropError("TileLang Metal source does not contain a kernel function")
    if _KERNEL_DEF_RE.search(masked, match.end()) is not None:
        raise MLXGraphInteropError("TileLang Metal source contains multiple kernel functions")

    prelude = msl[: match.start()].rstrip()
    kernel_name = match.group("name")

    sig_start = match.end()
    depth = 1
    i = sig_start
    while i < len(msl) and depth > 0:
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise MLXGraphInteropError("TileLang Metal source has an unbalanced kernel signature")
    sig_text = msl[sig_start : i - 1]

    j = i
    while j < len(msl) and msl[j].isspace():
        j += 1
    if j >= len(msl) or msl[j] != "{":
        raise MLXGraphInteropError("TileLang Metal source is missing a kernel body")
    body_start = j
    depth = 1
    j += 1
    while j < len(msl) and depth > 0:
        ch = masked[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        raise MLXGraphInteropError("TileLang Metal source has an unbalanced kernel body")
    return prelude, kernel_name, sig_text, msl[body_start:j]


def _split_signature_decls(sig_text: str) -> list[str]:
    decls: list[str] = []
    current: list[str] = []
    depth_paren = 0
    depth_attr = 0
    i = 0
    while i < len(sig_text):
        ch = sig_text[i]
        if ch == "[" and i + 1 < len(sig_text) and sig_text[i + 1] == "[":
            depth_attr += 1
            current.append("[[")
            i += 2
            continue
        if ch == "]" and i + 1 < len(sig_text) and sig_text[i + 1] == "]" and depth_attr:
            depth_attr -= 1
            current.append("]]")
            i += 2
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        if ch == "," and depth_paren == 0 and depth_attr == 0:
            decls.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        decls.append(tail)
    return decls


def _strip_attribute_markers(decl: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(decl):
        if decl[i] == "[" and i + 1 < len(decl) and decl[i + 1] == "[":
            depth = 1
            i += 2
            while i < len(decl) and depth:
                if decl[i] == "[" and i + 1 < len(decl) and decl[i + 1] == "[":
                    depth += 1
                    i += 2
                elif decl[i] == "]" and i + 1 < len(decl) and decl[i + 1] == "]":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            out.append(" ")
            continue
        out.append(decl[i])
        i += 1
    return "".join(out)


def _extract_param_identifier(decl: str) -> str | None:
    cleaned = _strip_attribute_markers(decl).strip()
    cleaned = re.sub(r"\[[^\]]*\]\s*$", "", cleaned).strip()
    cleaned = cleaned.replace("*", " ").replace("&", " ").strip()
    match = _PARAM_NAME_RE.search(cleaned)
    return match.group(1) if match else None


def _parse_buffer_param_names(sig_text: str) -> list[str]:
    names: list[str] = []
    for decl in _split_signature_decls(sig_text):
        clean = _strip_attribute_markers(decl).strip()
        if not clean or re.search(r"\bthreadgroup\b", clean):
            continue
        if re.search(r"_args_t\s*&", clean):
            continue
        if not (re.search(r"\bdevice\b", clean) or re.search(r"\bconstant\b", clean)):
            continue
        ident = _extract_param_identifier(clean)
        if ident is None or ident in _METAL_BUILTIN_PARAM_NAMES:
            continue
        names.append(ident)
    return names


def _metal_builtin_for_tilelang_alias(alias: str, axis: str) -> str:
    if alias == "threadIdx":
        return f"thread_position_in_threadgroup.{axis}"
    if alias == "blockIdx":
        return f"threadgroup_position_in_grid.{axis}"
    if alias == "blockDim":
        return f"threads_per_threadgroup.{axis}"
    if alias == "gridDim":
        return f"threadgroups_per_grid.{axis}"
    raise ValueError(f"unexpected TileLang builtin alias: {alias}")


def _canonicalize_tilelang_builtin_aliases(body: str) -> str:
    def rewrite(code: str) -> str:
        code = re.sub(
            r"\b(?P<alias>threadIdx|blockIdx|blockDim|gridDim)\.(?P<axis>[xyz])\b",
            lambda m: _metal_builtin_for_tilelang_alias(m.group("alias"), m.group("axis")),
            code,
        )
        return code

    return _rewrite_msl_code_segments(body, rewrite)


def _canonicalize_metal_surface(body: str) -> str:
    def rewrite(code: str) -> str:
        code = re.sub(
            r"\bthreadgroup_barrier\s*\(\s*mem_flags::mem_threadgroup\s*\)",
            "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup)",
            code,
        )
        code = re.sub(r"\bmemory_order_relaxed\b", "metal::memory_order_relaxed", code)
        for name in (
            "atomic_fetch_add_explicit",
            "atomic_fetch_min_explicit",
            "atomic_fetch_max_explicit",
        ):
            code = re.sub(rf"(?<![:\w]){name}\b", f"metal::{name}", code)
        return code

    return _rewrite_msl_code_segments(body, rewrite)


def _tilelang_msl_body_for_mlx(body_text: str) -> str:
    body = body_text[1:-1]
    body = (
        "    uint3 blockIdx = threadgroup_position_in_grid;\n"
        "    uint3 threadIdx = thread_position_in_threadgroup;\n"
        "    uint3 blockDim = threads_per_threadgroup;\n"
        "    uint3 gridDim = threadgroups_per_grid;\n"
        + body
    )
    body = _canonicalize_tilelang_builtin_aliases(body)
    return _canonicalize_metal_surface(body)


@functools.lru_cache(maxsize=128)
def _cached_mlx_tilelang_metal_kernel(
    msl_source: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
):
    mx = _mlx_core()
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None) if mx is not None else None
    if metal_kernel is None:
        raise MLXGraphInteropError("mlx.core.fast.metal_kernel is not available")

    prelude, kernel_name, sig_text, body_text = _split_kernel_msl(msl_source)
    buffer_names = _parse_buffer_param_names(sig_text)
    expected = set(input_names) | set(output_names)
    missing = expected.difference(buffer_names)
    if missing:
        raise MLXGraphInteropError(
            "TileLang Metal source is missing expected MLX buffers: "
            + ", ".join(sorted(missing))
        )
    unsupported = set(buffer_names).difference(expected)
    if unsupported:
        raise MLXGraphInteropError(
            "TileLang Metal source has unsupported non-MLX buffers: "
            + ", ".join(sorted(unsupported))
        )

    abi_fingerprint = "\0".join(
        (msl_source, *input_names, "\1", *output_names)
    ).encode()
    kernel_digest = hashlib.sha1(abi_fingerprint).hexdigest()[:16]
    header = prelude + ("\n" if prelude and not prelude.endswith("\n") else "")
    source = _tilelang_msl_body_for_mlx(body_text)
    return metal_kernel(
        name=f"tilelang_{kernel_name}_{kernel_digest}",
        input_names=list(input_names),
        output_names=list(output_names),
        source=source,
        header=header,
        ensure_row_contiguous=False,
    )


def mlx_tilelang_metal_kernel(
    msl_source: str | None,
    *,
    input_names: Iterable[str],
    output_names: Iterable[str],
):
    """Build an MLX graph-safe custom Metal kernel from TileLang MSL."""

    if not msl_source:
        return None
    try:
        return _cached_mlx_tilelang_metal_kernel(
            msl_source,
            tuple(input_names),
            tuple(output_names),
        )
    except MLXGraphInteropError:
        return None


def is_mlx_array(arg: Any) -> bool:
    """Return whether *arg* is an ``mlx.core.array`` without requiring MLX."""

    mx = _mlx_core()
    if mx is None:
        return False
    mx_array = getattr(mx, "array", None)
    if not isinstance(mx_array, type):
        return False
    return isinstance(arg, mx_array)


def _contains_mlx_array(arg: Any) -> bool:
    if is_mlx_array(arg):
        return True
    if isinstance(arg, Mapping):
        return any(_contains_mlx_array(value) for value in arg.values())
    if isinstance(arg, (list, tuple, set, frozenset)):
        return any(_contains_mlx_array(value) for value in arg)
    return False


def has_mlx_arrays(args: Iterable[Any]) -> bool:
    """Return whether a positional argument sequence contains any MLX array."""

    return any(_contains_mlx_array(arg) for arg in args)


def _format_dlpack_device(device_type: int, device_id: int) -> str:
    name = _DLPACK_DEVICE_NAMES.get(device_type, f"DLDeviceType({device_type})")
    return f"{name}:{device_id}"


def _is_py_capsule(arg: Any) -> bool:
    return type(arg).__name__ == "PyCapsule"


def _pycapsule_is_valid(arg: Any, name: bytes) -> bool:
    is_valid = ctypes.pythonapi.PyCapsule_IsValid
    is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    is_valid.restype = ctypes.c_int
    return bool(is_valid(arg, name))


def _pycapsule_get_pointer(arg: Any, name: bytes) -> int:
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    ptr = get_pointer(arg, name)
    if not ptr:
        raise DLPackOwnershipError("DLPack capsule has a null tensor pointer")
    return int(ptr)


def _get_dlpack_capsule_device(arg: Any) -> tuple[int, int]:
    if _pycapsule_is_valid(arg, b"dltensor"):
        ptr = _pycapsule_get_pointer(arg, b"dltensor")
        tensor = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor)).contents
        return int(tensor.dl_tensor.device.device_type), int(tensor.dl_tensor.device.device_id)
    if _pycapsule_is_valid(arg, b"used_dltensor") or _pycapsule_is_valid(
        arg, b"used_dltensor_versioned"
    ):
        raise DLPackOwnershipError("DLPack capsule has already been consumed")
    if _pycapsule_is_valid(arg, b"dltensor_versioned"):
        raise DLPackConversionError(
            "Versioned DLPack capsules cannot be preflighted for device ownership"
        )
    raise DLPackConversionError("DLPack capsule is not a valid dltensor capsule")


def _is_used_dlpack_capsule(arg: Any) -> bool:
    return _is_py_capsule(arg) and (
        _pycapsule_is_valid(arg, b"used_dltensor")
        or _pycapsule_is_valid(arg, b"used_dltensor_versioned")
    )


def _iter_nested_values(args: Iterable[Any]):
    for arg in args:
        if isinstance(arg, Mapping):
            yield from _iter_nested_values(arg.values())
        elif isinstance(arg, (list, tuple, set, frozenset)):
            yield from _iter_nested_values(arg)
        else:
            yield arg


def _get_dlpack_device(arg: Any) -> tuple[int, int]:
    get_device = getattr(arg, "__dlpack_device__", None)
    if get_device is None:
        if _is_py_capsule(arg):
            return _get_dlpack_capsule_device(arg)
        raise DLPackDeviceError(f"{type(arg).__name__} does not expose __dlpack_device__")
    try:
        device = get_device()
    except Exception as exc:
        raise DLPackDeviceError(
            f"{type(arg).__name__}.__dlpack_device__() failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(device, tuple) or len(device) != 2:
        raise DLPackDeviceError(
            f"{type(arg).__name__}.__dlpack_device__() must return (device_type, device_id), "
            f"got {device!r}"
        )
    try:
        return int(device[0]), int(device[1])
    except Exception as exc:
        raise DLPackDeviceError(
            f"{type(arg).__name__}.__dlpack_device__() returned non-integer values: {device!r}"
        ) from exc


def validate_dlpack_device(
    arg: Any,
    *,
    expected_device_type: int | None = None,
    expected_device_id: int | None = None,
    owner_name: str = "DLPack producer",
) -> tuple[int, int]:
    """Validate a DLPack producer's reported device before consuming it."""

    device_type, device_id = _get_dlpack_device(arg)
    if expected_device_type is not None and device_type != expected_device_type:
        raise DLPackDeviceError(
            f"{owner_name} is on {_format_dlpack_device(device_type, device_id)}, "
            f"but this path requires "
            f"{_format_dlpack_device(expected_device_type, expected_device_id or 0)}"
        )
    if expected_device_id is not None and device_id != expected_device_id:
        raise DLPackDeviceError(
            f"{owner_name} is on {_format_dlpack_device(device_type, device_id)}, "
            f"but this path requires {_format_dlpack_device(device_type, expected_device_id)}"
        )
    return device_type, device_id


def validate_dlpack_inputs_for_target(args: Iterable[Any], target_kind: str) -> None:
    """Fail early when DLPack inputs cannot be consumed by the target backend."""

    expected_device_type = None
    if target_kind == "metal":
        expected_device_type = DLPACK_DEVICE_METAL
    if expected_device_type is None:
        return

    for arg in _iter_nested_values(args):
        if hasattr(arg, "__dlpack_device__") or _is_py_capsule(arg):
            validate_dlpack_device(
                arg,
                expected_device_type=expected_device_type,
                owner_name=type(arg).__name__,
            )


def first_mlx_array_device(args: Iterable[Any]) -> tuple[int, int] | None:
    """Return the first MLX array's DLPack device, if any."""

    for arg in _iter_nested_values(args):
        if is_mlx_array(arg):
            return validate_dlpack_device(
                arg,
                expected_device_type=DLPACK_DEVICE_METAL,
                owner_name="MLX array",
            )
    return None


def dlpack_to_tvm_tensor(
    arg: Any,
    *,
    expected_device_type: int | None = None,
    expected_device_id: int | None = None,
    owner_name: str = "DLPack producer",
):
    """Import a DLPack producer as a TVM tensor view with typed failures."""

    if (
        hasattr(arg, "__dlpack_device__")
        or expected_device_type is not None
        or expected_device_id is not None
        or _is_used_dlpack_capsule(arg)
    ):
        validate_dlpack_device(
            arg,
            expected_device_type=expected_device_type,
            expected_device_id=expected_device_id,
            owner_name=owner_name,
        )
    try:
        return tvm.runtime.from_dlpack(arg)
    except ValueError as exc:
        msg = str(exc)
        if "consume" in msg or "used_dltensor" in msg or "used_dltensor_versioned" in msg:
            raise DLPackOwnershipError(
                f"{owner_name} ownership transfer failed: {msg}"
            ) from exc
        raise DLPackConversionError(f"{owner_name} import failed: {msg}") from exc
    except BufferError as exc:
        raise DLPackOwnershipError(
            f"{owner_name} ownership transfer failed: {exc}"
        ) from exc
    except TypeError as exc:
        msg = str(exc)
        if "PyCapsule" in msg and (
            "used_dltensor" in repr(arg) or "used_dltensor_versioned" in repr(arg)
        ):
            raise DLPackOwnershipError(
                f"{owner_name} ownership transfer failed: {msg}"
            ) from exc
        raise DLPackConversionError(
            f"{owner_name} import failed: {type(exc).__name__}: {exc}"
        ) from exc
    except RuntimeError as exc:
        raise DLPackConversionError(
            f"{owner_name} import failed: {type(exc).__name__}: {exc}"
        ) from exc


def _dtype_name(dtype: Any | None) -> str | None:
    if dtype is None:
        return None
    dtype_name = str(dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.removeprefix("torch.")
    return dtype_name


def _view_tvm_tensor_for_expected_dtype(tensor: Any, arg: Any, expected_dtype: Any | None):
    expected = _dtype_name(expected_dtype)
    if expected is None or str(tensor.dtype) == expected:
        return tensor
    if expected.startswith("float8") and str(tensor.dtype) in {"uint8", "int8"}:
        try:
            return tensor._create_view(tuple(int(dim) for dim in arg.shape), dtype=expected)
        except Exception as exc:
            raise DLPackConversionError(
                f"MLX array import failed: could not view {tensor.dtype} DLPack storage "
                f"as expected {expected} without copying: {type(exc).__name__}: {exc}"
            ) from exc
    raise DLPackConversionError(
        f"MLX array import failed: DLPack dtype {tensor.dtype} does not match "
        f"expected kernel dtype {expected}"
    )


def mlx_array_to_tvm_tensor(arg: Any, *, expected_dtype: Any | None = None):
    """Import an MLX Metal array as a TVM tensor view without copying."""

    if not is_mlx_array(arg):
        raise TypeError(f"expected mlx.core.array, got {type(arg).__name__}")
    tensor = dlpack_to_tvm_tensor(
        arg,
        expected_device_type=DLPACK_DEVICE_METAL,
        owner_name="MLX array",
    )
    return _view_tvm_tensor_for_expected_dtype(tensor, arg, expected_dtype)


def mlx_arrays_to_tvm_tensors(
    args: Iterable[Any],
    *,
    expected_dtypes: Iterable[Any | None] | None = None,
) -> list[Any]:
    """Convert MLX array arguments to TVM Tensor views with DLPack.

    DLPack imports borrow the existing producer allocation; this does not
    allocate or copy tensor payloads.  MLX stores FP8 arrays as uint8 DLPack
    buffers; when a TileLang ABI expects float8, re-view the TVM tensor with
    the expected dtype so the runtime sees the correct ABI without staging.
    """

    arg_list = list(args)
    if expected_dtypes is None:
        expected_list: list[Any | None] = [None] * len(arg_list)
    else:
        expected_list = list(expected_dtypes)
        if len(expected_list) != len(arg_list):
            raise DLPackConversionError(
                f"expected_dtypes length {len(expected_list)} does not match "
                f"argument length {len(arg_list)}"
            )
    converted = []
    for arg, expected_dtype in zip(arg_list, expected_list, strict=True):
        if is_mlx_array(arg):
            converted.append(mlx_array_to_tvm_tensor(arg, expected_dtype=expected_dtype))
        else:
            converted.append(arg)
    return converted


def mlx_dtype_from_tvm(dtype: Any):
    """Map a TVM/TileLang dtype to an MLX dtype without casting data."""

    mx = _mlx_core()
    if mx is None:
        raise DLPackConversionError("mlx.core is required to allocate MLX Metal outputs")

    dtype_name = str(dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.removeprefix("torch.")
    dtype_name = {
        "bool": "bool_",
        "uint1": "bool_",
    }.get(dtype_name, dtype_name)
    mlx_dtype = getattr(mx, dtype_name, None)
    if mlx_dtype is None:
        raise DLPackConversionError(f"MLX output allocation does not support dtype {dtype!s}")
    return mlx_dtype


def mlx_metal_output(
    shape: Iterable[int],
    dtype: Any,
    *,
    policy: str = MLX_OUTPUT_WRITE_ONLY,
):
    """Allocate an MLX Metal output buffer for TVM to fill through DLPack.

    ``write_only`` is the default because ``out_idx`` buffers are kernel
    results. They must have storage, but they do not need zero-fill work before
    TVM overwrites them. Older MLX builds without ``mx.empty`` fall back to
    ``mx.zeros`` so the ABI remains functional.
    """

    mx = _mlx_core()
    if mx is None:
        raise DLPackConversionError("mlx.core is required to allocate MLX Metal outputs")
    shape_tuple = tuple(int(dim) for dim in shape)
    mlx_dtype = mlx_dtype_from_tvm(dtype)
    if policy == MLX_OUTPUT_WRITE_ONLY:
        empty = getattr(mx, "empty", None)
        if empty is not None:
            return empty(shape_tuple, dtype=mlx_dtype)
        return mx.zeros(shape_tuple, dtype=mlx_dtype)
    if policy == MLX_OUTPUT_ZEROED:
        return mx.zeros(shape_tuple, dtype=mlx_dtype)
    raise DLPackConversionError(f"unknown MLX output allocation policy: {policy!r}")


def tvm_tensor_to_mlx_array(tensor: Any):
    """Export a TVM Metal tensor to MLX via DLPack without copying."""

    mx = _mlx_core()
    if mx is None:
        raise DLPackConversionError("mlx.core is required to export TVM Metal tensors to MLX")
    validate_dlpack_device(
        tensor,
        expected_device_type=DLPACK_DEVICE_METAL,
        owner_name="TVM tensor",
    )
    try:
        return mx.array(tensor.__dlpack__())
    except ValueError as exc:
        msg = str(exc)
        if "consume" in msg or "used_dltensor" in msg or "used_dltensor_versioned" in msg:
            raise DLPackOwnershipError(f"TVM tensor export failed: {msg}") from exc
        raise DLPackConversionError(f"TVM tensor export failed: {msg}") from exc
    except (BufferError, TypeError, RuntimeError) as exc:
        raise DLPackConversionError(
            f"TVM tensor export failed: {type(exc).__name__}: {exc}"
        ) from exc


def _metal_func(name: str):
    return tvm.ffi.get_global_func(name, allow_missing=True)


@contextmanager
def mlx_metal_external_command_buffer(stream: Any | None = None):
    """Run TVM Metal work on MLX's current command buffer.

    The command buffer is borrowed from MLX.  TVM may encode into it but must
    not retain, release, commit, or synchronize it.
    """

    mx = _mlx_core()
    if (
        mx is None
        or not hasattr(mx, "metal")
        or not hasattr(mx.metal, "_current_command_buffer")
    ):
        yield
        return

    get_external = _metal_func("metal.GetExternalCommandBuffer")
    set_external = _metal_func("metal.SetExternalCommandBuffer")
    clear_external = _metal_func("metal.ClearExternalCommandBuffer")
    if get_external is None or set_external is None or clear_external is None:
        yield
        return

    ptr = (
        mx.metal._current_command_buffer()
        if stream is None
        else mx.metal._current_command_buffer(stream)
    )
    if not ptr:
        yield
        return

    previous = get_external()
    previous_ptr = getattr(previous, "value", previous)
    clear_external()
    try:
        set_external(ctypes.c_void_p(ptr))
        yield
    finally:
        clear_external()
        if previous_ptr:
            set_external(ctypes.c_void_p(previous_ptr))


def maybe_mlx_metal_external_command_buffer(args: Iterable[Any], stream: Any | None = None):
    """Return an MLX Metal command-buffer context only when MLX arrays are present."""

    if has_mlx_arrays(args):
        return mlx_metal_external_command_buffer(stream)
    return nullcontext()


def mlx_external_command_buffer_available(stream: Any | None = None) -> bool:
    """Return True when TVM Metal work can be encoded onto MLX's command buffer.

    The external-command-buffer route borrows MLX's in-flight command buffer via
    ``mx.metal._current_command_buffer`` so that ``mx.eval``/``mx.synchronize``
    commits the TVM-encoded work together with MLX's own work. When this API is
    missing (some MLX builds) or returns a null pointer, TVM falls back to its
    own internal Metal command buffer, which MLX never commits/syncs -- so the
    caller must synchronize the TVM Metal stream itself before reading results.
    """

    mx = _mlx_core()
    if (
        mx is None
        or not hasattr(mx, "metal")
        or not hasattr(mx.metal, "_current_command_buffer")
    ):
        return False
    if (
        _metal_func("metal.GetExternalCommandBuffer") is None
        or _metal_func("metal.SetExternalCommandBuffer") is None
        or _metal_func("metal.ClearExternalCommandBuffer") is None
    ):
        return False
    try:
        ptr = (
            mx.metal._current_command_buffer()
            if stream is None
            else mx.metal._current_command_buffer(stream)
        )
    except Exception:
        return False
    return bool(ptr)


def sync_tvm_metal_internal_command_buffer(args: Iterable[Any]) -> None:
    """Commit+wait TVM's internal Metal command buffer for MLX-owned outputs.

    When :func:`mlx_external_command_buffer_available` is False, TVM encodes the
    kernel onto its own (per-thread) Metal command buffer instead of MLX's
    in-flight buffer. That buffer is never committed or synchronized by MLX, so
    an MLX output array aliasing the same ``MTLBuffer`` reads stale/zeroed
    memory. Synchronizing the TVM Metal device commits and waits that buffer so
    the writes are visible before the result is read.
    """

    if mlx_external_command_buffer_available():
        return
    if not has_mlx_arrays(args):
        return
    device_ids: set[int] = set()
    for value in _iter_nested_values(args):
        if is_mlx_array(value):
            try:
                device_type, device_id = _get_dlpack_device(value)
            except Exception:
                continue
            if device_type == DLPACK_DEVICE_METAL:
                device_ids.add(int(device_id))
    if not device_ids:
        device_ids.add(0)
    for device_id in device_ids:
        try:
            tvm.metal(device_id).sync()
        except Exception:
            pass
