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
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

from tilelang import tvm


def _mlx_core():
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception:
        return None
    return mx


def is_mlx_array(arg: Any) -> bool:
    """Return whether *arg* is an ``mlx.core.array`` without requiring MLX."""

    mx = _mlx_core()
    return mx is not None and isinstance(arg, mx.array)


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


def mlx_arrays_to_tvm_tensors(args: Iterable[Any]) -> list[Any]:
    """Convert MLX array arguments to TVM Tensor views with DLPack.

    DLPack imports borrow the existing producer allocation; this does not
    allocate or copy tensor payloads.
    """

    converted = []
    for arg in args:
        if is_mlx_array(arg):
            converted.append(tvm.runtime.from_dlpack(arg))
        else:
            converted.append(arg)
    return converted


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
