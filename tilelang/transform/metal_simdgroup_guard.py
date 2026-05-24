"""Fail-closed legality checks for Metal SIMDgroup warp intrinsics."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from tvm import IRModule, tir
from tvm.target import Target
from tvm.tir.transform import prim_func_pass

from . import _ffi_api

_FULL_WARP_MASK = 0xFFFFFFFF
_METAL_SIMDGROUP_WIDTH = 32
_GEMM_WARP_POLICY_NAMES = {
    1: "GemmWarpPolicy.FullRow",
    2: "GemmWarpPolicy.FullCol",
    3: "GemmWarpPolicy.Free",
}

_UNSUPPORTED_METAL_INTRINSICS = {
    "tl.shfl_sync": "absolute-lane CUDA warp broadcast",
    "tl.shfl_down_sync": "CUDA warp down-shuffle",
    "tl.shfl_up_sync": "CUDA warp up-shuffle",
}


def _target_kind_name(target: Any) -> str | None:
    if target is None:
        return None
    kind = getattr(getattr(target, "kind", None), "name", None)
    if kind is not None:
        return str(kind).lower()
    with suppress(Exception):
        target_object = Target(target)
        kind = getattr(getattr(target_object, "kind", None), "name", None)
        if kind is not None:
            return str(kind).lower()
    text = str(target).lower()
    if "metal" in text:
        return "metal"
    return text


def _func_target(func: tir.PrimFunc) -> Any:
    attrs = getattr(func, "attrs", None)
    if attrs is not None:
        with suppress(AttributeError, KeyError, TypeError):
            target = attrs.get("target", None)
            if target is not None:
                return target
        with suppress(KeyError, TypeError):
            return attrs["target"]
    return Target.current(allow_none=True)


def _op_name(call: tir.Call) -> str:
    name = getattr(call.op, "name", None)
    if name is not None:
        return str(name)
    return str(call.op)


def _const_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, tir.IntImm):
        return int(value.value)
    return None


def _validate_shfl_xor(call: tir.Call, func_name: str) -> None:
    if len(call.args) != 4:
        raise ValueError(
            "Metal SIMDgroup guard rejected malformed tl.shfl_xor_sync in "
            f"{func_name}: expected <mask, value, lane_mask, width>."
        )
    mask = _const_int(call.args[0])
    width = _const_int(call.args[3])
    if mask == _FULL_WARP_MASK and width == _METAL_SIMDGROUP_WIDTH:
        return
    raise ValueError(
        "Metal SIMDgroup guard rejected tl.shfl_xor_sync in "
        f"{func_name}: Metal lowering only preserves full-simdgroup semantics "
        f"(mask=0xFFFFFFFF, width=32), got mask={call.args[0]} width={call.args[3]}."
    )


def _validate_tile_gemm(call: tir.Call, func_name: str) -> None:
    if len(call.args) <= 8:
        return
    policy = _const_int(call.args[8])
    policy_name = _GEMM_WARP_POLICY_NAMES.get(policy)
    if policy_name is None:
        return
    raise ValueError(
        "Metal SIMDgroup guard rejected tl.tileop.gemm in "
        f"{func_name}: {policy_name} encodes a CUDA/HIP warp-partition policy. "
        "Metal schedules must use target-native simdgroup GEMM lowering or an "
        "explicit Metal policy instead of reusing a warp-policy tile GEMM."
    )


def validate_metal_simdgroup_intrinsics(
    func_or_mod: tir.PrimFunc | IRModule,
    *,
    target: Any = None,
) -> None:
    funcs: list[tuple[str, tir.PrimFunc]]
    if isinstance(func_or_mod, tir.PrimFunc):
        funcs = [("main", func_or_mod)]
    else:
        funcs = [
            (global_var.name_hint, func)
            for global_var, func in func_or_mod.functions.items()
            if isinstance(func, tir.PrimFunc)
        ]

    for func_name, func in funcs:
        effective_target = target if target is not None else _func_target(func)
        if _target_kind_name(effective_target) != "metal":
            continue

        def _visit(node: Any, *, current_func_name: str = func_name) -> None:
            if not isinstance(node, tir.Call):
                return
            op_name = _op_name(node)
            if op_name in _UNSUPPORTED_METAL_INTRINSICS:
                raise ValueError(
                    "Metal SIMDgroup guard rejected "
                    f"{op_name} in {current_func_name}: {_UNSUPPORTED_METAL_INTRINSICS[op_name]} "
                    "does not have a verified Metal SIMDgroup equivalent. Use a "
                    "Metal-specific simdgroup primitive or route this schedule to CUDA/HIP."
                )
            if op_name == "tl.shfl_xor_sync":
                _validate_shfl_xor(node, current_func_name)
            if op_name == "tl.tileop.gemm":
                _validate_tile_gemm(node, current_func_name)

        tir.stmt_functor.post_order_visit(func.body, _visit)


def _metal_simdgroup_guard(func: tir.PrimFunc, mod: IRModule, ctx) -> tir.PrimFunc:
    validate_metal_simdgroup_intrinsics(func)
    return func


_PythonMetalSimdgroupSemanticGuard = prim_func_pass(
    _metal_simdgroup_guard,
    opt_level=0,
    name="tl.MetalSimdgroupSemanticGuard",
)


def MetalSimdgroupSemanticGuard(mod: IRModule | None = None):
    """Return or apply the compiled Metal SIMDgroup semantic guard pass.

    The C++ FFI pass is the production path. The Python pass remains as a
    fallback for editable installs where the extension has not been rebuilt yet.
    """

    try:
        pass_obj = _ffi_api.MetalSimdgroupSemanticGuard()
    except AttributeError:
        pass_obj = _PythonMetalSimdgroupSemanticGuard
    if mod is None:
        return pass_obj
    return pass_obj(mod)
