from __future__ import annotations
import json
import re
from functools import wraps
from typing import Callable

import torch
from tvm import tir

from tilelang import tvm as tvm

from ..base import BaseKernelAdapter
from tilelang.engine.param import KernelParam


class MetalKernelAdapter(BaseKernelAdapter):
    _launch_info_prefix = "// tilelang_metal_launch_info: "

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        #  target: Union[str, Target],
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        #  host_mod: Optional[tvm.IRModule] = None,
        device_mod: tvm.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
        #  pass_configs: Optional[Dict[str, Any]] = None,
        #  compile_flags: Optional[List[str]] = None
    ):
        self.kernel_global_source = kernel_global_source
        self.kernel_name = self._get_kernel_name(func_or_mod, kernel_global_source)
        self.verbose = verbose

        self.block_info, self.grid_info = self._extract_launch_info(device_mod, self.kernel_name)
        self.arg_order = self._extract_arg_order(func_or_mod, device_mod, self.kernel_name, kernel_global_source)

        # print(self.block_info, self.grid_info)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _kernel = None

    @classmethod
    def from_database(
        cls,
        params: list[KernelParam],
        result_idx: list[int],
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        device_kernel_source: str,
        verbose: bool = False,
    ) -> "MetalKernelAdapter":
        adapter = cls.__new__(cls)
        adapter.kernel_global_source = device_kernel_source
        adapter.kernel_name = cls._get_kernel_name(func_or_mod, device_kernel_source)
        adapter.verbose = verbose
        adapter._kernel = None
        launch_info = cls._parse_launch_info(device_kernel_source)
        adapter.block_info = list(launch_info["block_info"])
        adapter.grid_info = list(launch_info["grid_info"])
        adapter.arg_order = launch_info.get("arg_order")
        if adapter.arg_order is None:
            adapter.arg_order = cls._extract_arg_order(func_or_mod, None, adapter.kernel_name, device_kernel_source)
        BaseKernelAdapter.__init__(adapter, func_or_mod, params=params, result_idx=result_idx)
        return adapter

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.kernel_global_source or ""

    def get_kernel_source_with_launch_info(self) -> str:
        source = self.kernel_global_source or ""
        launch_info = {
            "kernel_name": self.kernel_name,
            "block_info": [self._as_int_extent(extent) for extent in self.block_info],
            "grid_info": [self._as_int_extent(extent) for extent in self.grid_info],
            "arg_order": list(self.arg_order),
        }
        return self._strip_launch_info(source) + "\n" + self._launch_info_prefix + json.dumps(launch_info, sort_keys=True) + "\n"

    @classmethod
    def _get_kernel_name(cls, func_or_mod: tir.PrimFunc | tvm.IRModule, kernel_source: str | None = None) -> str:
        launch_info = cls._try_parse_launch_info(kernel_source)
        if launch_info is not None and launch_info.get("kernel_name"):
            return str(launch_info["kernel_name"])

        func_name = getattr(func_or_mod, "__name__", None)
        if func_name:
            return str(func_name) + "_kernel"

        match = re.search(r"\bkernel\s+void\s+([A-Za-z_]\w*)\s*\(", kernel_source or "")
        if match:
            return match.group(1)

        if isinstance(func_or_mod, tir.PrimFunc):
            func_name = func_or_mod.attrs.get("global_symbol")
            if func_name is not None:
                return str(func_name) + "_kernel"

        raise ValueError("Cannot determine Metal kernel name from cached source")

    @classmethod
    def _extract_launch_info(cls, device_mod: tvm.IRModule | None, kernel_name: str) -> tuple[list[int], list[int]]:
        block_info = [1, 1, 1]
        grid_info = [1, 1, 1]

        if device_mod is None:
            return block_info, grid_info

        for var, func in device_mod.functions.items():
            if var.name_hint != kernel_name:
                continue
            thread_extent = func.attrs["thread_extent"]
            for tag, extent in thread_extent.items():
                if "threadIdx" in tag:
                    block_info["xyz".index(tag[-1])] = cls._as_int_extent(extent)
                elif "blockIdx" in tag:
                    grid_info["xyz".index(tag[-1])] = cls._as_int_extent(extent)
            return block_info, grid_info

        raise AssertionError(f"no kernel with name {kernel_name}")

    @classmethod
    def _parse_launch_info(cls, kernel_source: str | None) -> dict:
        launch_info = cls._try_parse_launch_info(kernel_source)
        if launch_info is not None:
            return launch_info
        raise ValueError("Cached Metal kernel source is missing TileLang launch info")

    @classmethod
    def _try_parse_launch_info(cls, kernel_source: str | None) -> dict | None:
        for line in (kernel_source or "").splitlines():
            line = line.strip()
            if not line.startswith(cls._launch_info_prefix):
                continue
            return json.loads(line[len(cls._launch_info_prefix):])
        return None

    @classmethod
    def _strip_launch_info(cls, kernel_source: str) -> str:
        return "\n".join(line for line in kernel_source.splitlines() if not line.strip().startswith(cls._launch_info_prefix)).rstrip()

    @classmethod
    def _extract_arg_order(
        cls,
        func_or_mod: tir.PrimFunc | tvm.IRModule,
        device_mod: tvm.IRModule | None,
        kernel_name: str,
        kernel_source: str | None,
    ) -> list[int]:
        source_param_aliases = cls._source_param_aliases(func_or_mod)
        if not source_param_aliases:
            return []

        device_param_names = cls._device_param_names(device_mod, kernel_name)
        if not device_param_names:
            device_param_names = cls._parse_msl_buffer_param_names(kernel_source, kernel_name)
        if not device_param_names:
            return list(range(len(source_param_aliases)))

        source_index_by_name: dict[str, int] = {}
        for i, aliases in enumerate(source_param_aliases):
            for alias in aliases:
                source_index_by_name.setdefault(alias, i)

        try:
            arg_order = [source_index_by_name[name] for name in device_param_names]
        except KeyError:
            return list(range(len(source_param_aliases)))

        if sorted(arg_order) != list(range(len(source_param_aliases))):
            return list(range(len(source_param_aliases)))
        return arg_order

    @classmethod
    def _source_param_aliases(cls, func_or_mod: tir.PrimFunc | tvm.IRModule) -> list[set[str]]:
        if not isinstance(func_or_mod, tir.PrimFunc):
            return []

        param_aliases: list[set[str]] = []
        for var in func_or_mod.params:
            aliases = {cls._tir_name(var)}
            if var in func_or_mod.buffer_map:
                buffer = func_or_mod.buffer_map[var]
                aliases.add(str(buffer.name))
                aliases.add(cls._tir_name(buffer.data))
            param_aliases.append(aliases)
        return param_aliases

    @classmethod
    def _device_param_names(cls, device_mod: tvm.IRModule | None, kernel_name: str) -> list[str]:
        if device_mod is None:
            return []

        for var, func in device_mod.functions.items():
            if var.name_hint == kernel_name:
                return [cls._tir_name(param) for param in func.params]
        return []

    @staticmethod
    def _tir_name(var) -> str:
        return str(getattr(var, "name_hint", getattr(var, "name", var)))

    @classmethod
    def _parse_msl_buffer_param_names(cls, kernel_source: str | None, kernel_name: str) -> list[str]:
        if not kernel_source:
            return []

        signature_match = re.search(
            rf"\bkernel\s+void\s+{re.escape(kernel_name)}\s*\((.*?)\)\s*\{{",
            cls._strip_launch_info(kernel_source),
            re.DOTALL,
        )
        if signature_match is None:
            return []

        buffer_params: list[tuple[int, str]] = []
        for line in signature_match.group(1).splitlines():
            if "_args_t" in line:
                continue
            match = re.search(r"\b([A-Za-z_]\w*)\s*\[\[\s*buffer\((\d+)\)\s*\]\]", line)
            if match:
                buffer_params.append((int(match.group(2)), match.group(1)))
        return [name for _, name in sorted(buffer_params)]

    @staticmethod
    def _as_int_extent(extent):
        if hasattr(extent, "value"):
            extent = extent.value
        return int(extent)

    def _convert_torch_func(self) -> Callable:
        if self._kernel is None:
            _kernel = getattr(torch.mps.compile_shader(self.kernel_global_source), self.kernel_name)
            _threads = [x * y for (x, y) in zip(self.block_info, self.grid_info)]

            @wraps(_kernel)
            def launcher(*args: torch.Tensor):
                if self.arg_order and self.arg_order != list(range(len(args))) and len(self.arg_order) == len(args):
                    args = tuple(args[i] for i in self.arg_order)
                return _kernel(
                    *args,
                    threads=_threads,
                    group_size=self.block_info,
                )

            self._kernel = launcher

        return self._kernel
