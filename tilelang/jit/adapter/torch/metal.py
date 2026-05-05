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
        adapter.block_info, adapter.grid_info = cls._parse_launch_info(device_kernel_source)
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
        }
        return self._strip_launch_info(source) + "\n" + self._launch_info_prefix + json.dumps(launch_info, sort_keys=True) + "\n"

    @classmethod
    def _get_kernel_name(cls, func_or_mod: tir.PrimFunc | tvm.IRModule, kernel_source: str | None = None) -> str:
        if isinstance(func_or_mod, tir.PrimFunc):
            func_name = func_or_mod.attrs.get("global_symbol")
            if func_name is not None:
                return str(func_name) + "_kernel"

        func_name = getattr(func_or_mod, "__name__", None)
        if func_name:
            return str(func_name) + "_kernel"

        match = re.search(r"\bkernel\s+void\s+([A-Za-z_]\w*)\s*\(", kernel_source or "")
        if match:
            return match.group(1)

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
    def _parse_launch_info(cls, kernel_source: str | None) -> tuple[list[int], list[int]]:
        for line in (kernel_source or "").splitlines():
            line = line.strip()
            if not line.startswith(cls._launch_info_prefix):
                continue
            launch_info = json.loads(line[len(cls._launch_info_prefix):])
            return list(launch_info["block_info"]), list(launch_info["grid_info"])
        raise ValueError("Cached Metal kernel source is missing TileLang launch info")

    @classmethod
    def _strip_launch_info(cls, kernel_source: str) -> str:
        return "\n".join(line for line in kernel_source.splitlines() if not line.strip().startswith(cls._launch_info_prefix)).rstrip()

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
                return _kernel(
                    *args,
                    threads=_threads,
                    group_size=self.block_info,
                )

            self._kernel = launcher

        return self._kernel
