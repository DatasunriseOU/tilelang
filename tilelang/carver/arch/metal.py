from __future__ import annotations
import tvm
from tvm.target import Target
from .arch_base import TileDevice


def is_metal_arch(arch: TileDevice) -> bool:
    return isinstance(arch, METAL)


class METAL(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = Target(target)
        self.target = target
        device = tvm.runtime.metal(0)
        if not device.exist:
            raise RuntimeError("Cannot find metal device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "METAL"
        self.smem_cap: int = 32 * 1024
        self.compute_max_core: int = 1
        self.warp_size: int = int(device.warp_size or 32)
        self.compute_capability: str = "metal"
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = 0
        self.transaction_size: list[int] = [32, 128]
        self.bandwidth: list[int] = [750, 1200]


__all__ = [
    "is_metal_arch",
    "METAL",
]
