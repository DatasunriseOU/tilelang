from .reduction import (  # noqa: F401
    ReductionLowererEntry,
    ReductionLowererSelection,
    reduction_lowerer_cache_info,
    registered_reduction_lowerers,
    register_reduction_lowerer,
    resolve_reduction_lowerer,
    select_reduction_lowerer,
)

# Import built-in backend packages so their reduction lowerers, GEMM lowerers,
# and (Task 12 / #2189) backend-aware compilation pipelines register.
# GEMM/GEMM-SP registries moved to ``tilelang.tileop.{gemm,gemm_sp}`` and the
# CUDA/CPU/ROCm op packages under ``tilelang.{cpu,cuda,rocm}`` (#2165). Each
# backend package now also registers a Pipeline via tilelang.backend.pipeline.
from . import common as _common  # noqa: F401,E402
from . import cpu as _cpu  # noqa: F401,E402
from . import cuda as _cuda  # noqa: F401,E402
from . import metal as _metal  # noqa: F401,E402
from . import rocm as _rocm  # noqa: F401,E402
