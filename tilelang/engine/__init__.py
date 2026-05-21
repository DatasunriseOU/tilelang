from .lower import lower, is_device_call  # noqa: F401
from .param import KernelParam  # noqa: F401
from .fusion import (  # noqa: F401
    BaselineComparison,
    FusionBlockDescriptor,
    FusionBlockRegistry,
    FusionAutogradPlan,
    FusionCacheKeyAudit,
    FusionCompilePlan,
    FusionCompileResult,
    FusionEdge,
    FusionNode,
    FusionOptimizer,
    FusionRegion,
    FusionRegionBuilder,
    FusionScheduleEntry,
    FusionScheduleRegistry,
    WarmCacheAudit,
    audit_fusion_cache_key,
    audit_warm_cache_reuse,
    build_fusion_region,
    build_fusion_region_from_blocks,
    build_fusion_regions_from_blocks,
    build_mamba3_fp8_train_block_region,
    compile_fusion_region,
    fusion_cache_key_digest,
    fusion_default_allowed,
    path_b_baseline_clean,
    plan_fusion_region,
)
from .callback import (
    register_cuda_postproc,  # noqa: F401
    register_hip_postproc,  # noqa: F401
    register_c_postproc,  # noqa: F401
)
