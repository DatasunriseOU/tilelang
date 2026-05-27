"""Metal backend lowering pipeline.

Task 12 / #2189 introduced the per-backend ``Pipeline`` dispatch. Our fork's
Metal pipeline is significantly richer than the generic upstream template
(MetalSimdLiftReductions, MetalFragmentToSimdgroup, MetalSimdgroupSemanticGuard,
MetalMergeRoundBarrierCleanup, BindMetalScalarIntrinsics, the HoistExpression
loop, etc.) and is implemented as ``LowerAndLegalize`` + ``OptimizeForTarget``
in :mod:`tilelang.engine.phase`. Rather than re-implement that here, we slot
into the backend-aware dispatch shape by registering a Pipeline that delegates
to those existing functions inside the required ``with target`` context.
"""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from tilelang.backend.pipeline import Pipeline, register_pipeline


def MetalPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    # Import locally to avoid a circular import: tilelang.engine.phase imports
    # tilelang.transform, which in turn pulls tilelang.backend at module load.
    from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget

    # CPPMEGA / fork: apache/tvm latest requires an ambient Target context for
    # passes that invoke the arith analyzer (e.g. LayoutInference ->
    # const_int_bound calls Target::Current() with allow_not_defined=false).
    with target:
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target)
    return mod


metal_pipeline = Pipeline("metal", MetalPassPipelineBody)
register_pipeline(metal_pipeline)
