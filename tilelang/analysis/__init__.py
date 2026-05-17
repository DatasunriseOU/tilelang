"""Tilelang IR analysis & visitors."""

from .ast_printer import ASTPrinter  # noqa: F401
from .nested_loop_checker import NestedLoopChecker  # noqa: F401
from .fragment_loop_checker import FragmentLoopChecker  # noqa: F401
from .layout_visual import LayoutVisual  # noqa: F401
from .reduction_plan import (  # noqa: F401
    ReductionAxisPlan,
    ReductionAliasConstraints,
    ReductionMemoryPlan,
    ReductionPlan,
    ReductionPlanError,
    ReductionThreadMapping,
    BufferRegion,
    attach_reduction_plan_metadata,
    candidate_strategies_for_extent,
    extract_reduction_plans,
    selected_strategy_for_extent,
)
from .reduction_legality import (  # noqa: F401
    ReductionLegalityProof,
    attach_reduction_legality_metadata,
    prove_reduction_plan_legality,
    prove_reduction_plans,
)
from .sync_event_plan import (  # noqa: F401
    SyncEventDecision,
    attach_sync_event_plan_metadata,
    build_reduction_sync_event_plan,
)
from .backend_lowerer_selection import (  # noqa: F401
    ReductionBackendLowererDiagnostic,
    attach_reduction_backend_lowerer_metadata,
    build_reduction_backend_lowerer_diagnostics,
)
from .autotune_plan import (  # noqa: F401
    ScheduleAbiFingerprint,
    ScheduleCandidate,
    ScheduleTiming,
    WarmScheduleSelection,
    schedule_candidate_key,
    schedule_selection_key,
    select_warm_schedule,
    serialize_warm_schedule_selection,
)
from .cost_model import (  # noqa: F401
    RecurrenceScanCostEstimate,
    ReductionCostEstimate,
    attach_reduction_cost_metadata,
    build_reduction_cost_estimates,
    estimate_recurrence_scan_cost,
    estimate_reduction_cost,
)
from .scan_plan import (  # noqa: F401
    RecurrenceAliasPlan,
    RecurrenceScanPlan,
    RecurrenceSnapshotPlan,
    plan_recurrence_scan,
    serialize_recurrence_scan_plans,
)
