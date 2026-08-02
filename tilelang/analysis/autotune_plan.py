"""Legal-schedule autotune memoization metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Callable
from collections.abc import Iterable


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _normalize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_config(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(item) for item in value]
    return value


@dataclass(frozen=True)
class ScheduleAbiFingerprint:
    """ABI components that invalidate warm schedule reuse."""

    tilelang: str
    tvm: str
    tvm_ffi: str
    mlx: str

    def to_json(self) -> dict[str, str]:
        return {
            "tilelang": self.tilelang,
            "tvm": self.tvm,
            "tvm_ffi": self.tvm_ffi,
            "mlx": self.mlx,
        }


@dataclass(frozen=True)
class ScheduleCandidate:
    """One legal or rejected schedule candidate for an op signature."""

    schedule_id: str
    op_signature: str
    shape: tuple[int, ...]
    dtype: str
    target_kind: str
    config: dict[str, Any]
    legal: bool
    proof_hash: str
    codegen_hash: str
    estimated_cost: float = 0.0
    rejection_reason: str = ""

    def config_hash(self) -> str:
        return _stable_hash(_normalize_config(self.config))

    def to_json(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "op_signature": self.op_signature,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "target_kind": self.target_kind,
            "config": _normalize_config(self.config),
            "legal": self.legal,
            "proof_hash": self.proof_hash,
            "codegen_hash": self.codegen_hash,
            "estimated_cost": self.estimated_cost,
            "rejection_reason": self.rejection_reason,
            "config_hash": self.config_hash(),
        }


@dataclass(frozen=True)
class ScheduleTiming:
    """Separate cold compile and warm execution timing."""

    cold_compile_ms: float
    warm_run_ms: float

    def to_json(self) -> dict[str, float]:
        return {
            "cold_compile_ms": self.cold_compile_ms,
            "warm_run_ms": self.warm_run_ms,
        }


@dataclass(frozen=True)
class WarmScheduleSelection:
    """Selected schedule and cache receipt."""

    cache_key: str
    selected_schedule_key: str
    selected_schedule_id: str
    selected_config: dict[str, Any]
    cache_hit: bool
    cold_compile_ms: float
    warm_run_ms: float
    proof_hash: str
    codegen_hash: str
    profiled_candidate_count: int
    skipped_illegal_candidate_count: int

    def with_cache_hit(self) -> WarmScheduleSelection:
        return replace(self, cache_hit=True, profiled_candidate_count=0)

    def to_json(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "selected_schedule_key": self.selected_schedule_key,
            "selected_schedule_id": self.selected_schedule_id,
            "selected_config": _normalize_config(self.selected_config),
            "cache_hit": self.cache_hit,
            "cold_compile_ms": self.cold_compile_ms,
            "warm_run_ms": self.warm_run_ms,
            "proof_hash": self.proof_hash,
            "codegen_hash": self.codegen_hash,
            "profiled_candidate_count": self.profiled_candidate_count,
            "skipped_illegal_candidate_count": self.skipped_illegal_candidate_count,
        }


def schedule_candidate_key(
    candidate: ScheduleCandidate,
    abi: ScheduleAbiFingerprint,
) -> str:
    """Stable cache key for one concrete schedule candidate."""

    return _stable_hash(
        {
            "abi": abi.to_json(),
            "op_signature": candidate.op_signature,
            "shape": list(candidate.shape),
            "dtype": candidate.dtype,
            "target_kind": candidate.target_kind,
            "proof_hash": candidate.proof_hash,
            "codegen_hash": candidate.codegen_hash,
            "config_hash": candidate.config_hash(),
        }
    )


def schedule_selection_key(
    candidates: Iterable[ScheduleCandidate],
    abi: ScheduleAbiFingerprint,
) -> str:
    """Stable cache key for an ordered legal schedule search space."""

    legal_keys = [schedule_candidate_key(candidate, abi) for candidate in candidates if candidate.legal]
    return _stable_hash({"legal_candidate_keys": legal_keys})


def select_warm_schedule(
    candidates: tuple[ScheduleCandidate, ...],
    *,
    abi: ScheduleAbiFingerprint,
    profile_candidate: Callable[[ScheduleCandidate], ScheduleTiming],
    cache: dict[str, WarmScheduleSelection] | None = None,
) -> WarmScheduleSelection:
    """Profile only legal candidates and memoize the selected warm schedule."""

    legal_candidates = tuple(candidate for candidate in candidates if candidate.legal)
    if not legal_candidates:
        raise ValueError("No legal schedule candidates to autotune")

    selection_key = schedule_selection_key(legal_candidates, abi)
    memo = cache if cache is not None else {}
    cached = memo.get(selection_key)
    if cached is not None:
        return cached.with_cache_hit()

    profiled: list[tuple[ScheduleCandidate, ScheduleTiming, str]] = []
    for candidate in legal_candidates:
        timing = profile_candidate(candidate)
        profiled.append((candidate, timing, schedule_candidate_key(candidate, abi)))

    selected, timing, selected_key = min(
        profiled,
        key=lambda item: (item[1].warm_run_ms, item[0].estimated_cost, item[0].schedule_id),
    )
    selection = WarmScheduleSelection(
        cache_key=selection_key,
        selected_schedule_key=selected_key,
        selected_schedule_id=selected.schedule_id,
        selected_config=selected.config,
        cache_hit=False,
        cold_compile_ms=timing.cold_compile_ms,
        warm_run_ms=timing.warm_run_ms,
        proof_hash=selected.proof_hash,
        codegen_hash=selected.codegen_hash,
        profiled_candidate_count=len(profiled),
        skipped_illegal_candidate_count=len(candidates) - len(legal_candidates),
    )
    memo[selection_key] = selection
    return selection


def serialize_warm_schedule_selection(selection: WarmScheduleSelection) -> str:
    return _stable_json(selection.to_json())


__all__ = [
    "ScheduleAbiFingerprint",
    "ScheduleCandidate",
    "ScheduleTiming",
    "WarmScheduleSelection",
    "schedule_candidate_key",
    "schedule_selection_key",
    "select_warm_schedule",
    "serialize_warm_schedule_selection",
]
