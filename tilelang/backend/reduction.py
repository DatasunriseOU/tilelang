from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from tvm.target import Target


_ANY_OP = "*"


@dataclass(frozen=True)
class ReductionLowererEntry:
    """Registered backend implementation for one or more reduction strategies."""

    name: str
    backend: str
    target_kinds: tuple[str, ...]
    strategies: tuple[str, ...]
    ops: tuple[str, ...]
    lowerer: str
    memory_visibility_scope: str
    scratch_scope: str | None
    internal_scratch_required: bool
    external_materialization_required: bool
    notes: str = ""

    def matches(self, key: "ReductionLowererKey") -> bool:
        return (
            key.target_kind in self.target_kinds
            and key.strategy in self.strategies
            and (_ANY_OP in self.ops or key.op in self.ops)
        )


@dataclass(frozen=True)
class ReductionLowererKey:
    """Cache key for a reproducible backend lowerer selection."""

    target_kind: str
    op: str
    strategy: str
    reduction_extent: int | None
    accumulator_dtype: str | None

    def stable_key(self) -> str:
        extent = "*" if self.reduction_extent is None else str(self.reduction_extent)
        dtype = "*" if self.accumulator_dtype is None else self.accumulator_dtype
        return "|".join((self.target_kind, self.op, self.strategy, extent, dtype))


@dataclass(frozen=True)
class ReductionLowererSelection:
    """Selected backend lowerer and sync/scratch contract."""

    name: str
    backend: str
    target_kind: str
    op: str
    strategy: str
    lowerer: str
    memory_visibility_scope: str
    scratch_scope: str | None
    internal_scratch_required: bool
    external_materialization_required: bool
    cache_key: str
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "target_kind": self.target_kind,
            "op": self.op,
            "strategy": self.strategy,
            "lowerer": self.lowerer,
            "memory_visibility_scope": self.memory_visibility_scope,
            "scratch_scope": self.scratch_scope,
            "internal_scratch_required": self.internal_scratch_required,
            "external_materialization_required": self.external_materialization_required,
            "cache_key": self.cache_key,
            "notes": self.notes,
        }


_REDUCTION_LOWERERS: list[ReductionLowererEntry] = []


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_target_kind(kind: str) -> str:
    normalized = str(kind).strip().split(maxsplit=1)[0].lower()
    if normalized in {"hip", "rocm"}:
        return "rocm"
    if normalized in {"llvm", "c", "cpu"}:
        return "cpu"
    return normalized


def _target_kind(target: str | Target) -> str:
    if isinstance(target, Target):
        return _normalize_target_kind(target.kind.name)
    try:
        parsed = Target(target)
    except Exception:
        return _normalize_target_kind(target)
    return _normalize_target_kind(parsed.kind.name)


def register_reduction_lowerer(
    *,
    name: str,
    backend: str,
    target_kinds: str | Iterable[str],
    strategies: str | Iterable[str],
    ops: str | Iterable[str],
    lowerer: str,
    memory_visibility_scope: str,
    scratch_scope: str | None,
    internal_scratch_required: bool,
    external_materialization_required: bool,
    notes: str = "",
) -> None:
    """Register or replace a backend-specific reduction lowerer."""

    entry = ReductionLowererEntry(
        name=name,
        backend=backend,
        target_kinds=tuple(_normalize_target_kind(kind) for kind in _as_tuple(target_kinds)),
        strategies=_as_tuple(strategies),
        ops=_as_tuple(ops),
        lowerer=lowerer,
        memory_visibility_scope=memory_visibility_scope,
        scratch_scope=scratch_scope,
        internal_scratch_required=internal_scratch_required,
        external_materialization_required=external_materialization_required,
        notes=notes,
    )
    for idx, registered in enumerate(_REDUCTION_LOWERERS):
        if registered.name == name:
            _REDUCTION_LOWERERS[idx] = entry
            _select_reduction_lowerer_cached.cache_clear()
            return
    _REDUCTION_LOWERERS.append(entry)
    _select_reduction_lowerer_cached.cache_clear()


def registered_reduction_lowerers() -> tuple[ReductionLowererEntry, ...]:
    """Return registered lowerers in deterministic registration order."""

    return tuple(_REDUCTION_LOWERERS)


@lru_cache(maxsize=256)
def _select_reduction_lowerer_cached(
    key: ReductionLowererKey,
) -> ReductionLowererSelection:
    matches = [entry for entry in _REDUCTION_LOWERERS if entry.matches(key)]
    if not matches:
        raise ValueError(
            "No reduction lowerer registered for "
            f"target={key.target_kind!r}, op={key.op!r}, strategy={key.strategy!r}"
        )
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise ValueError(
            "Multiple reduction lowerers matched "
            f"target={key.target_kind!r}, op={key.op!r}, "
            f"strategy={key.strategy!r}: {names}"
        )
    entry = matches[0]
    return ReductionLowererSelection(
        name=entry.name,
        backend=entry.backend,
        target_kind=key.target_kind,
        op=key.op,
        strategy=key.strategy,
        lowerer=entry.lowerer,
        memory_visibility_scope=entry.memory_visibility_scope,
        scratch_scope=entry.scratch_scope,
        internal_scratch_required=entry.internal_scratch_required,
        external_materialization_required=entry.external_materialization_required,
        cache_key=key.stable_key(),
        notes=entry.notes,
    )


def select_reduction_lowerer(
    target: str | Target,
    *,
    op: str,
    strategy: str,
    reduction_extent: int | None = None,
    accumulator_dtype: str | None = None,
) -> ReductionLowererSelection:
    """Select the exact backend lowerer for one reduction strategy."""

    key = ReductionLowererKey(
        target_kind=_target_kind(target),
        op=str(op),
        strategy=str(strategy),
        reduction_extent=reduction_extent,
        accumulator_dtype=accumulator_dtype,
    )
    return _select_reduction_lowerer_cached(key)


def resolve_reduction_lowerer(
    target: str | Target,
    *,
    op: str,
    candidate_strategies: Iterable[str],
    reduction_extent: int | None = None,
    accumulator_dtype: str | None = None,
) -> ReductionLowererSelection:
    """Select the first registered backend lowerer from ordered candidates."""

    strategies = tuple(candidate_strategies)
    errors: list[str] = []
    for strategy in strategies:
        try:
            return select_reduction_lowerer(
                target,
                op=op,
                strategy=strategy,
                reduction_extent=reduction_extent,
                accumulator_dtype=accumulator_dtype,
            )
        except ValueError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors)
    raise ValueError(
        f"No reduction lowerer resolved for target={_target_kind(target)!r}, "
        f"op={op!r}, candidates={strategies!r}. {detail}"
    )


def reduction_lowerer_cache_info():
    """Expose cache stats for tests and diagnostics."""

    return _select_reduction_lowerer_cached.cache_info()


__all__ = [
    "ReductionLowererEntry",
    "ReductionLowererKey",
    "ReductionLowererSelection",
    "reduction_lowerer_cache_info",
    "registered_reduction_lowerers",
    "register_reduction_lowerer",
    "resolve_reduction_lowerer",
    "select_reduction_lowerer",
]
