from __future__ import annotations

import json

import pytest

from tilelang.analysis.autotune_plan import (
    ScheduleAbiFingerprint,
    ScheduleCandidate,
    ScheduleTiming,
    schedule_candidate_key,
    select_warm_schedule,
    serialize_warm_schedule_selection,
)


def _abi(tilelang_hash: str = "tl-a") -> ScheduleAbiFingerprint:
    return ScheduleAbiFingerprint(
        tilelang=tilelang_hash,
        tvm="tvm-a",
        tvm_ffi="ffi-a",
        mlx="mlx-a",
    )


def _candidate(
    schedule_id: str,
    *,
    config: dict,
    legal: bool = True,
    proof_hash: str = "proof-a",
    codegen_hash: str = "codegen-a",
    estimated_cost: float = 1.0,
    rejection_reason: str = "",
) -> ScheduleCandidate:
    return ScheduleCandidate(
        schedule_id=schedule_id,
        op_signature="mamba3_path_c_bwd",
        shape=(1, 2048, 16, 128),
        dtype="bfloat16",
        target_kind="metal",
        config=config,
        legal=legal,
        proof_hash=proof_hash,
        codegen_hash=codegen_hash,
        estimated_cost=estimated_cost,
        rejection_reason=rejection_reason,
    )


def test_warm_schedule_profiles_only_legal_candidates_and_caches_hit():
    candidates = (
        _candidate("illegal", config={"threads": 64}, legal=False, rejection_reason="alias"),
        _candidate("slow", config={"threads": 128}, estimated_cost=2.0),
        _candidate("fast", config={"threads": 256}, estimated_cost=3.0),
    )
    timings = {
        "slow": ScheduleTiming(cold_compile_ms=10.0, warm_run_ms=2.0),
        "fast": ScheduleTiming(cold_compile_ms=12.0, warm_run_ms=1.0),
    }
    profiled: list[str] = []
    cache = {}

    def profile(candidate: ScheduleCandidate) -> ScheduleTiming:
        profiled.append(candidate.schedule_id)
        return timings[candidate.schedule_id]

    first = select_warm_schedule(
        candidates,
        abi=_abi(),
        profile_candidate=profile,
        cache=cache,
    )
    second = select_warm_schedule(
        candidates,
        abi=_abi(),
        profile_candidate=profile,
        cache=cache,
    )

    assert first.cache_hit is False
    assert first.selected_schedule_id == "fast"
    assert first.profiled_candidate_count == 2
    assert first.skipped_illegal_candidate_count == 1
    assert first.cold_compile_ms == 12.0
    assert first.warm_run_ms == 1.0
    assert profiled == ["slow", "fast"]

    assert second.cache_hit is True
    assert second.profiled_candidate_count == 0
    assert second.selected_schedule_id == "fast"
    assert profiled == ["slow", "fast"]


def test_warm_schedule_cache_invalidates_on_proof_or_codegen_hash_change():
    base = (_candidate("fast", config={"threads": 256}),)
    changed_proof = (
        _candidate("fast", config={"threads": 256}, proof_hash="proof-b"),
    )
    changed_codegen = (
        _candidate("fast", config={"threads": 256}, codegen_hash="codegen-b"),
    )
    cache = {}
    profiled: list[str] = []

    def profile(candidate: ScheduleCandidate) -> ScheduleTiming:
        profiled.append(f"{candidate.proof_hash}:{candidate.codegen_hash}")
        return ScheduleTiming(cold_compile_ms=5.0, warm_run_ms=1.0)

    first = select_warm_schedule(base, abi=_abi(), profile_candidate=profile, cache=cache)
    proof_miss = select_warm_schedule(
        changed_proof,
        abi=_abi(),
        profile_candidate=profile,
        cache=cache,
    )
    codegen_miss = select_warm_schedule(
        changed_codegen,
        abi=_abi(),
        profile_candidate=profile,
        cache=cache,
    )

    assert first.cache_key != proof_miss.cache_key
    assert first.cache_key != codegen_miss.cache_key
    assert proof_miss.cache_hit is False
    assert codegen_miss.cache_hit is False
    assert profiled == [
        "proof-a:codegen-a",
        "proof-b:codegen-a",
        "proof-a:codegen-b",
    ]


def test_warm_schedule_cache_invalidates_on_abi_hash_change():
    candidates = (_candidate("fast", config={"threads": 256}),)
    cache = {}
    profiled = 0

    def profile(_: ScheduleCandidate) -> ScheduleTiming:
        nonlocal profiled
        profiled += 1
        return ScheduleTiming(cold_compile_ms=5.0, warm_run_ms=1.0)

    first = select_warm_schedule(
        candidates,
        abi=_abi("tl-a"),
        profile_candidate=profile,
        cache=cache,
    )
    second = select_warm_schedule(
        candidates,
        abi=_abi("tl-b"),
        profile_candidate=profile,
        cache=cache,
    )

    assert first.cache_key != second.cache_key
    assert second.cache_hit is False
    assert profiled == 2


def test_warm_schedule_records_hashes_key_and_timing_stably():
    candidate = _candidate("fast", config={"threads": 256, "block": [16, 32]})
    selection = select_warm_schedule(
        (candidate,),
        abi=_abi(),
        profile_candidate=lambda _: ScheduleTiming(cold_compile_ms=7.5, warm_run_ms=1.25),
        cache={},
    )
    payload = json.loads(serialize_warm_schedule_selection(selection))

    assert payload == {
        "cache_hit": False,
        "cache_key": selection.cache_key,
        "codegen_hash": "codegen-a",
        "cold_compile_ms": 7.5,
        "profiled_candidate_count": 1,
        "proof_hash": "proof-a",
        "selected_config": {"block": [16, 32], "threads": 256},
        "selected_schedule_id": "fast",
        "selected_schedule_key": schedule_candidate_key(candidate, _abi()),
        "skipped_illegal_candidate_count": 0,
        "warm_run_ms": 1.25,
    }


def test_warm_schedule_rejects_all_illegal_candidates():
    with pytest.raises(ValueError, match="No legal schedule candidates"):
        select_warm_schedule(
            (
                _candidate(
                    "illegal",
                    config={"threads": 64},
                    legal=False,
                    rejection_reason="alias",
                ),
            ),
            abi=_abi(),
            profile_candidate=lambda _: ScheduleTiming(cold_compile_ms=1.0, warm_run_ms=1.0),
            cache={},
        )
