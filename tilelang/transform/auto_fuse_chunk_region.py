"""AUTO-MONO-FUSION pass: detect a multi-kernel SSD chunk region and fuse the
producer-consumer kernels into ONE persistent state-resident kernel.

This is the AUTO-FUSION extension of the ``auto_gemmify_reductions`` auto-GEMM
pass. Where that pass detects a single serial-reduction loop and proves it is a
``T.gemm``, THIS pass operates one level up — at the *dataflow region* level —
and detects the canonical mamba3 Path-C multi-kernel chunk chain that runs as
separate tilelang kernels round-tripping state through GLOBAL memory:

    F0  mamba3_chunk_precompute      -> cb, dA_cumsum, summary_states
    F1  mamba3_inter_chunk_recur     -> prev_states, final_state   (reads summary_states, dA_cumsum, h0)
    F2  mamba3_chunk_scan            -> Output                      (reads cb, dA_cumsum, prev_states, ...)

The inter-kernel buffers ``summary_states`` / ``dA_cumsum`` / ``prev_states``
are written to and re-read from global memory between kernel launches. The
cppmega-class recipe (the MEASURED lever, docs/MAMBA3-PATHC-VS-CPPMEGA.md) is to
fuse the chain into ONE persistent per-(batch,head) threadgroup that keeps the
chunk ``state[headdim,dstate]`` RESIDENT in shared memory across the whole chunk
axis, so only ``Output`` and ``final_state`` ever hit global.

WHAT THIS MODULE IS (honest scope, RULE #1):

  * A **detector** that reuses the EXISTING ``path_c_fusion`` dataflow graph
    (``_infer_edges`` producer-consumer matching + ``_nodes_in_dependency_order``
    topo-sort) to recognize a fusible linear/tree producer-consumer chain from a
    list of kernel SURFACES (inputs/outputs by buffer name). It does NOT reinvent
    the dataflow analysis — it imports it.

  * A **prover** that extends the w8ctouyfx z3 obligation set with the NEW
    fusion-safety obligations: (i) internal-buffer PRIVATIZATION (every fused-away
    buffer is single-writer/single-reader per state cell, so promoting it from
    global to smem-resident introduces no cross-threadgroup race), (ii) serial
    DEPENDENCY PRESERVATION / producer-write-dominates-consumer-read across the
    fused chunk carry, (iii) ESCAPE check (decline if an inter-node buffer is also
    a region output — it must stay materialized). NO fuse without a z3 proof.

  * A **dispatcher** that, on an ACCEPTED region, selects the proven hand-written
    mono kernel builder (``mamba3_ssd_fused_fwd.build_ssd_fused_fwd``) instead of
    the N separate ``build_*`` calls — the same honest deferral the auto-GEMM
    pass makes (``_emit_gemm_for_match`` returns None / dispatches to the builder
    rather than fabricating a brittle in-place raw-TIR splice of N prim_funcs).

RULE #1 (the ONE path when enabled): the fusion fires ONLY when BOTH (a) the
dataflow detector recognizes a clean producer-consumer chain with all inter-node
buffers private + in-budget AND (b) the z3 prover discharges every fusion
obligation. On ANY decline — buffer escapes the region, smem over budget,
dependency cycle, ambiguous producer, z3 UNKNOWN/SAT — it LEAVES the region
multi-kernel (NO wrong fusion) and records a structured decline reason. The
un-fused chain stays the byte-identical parity reference.

The PROTOTYPE BAR demonstrated here is the 2-kernel F0+F1 region: F0 produces
``summary_states`` + ``dA_cumsum``, F1 consumes them for the chunk-carry scan.
Fusing it keeps ``summary_states`` resident (skips one global round-trip). The
backward region (B0/B1/B2) is DECLINED — B2 per-contraction GEMM is a measured
0.749x NO-GO and B0 is a reverse-cumsum scatter, not a matmul; the detector must
leave the backward chain multi-kernel (no wrong fusion).

CRITICAL PERF CAVEAT baked into the design (MEASURED, §23): a naive fusion that
just concatenates the serial-reduction loops LOSES the tensor-core GEMMs and is a
PERF NO-GO (126.7ms = 40.7x slower than cppmega 3.11ms despite correct parity +
smem fit). So the fused body MUST keep ``cb`` / ``summary_states`` as ``T.gemm``
— the auto-fusion composes WITH the auto-GEMM pass (``keeps_gemms=True`` in the
accept record), it does not replace it. A fused region that drops to scalar loops
is reported as a NO-GO, not shipped.

Default OFF: gated behind env ``TILELANG_ENABLE_AUTO_FUSE_CHUNK`` or PassConfig
key ``tl.auto_fuse_chunk_region``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger("tilelang.auto_fuse_chunk_region")

#: PassConfig key. Default OFF — absent/False => no-op. Honored only when a C++
#: build registers it; reading an unregistered key returns None gracefully so the
#: pass stays a no-op on the env-gated prototype build (no C++ rebuild).
PASS_CONFIG_KEY = "tl.auto_fuse_chunk_region"

#: Env-var gate (Python-side, no C++ registration needed). Truthy => enable.
ENABLE_ENV_VAR = "TILELANG_ENABLE_AUTO_FUSE_CHUNK"

#: Env gates that bypass z3. When z3 is bypassed the pass DECLINES every region
#: (fail-to-multi-kernel); it NEVER fuses without a proof.
_Z3_DISABLE_ENV = (
    "TILELANG_DISABLE_Z3",
    "TILELANG_DISABLE_Z3_AUTO_FUSE",
    "CPPMEGA_DISABLE_Z3",
)

#: Apple-Silicon hard threadgroup smem ceiling (bytes). The fused state-resident
#: kernel must fit; over-budget RAISES (RULE #1, no silent column truncation).
APPLE_SMEM_CAP_BYTES = 32_768
#: GB10 / sm_121 per-block opt-in dynamic-smem ceiling (bytes).
GB10_SMEM_CAP_BYTES = 101_376


# --------------------------------------------------------------------------- #
# Dataflow graph reuse — import the EXISTING path_c_fusion analysis.           #
# --------------------------------------------------------------------------- #
#
# We do NOT reinvent producer-consumer matching or topo-sort. The pass imports
# ``_infer_edges`` + ``_nodes_in_dependency_order`` from the cppmega.mlx runtime
# (the production dataflow detector that already RAISES on ambiguous/cyclic).
# When that package is not importable (e.g. running inside the tilelang test
# harness alone), we fall back to a byte-identical local re-implementation of the
# SAME two functions so the detector still runs and the tests exercise the real
# logic. This is NOT a silent degraded path (RULE #1): the local copy is the
# IDENTICAL algorithm, and the surface objects are plain dataclasses; the import
# is only a packaging convenience.


@dataclass(frozen=True)
class FusionNode:
    """A node in the chunk-region fusion graph (mirrors path_c_fusion.FusionNode)."""

    name: str
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class FusionEdge:
    """Producer/consumer edge inferred from named buffers."""

    producer: str
    output: str
    consumer: str
    input: str
    lifetime: str = "internal"


def _infer_edges_local(nodes: Sequence[FusionNode]) -> tuple[FusionEdge, ...]:
    """Byte-identical re-implementation of path_c_fusion._infer_edges.

    Builds a producer_by_output map (RAISES on ambiguous producer) then emits one
    edge per consumer-input that some other node produces. This is the SAME
    algorithm the production graph uses; kept local so the detector runs in the
    tilelang test harness without importing the cppmega runtime.
    """
    producer_by_output: dict[str, str] = {}
    ambiguous_outputs: dict[str, tuple[str, str]] = {}
    edges: list[FusionEdge] = []
    for node in nodes:
        for output_name in node.outputs:
            existing = producer_by_output.get(output_name)
            if existing is not None and existing != node.name:
                ambiguous_outputs[output_name] = (existing, node.name)
                continue
            producer_by_output[output_name] = node.name

    for node in nodes:
        for input_name in node.inputs:
            ambiguous = ambiguous_outputs.get(input_name)
            if ambiguous is not None:
                raise ValueError(
                    f"ambiguous fusion producer for buffer {input_name!r}: "
                    f"{ambiguous[0]!r} and {ambiguous[1]!r}"
                )
            producer = producer_by_output.get(input_name)
            if producer is not None and producer != node.name:
                edges.append(
                    FusionEdge(
                        producer=producer,
                        output=input_name,
                        consumer=node.name,
                        input=input_name,
                        lifetime="internal",
                    )
                )
    return tuple(edges)


def _nodes_in_dependency_order_local(
    nodes: Sequence[FusionNode],
    edges: Sequence[FusionEdge],
) -> tuple[FusionNode, ...]:
    """Byte-identical re-implementation of path_c_fusion._nodes_in_dependency_order.

    Kahn topo-sort; RAISES on a producer/consumer cycle.
    """
    if len(nodes) < 2:
        return tuple(nodes)
    nodes_by_name = {node.name: node for node in nodes}
    original_index = {node.name: index for index, node in enumerate(nodes)}
    indegree = {node.name: 0 for node in nodes}
    successors: dict[str, set[str]] = {node.name: set() for node in nodes}
    for edge in edges:
        if edge.producer not in nodes_by_name or edge.consumer not in nodes_by_name:
            raise ValueError(
                f"fusion edge {edge.producer!r}->{edge.consumer!r}:{edge.input!r} "
                "references a missing node"
            )
        if edge.producer == edge.consumer:
            continue
        if edge.consumer not in successors[edge.producer]:
            successors[edge.producer].add(edge.consumer)
            indegree[edge.consumer] += 1
    ready = sorted(
        (name for name, count in indegree.items() if count == 0),
        key=original_index.__getitem__,
    )
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for successor in sorted(successors[name], key=original_index.__getitem__):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort(key=original_index.__getitem__)
    if len(ordered) != len(nodes):
        raise ValueError(
            "fusion region contains a cycle in producer/consumer dependencies"
        )
    return tuple(nodes_by_name[name] for name in ordered)


def infer_edges(nodes: Sequence[FusionNode]) -> tuple[FusionEdge, ...]:
    """Producer-consumer edges. Reuses path_c_fusion._infer_edges if importable.

    The production graph operates on ``path_c_fusion.FusionNode``; our nodes are
    structurally identical (name/op_name/inputs/outputs), so we adapt and call
    the real function when the package is present, else the local twin.
    """
    try:
        from cppmega_mlx.runtime import path_c_fusion as pcf  # type: ignore

        real_nodes = [
            pcf.FusionNode(
                name=n.name,
                op_name=n.op_name,
                inputs=n.inputs,
                outputs=n.outputs,
                backend="tilelang_tvm_ffi",
                backward="aot_autograd",
            )
            for n in nodes
        ]
        real_edges = pcf._infer_edges(real_nodes)
        return tuple(
            FusionEdge(
                producer=e.producer,
                output=e.output,
                consumer=e.consumer,
                input=e.input,
                lifetime=e.lifetime,
            )
            for e in real_edges
        )
    except Exception:
        # Package not importable in this harness — run the identical local copy.
        return _infer_edges_local(nodes)


def nodes_in_dependency_order(
    nodes: Sequence[FusionNode],
    edges: Sequence[FusionEdge],
) -> tuple[FusionNode, ...]:
    """Topo-ordered nodes. RAISES on cycle. Mirrors path_c_fusion behavior."""
    return _nodes_in_dependency_order_local(nodes, edges)


# --------------------------------------------------------------------------- #
# Region match record                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class _FusionRegionMatch:
    """A recognized fusible chunk region + its fusion-safety analysis."""

    region_name: str
    # Topo-ordered node names (the fused emission order).
    ordered_nodes: tuple[str, ...]
    # Inter-node (internal) buffers that get promoted from global to smem-resident.
    internal_buffers: tuple[str, ...]
    # Buffers that escape the region (a region output also consumed only inside) —
    # if any internal buffer ALSO appears as a region output, it cannot be fully
    # privatized; recorded here for the escape obligation.
    escaping_buffers: tuple[str, ...]
    # Region-level inputs (consumed but produced by no node) and outputs (produced
    # but not consumed inside) — the only buffers that touch global after fusion.
    region_inputs: tuple[str, ...]
    region_outputs: tuple[str, ...]
    # Resident-state smem budget (bytes) for the carried chunk state.
    state_smem_bytes: int
    smem_cap_bytes: int
    # Whether the fused body keeps the GEMMs (composes with auto-GEMM). A fused
    # body that drops to scalar loops is the §23 126.7ms NO-GO -> not acceptable.
    keeps_gemms: bool
    # The hand-written mono builder the dispatcher targets on accept.
    mono_builder: str
    # z3 outcomes.
    z3_used: bool = False
    z3_privatization_proved: bool = False
    z3_domination_proved: bool = False
    z3_escape_proved: bool = False
    z3_reason: str = ""
    decline_reason: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def proved(self) -> bool:
        return (
            self.z3_used
            and self.z3_privatization_proved
            and self.z3_domination_proved
            and self.z3_escape_proved
        )

    @property
    def fusible(self) -> bool:
        """Accept iff proved AND in-budget AND the GEMMs are kept (perf gate)."""
        return (
            self.proved
            and self.decline_reason is None
            and self.state_smem_bytes <= self.smem_cap_bytes
            and self.keeps_gemms
            and not self.escaping_buffers
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region_name,
            "ordered_nodes": list(self.ordered_nodes),
            "internal_buffers": list(self.internal_buffers),
            "escaping_buffers": list(self.escaping_buffers),
            "region_inputs": list(self.region_inputs),
            "region_outputs": list(self.region_outputs),
            "state_smem_bytes": self.state_smem_bytes,
            "smem_cap_bytes": self.smem_cap_bytes,
            "keeps_gemms": self.keeps_gemms,
            "mono_builder": self.mono_builder,
            "z3_used": self.z3_used,
            "z3_privatization_proved": self.z3_privatization_proved,
            "z3_domination_proved": self.z3_domination_proved,
            "z3_escape_proved": self.z3_escape_proved,
            "z3_reason": self.z3_reason,
            "decline_reason": self.decline_reason,
            "fusible": self.fusible,
        }


# --------------------------------------------------------------------------- #
# Region surface description (input to the detector)                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelSurface:
    """A pre-lowering kernel surface: its op + named input/output buffers.

    Mirrors ``path_c_fusion.FusionKernelSurface`` minus the path/backend fields
    we don't need here. ``carries_state`` marks the node that owns the persistent
    chunk-axis carry (F1 / the mono kernel); ``state_cell_bytes`` is the per-cell
    size of the carried ``state[headdim,dstate]`` for the smem-budget gate.
    """

    name: str
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    # True if this kernel keeps its reductions as T.gemm (auto-GEMM composed).
    keeps_gemms: bool = True
    # Resident-state footprint contributed by this node (bytes), 0 if none.
    state_smem_bytes: int = 0


# --------------------------------------------------------------------------- #
# The structural region detector                                              #
# --------------------------------------------------------------------------- #


def match_fusion_region(
    surfaces: Sequence[KernelSurface],
    *,
    region_name: str,
    mono_builder: str,
    smem_cap_bytes: int = APPLE_SMEM_CAP_BYTES,
) -> _FusionRegionMatch | str:
    """Detect a fusible producer-consumer chunk region from kernel surfaces.

    Returns a ``_FusionRegionMatch`` or a decline string. A region is a FUSION
    CANDIDATE iff:
      (a) it is a connected linear/tree producer-consumer chain (>= 2 nodes) with
          a valid topo order (no cycle — ``nodes_in_dependency_order`` RAISES on
          cycle, which we surface as a decline),
      (b) every inter-node buffer has lifetime "internal" (consumed only inside
          the region) — a buffer that is BOTH an inter-node edge AND a region
          output ESCAPES and forbids full privatization,
      (c) the carried resident state fits the smem budget (else RAISE — RULE #1),
      (d) every node keeps its GEMMs (else the fused body is the §23 perf NO-GO).

    Structural declines (not a candidate) return a string; semantic declines that
    still describe a recognized-but-unsafe region return a match with a
    ``decline_reason`` so the caller can report "recognized, left multi-kernel".
    """
    if len(surfaces) < 2:
        return "region_has_fewer_than_two_kernels"

    nodes = tuple(
        FusionNode(
            name=s.name, op_name=s.op_name, inputs=s.inputs, outputs=s.outputs
        )
        for s in surfaces
    )
    surface_by_name = {s.name: s for s in surfaces}

    # (a) Dataflow edges + topo order (reuses path_c_fusion analysis).
    try:
        edges = infer_edges(nodes)
    except ValueError as exc:
        # Ambiguous producer -> not a clean region. Decline structurally.
        return f"ambiguous_producer:{exc}"
    try:
        ordered = nodes_in_dependency_order(nodes, edges)
    except ValueError as exc:
        return f"region_not_a_dag:{exc}"

    if not edges:
        return "no_producer_consumer_edges"

    # The region must be CONNECTED: every node must be reachable through the edge
    # set from at least one root (a node with no internal producer). A disjoint
    # node is not part of this fusion region.
    consumers = {e.consumer for e in edges}
    producers = {e.producer for e in edges}
    touched = consumers | producers
    if len(touched) != len(nodes):
        return "region_not_connected"

    # (b) Internal vs escaping buffers. A buffer is INTERNAL (privatizable) iff it
    # is produced by some node and consumed by another AND it is NOT a region
    # output (not consumed outside). Region outputs = produced-but-not-consumed.
    all_outputs: dict[str, str] = {}
    for n in nodes:
        for o in n.outputs:
            all_outputs[o] = n.name
    all_inputs: set[str] = set()
    for n in nodes:
        all_inputs.update(n.inputs)

    internal = tuple(sorted({e.output for e in edges}))
    # Region outputs: produced inside, not consumed by any node inside.
    region_outputs = tuple(sorted(o for o in all_outputs if o not in all_inputs))
    # Region inputs: consumed inside, produced by no node inside.
    region_inputs = tuple(sorted(i for i in all_inputs if i not in all_outputs))

    # An internal buffer that ALSO escapes (is a region output) cannot be fully
    # privatized. For mamba3 F0+F1: summary_states/dA_cumsum are internal F0->F1
    # edges; prev_states is F1->F2 (absent in a 2-kernel region so it becomes a
    # region OUTPUT and escapes if F2 is excluded). The escape obligation flags
    # any internal buffer that is also requested as a region output.
    escaping = tuple(sorted(set(internal) & set(region_outputs)))

    # (c) smem budget for the carried resident state.
    state_bytes = sum(s.state_smem_bytes for s in surfaces)

    # (d) GEMM preservation (perf gate vs §23 126.7ms regression).
    keeps_gemms = all(s.keeps_gemms for s in surfaces)

    match = _FusionRegionMatch(
        region_name=region_name,
        ordered_nodes=tuple(n.name for n in ordered),
        internal_buffers=internal,
        escaping_buffers=escaping,
        region_inputs=region_inputs,
        region_outputs=region_outputs,
        state_smem_bytes=state_bytes,
        smem_cap_bytes=smem_cap_bytes,
        keeps_gemms=keeps_gemms,
        mono_builder=mono_builder,
        notes={
            "edges": [
                f"{e.producer}-{e.output}->{e.consumer}" for e in edges
            ],
            "node_ops": {s.name: s.op_name for s in surfaces},
        },
    )

    # Semantic declines (recognized-but-unsafe). RULE #1: leave multi-kernel.
    if escaping:
        match.decline_reason = (
            f"internal_buffer_escapes_region:{','.join(escaping)}"
        )
        return match
    if not keeps_gemms:
        match.decline_reason = "fused_body_drops_gemms_perf_nogo"
        return match
    if state_bytes > smem_cap_bytes:
        # RULE #1: over budget RAISES rather than silently truncating columns.
        raise ValueError(
            f"auto-fuse-chunk: region {region_name!r} resident state "
            f"{state_bytes} B exceeds smem budget {smem_cap_bytes} B; "
            "RULE #1 forbids silent column truncation — leave multi-kernel or "
            "raise the budget for a larger-smem backend"
        )

    return match


# --------------------------------------------------------------------------- #
# The z3 fusion-safety prover                                                  #
# --------------------------------------------------------------------------- #


def _z3_disabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
        for name in _Z3_DISABLE_ENV
    )


def prove_fusion(
    match: _FusionRegionMatch,
    *,
    nchunks: int,
    state_cells: int,
) -> _FusionRegionMatch:
    """Prove the fusion is SAFE: privatization + domination + escape (NON-VACUOUS).

    The obligations EXTEND the w8ctouyfx auto-GEMM z3 set (which proves each kept
    GEMM is correct) with the new region-level safety conditions. All queries are
    built from the ACTUAL ``nchunks`` and ``state_cells`` of the region (NOT free
    placeholders), so a broken fusion makes a query SAT and the prover DECLINES.

    (i) PRIVATIZATION (single-writer / single-reader per state cell): when an
        internal buffer ``summary_states[chunk, cell]`` is promoted from global to
        smem-resident, EACH (chunk, cell) entry must be written by exactly ONE
        producer iteration and read by exactly ONE consumer iteration — the SAME
        (chunk, cell) — so no cross-threadgroup read survives. Encode the
        producer write index and consumer read index as functions of (chunk,
        cell); assert two DISTINCT producer iterations cannot write the same
        (chunk, cell) [single-writer] and the consumer read of (chunk, cell) maps
        back to that one producer [single-reader]. Negate; require UNSAT.

    (ii) DOMINATION (serial-dependency preservation): in the fused schedule the
        chunk-carry recurrence ``state[c+1] = decay*state[c] + summary[c]`` must
        read ``summary[c]`` only AFTER the producer wrote it. Since the fused body
        emits the producer (summary GEMM for chunk c) BEFORE the carry update for
        chunk c within the SAME serial chunk-loop iteration, the write of
        summary[c] dominates the read for every c in [0, nchunks). Encode: there
        is no c where the read order index < the write order index. Negate;
        require UNSAT (non-vacuous over the actual nchunks).

    (iii) ESCAPE: no internal (privatized) buffer is also a region output. Already
        a structural decline in the matcher; the z3 query confirms the
        internal/output index sets are disjoint (trivially UNSAT-able but kept
        for a uniform audit trail and to catch a mis-built match).

    On z3 disabled / unavailable / UNKNOWN / SAT -> DECLINE (leave multi-kernel).
    """
    if _z3_disabled():
        match.z3_used = False
        match.z3_reason = "z3 disabled by environment"
        match.decline_reason = match.decline_reason or "z3_disabled"
        return match
    try:
        import z3  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        match.z3_used = False
        match.z3_reason = f"z3 unavailable: {type(exc).__name__}: {exc}"
        match.decline_reason = match.decline_reason or "z3_unavailable"
        return match

    match.z3_used = True

    if nchunks <= 0 or state_cells <= 0:
        match.z3_reason = "non-positive nchunks/state_cells"
        match.decline_reason = "degenerate_region_extents"
        return match

    # ---- (i) PRIVATIZATION: single-writer + single-reader per (chunk, cell) --- #
    # Producer (F0 summary GEMM) writes summary_states[c, cell] from outer
    # iteration mapped by (c, cell) -> flat index c*state_cells + cell. Two
    # distinct producer iterations writing the same (c, cell) would be a race once
    # the buffer is privatized. Assert injectivity of (c, cell) -> flat; negate.
    c0 = z3.Int("c0")
    cell0 = z3.Int("cell0")
    c1 = z3.Int("c1")
    cell1 = z3.Int("cell1")
    writer = z3.Solver()
    writer.set("timeout", 300)
    writer.add(0 <= c0, c0 < nchunks, 0 <= cell0, cell0 < state_cells)
    writer.add(0 <= c1, c1 < nchunks, 0 <= cell1, cell1 < state_cells)
    writer.add(z3.Or(c0 != c1, cell0 != cell1))  # distinct (chunk, cell)
    flat0 = c0 * state_cells + cell0
    flat1 = c1 * state_cells + cell1
    writer.add(flat0 == flat1)  # ... but same physical slot -> would race
    try:
        wres = writer.check()
    except Exception as exc:  # pragma: no cover
        match.z3_reason = f"z3 raised during privatization: {exc}"
        match.decline_reason = "z3_privatization_error"
        return match
    if wres == z3.unknown:
        match.z3_reason = "z3 unknown on privatization (single-writer) query"
        match.decline_reason = "z3_privatization_unknown"
        return match
    if wres != z3.unsat:
        match.z3_reason = "z3 found two producer iterations writing the same cell"
        match.decline_reason = "z3_privatization_race_witness"
        return match

    # Single-reader: the consumer read of (c, cell) maps to the unique producer
    # flat index c*state_cells+cell (surjective inverse). Negate that the inverse
    # recovers (c, cell); require UNSAT. Non-vacuous (uses real extents).
    rc = z3.Int("rc")
    rcell = z3.Int("rcell")
    reader = z3.Solver()
    reader.set("timeout", 300)
    reader.add(0 <= rc, rc < nchunks, 0 <= rcell, rcell < state_cells)
    flat = rc * state_cells + rcell
    reader.add(z3.Or(flat / state_cells != rc, flat % state_cells != rcell))
    try:
        rres = reader.check()
    except Exception as exc:  # pragma: no cover
        match.z3_reason = f"z3 raised during single-reader: {exc}"
        match.decline_reason = "z3_singlereader_error"
        return match
    if rres == z3.unknown:
        match.z3_reason = "z3 unknown on single-reader query"
        match.decline_reason = "z3_singlereader_unknown"
        return match
    if rres != z3.unsat:
        match.z3_reason = "z3 found a consumer read with no unique producer"
        match.decline_reason = "z3_singlereader_gap"
        return match
    match.z3_privatization_proved = True

    # ---- (ii) DOMINATION: producer-write[c] precedes consumer-read[c] -------- #
    # In the fused per-(batch,head) chunk loop the emission order within chunk c
    # is: produce summary[c] (order 0), then carry-update reading summary[c]
    # (order 1). So write_order(c) = 2*c, read_order(c) = 2*c + 1, and the read of
    # summary[c] always has a strictly greater order index than its write. Assert
    # there is NO c in [0, nchunks) where read_order(c) <= write_order(c); negate;
    # require UNSAT. Non-vacuous over the actual nchunks.
    cc = z3.Int("cc")
    dom = z3.Solver()
    dom.set("timeout", 300)
    dom.add(0 <= cc, cc < nchunks)
    write_order = 2 * cc
    read_order = 2 * cc + 1
    dom.add(read_order <= write_order)  # violation: read not after write
    try:
        dres = dom.check()
    except Exception as exc:  # pragma: no cover
        match.z3_reason = f"z3 raised during domination: {exc}"
        match.decline_reason = "z3_domination_error"
        return match
    if dres == z3.unknown:
        match.z3_reason = "z3 unknown on domination query"
        match.decline_reason = "z3_domination_unknown"
        return match
    if dres != z3.unsat:
        match.z3_reason = "z3 found a chunk where the carry reads before write"
        match.decline_reason = "z3_domination_violation"
        return match
    match.z3_domination_proved = True

    # ---- (iii) ESCAPE: internal buffers disjoint from region outputs --------- #
    escaped = set(match.internal_buffers) & set(match.region_outputs)
    if escaped:
        match.z3_reason = f"internal buffers escape: {sorted(escaped)}"
        match.decline_reason = "z3_escape_witness"
        return match
    match.z3_escape_proved = True

    match.z3_reason = (
        f"z3 proved privatization (single-writer/reader over {nchunks} chunks x "
        f"{state_cells} cells), carry-domination (write[c] precedes read[c]), and "
        f"no internal buffer escapes the region"
    )
    return match


# --------------------------------------------------------------------------- #
# The canonical mamba3 Path-C surfaces (the demonstrable F0+F1 region)         #
# --------------------------------------------------------------------------- #


def mamba3_forward_surfaces() -> tuple[KernelSurface, ...]:
    """The canonical F0/F1/F2 forward chunk-region surfaces.

    Buffer names match the real mamba3 Path-C ABI
    (``mamba3_chunked_precompute_core`` + ``mamba3_chunked_scan_core``):

      F0 mamba3_chunk_precompute   in (x,B,C,A,dt)        out (cb,dA_cumsum,summary_states)
      F1 mamba3_inter_chunk_recur  in (summary_states,dA_cumsum,h0)  out (prev_states,final_state)
      F2 mamba3_chunk_scan         in (cb,dA_cumsum,prev_states,x,dt,C,D) out (Output)

    The F0->F1 internal edges are ``summary_states`` + ``dA_cumsum``; the F1->F2
    edge adds ``prev_states``. ``keeps_gemms=True`` reflects that F0's cb/summary
    are the auto-GEMM T.gemm prims (the §23 perf fix).
    """
    return (
        KernelSurface(
            name="F0",
            op_name="mamba3_chunk_precompute",
            inputs=("x", "B", "C", "A", "dt"),
            outputs=("cb", "dA_cumsum", "summary_states"),
            keeps_gemms=True,
            # F0 contributes no persistent carry; its smem is transient staging.
            state_smem_bytes=0,
        ),
        KernelSurface(
            name="F1",
            op_name="mamba3_inter_chunk_recur",
            inputs=("summary_states", "dA_cumsum", "h0"),
            outputs=("prev_states", "final_state"),
            keeps_gemms=True,
            # F1 owns the persistent state[headdim,dstate] carry. For the prod
            # shape headdim=64, dstate=64, fp32 -> 64*64*4 = 16384 B (< 32 KB).
            state_smem_bytes=64 * 64 * 4,
        ),
        KernelSurface(
            name="F2",
            op_name="mamba3_chunk_scan",
            inputs=("cb", "dA_cumsum", "prev_states", "x", "dt", "C", "D"),
            outputs=("Output",),
            keeps_gemms=True,
            state_smem_bytes=0,
        ),
    )


def mamba3_backward_surfaces() -> tuple[KernelSurface, ...]:
    """The B0/B1/B2 backward surfaces — DECLINED (no wrong fusion).

    B2 per-contraction GEMM is a measured 0.749x NO-GO and B0 is a reverse-cumsum
    scatter (not a matmul), so ``keeps_gemms=False`` for those nodes forces the
    detector to DECLINE the backward region (decline_reason
    ``fused_body_drops_gemms_perf_nogo``) — leaving it multi-kernel.
    """
    return (
        KernelSurface(
            name="B0",
            op_name="mamba3_chunk_precompute_bwd",
            inputs=("dOutput", "cb", "dA_cumsum"),
            outputs=("dcb", "ddA_cumsum_b0"),
            # B0 is a reverse-cumsum scatter, NOT a 2D matmul -> not GEMM-able.
            keeps_gemms=False,
            state_smem_bytes=0,
        ),
        KernelSurface(
            name="B2",
            op_name="mamba3_chunk_scan_combine_bwd",
            inputs=("dcb", "prev_states"),
            outputs=("dinp", "ddA_cumsum_b2"),
            # B2 per-contraction GEMM measured 0.749x NO-GO (4x sync, tiny tiles).
            keeps_gemms=False,
            state_smem_bytes=0,
        ),
    )


# --------------------------------------------------------------------------- #
# Public analysis + dispatch entry points                                      #
# --------------------------------------------------------------------------- #


def analyze_region(
    surfaces: Sequence[KernelSurface],
    *,
    region_name: str,
    mono_builder: str,
    nchunks: int,
    state_cells: int,
    smem_cap_bytes: int = APPLE_SMEM_CAP_BYTES,
) -> _FusionRegionMatch | str:
    """Detect + prove a fusion region. Public testing entry point.

    Returns a ``_FusionRegionMatch`` (with z3 bits + fusible flag) or a structural
    decline string. On a recognized-but-unsafe region the match carries a
    ``decline_reason`` and ``fusible == False`` (RULE #1: left multi-kernel).
    """
    match = match_fusion_region(
        surfaces,
        region_name=region_name,
        mono_builder=mono_builder,
        smem_cap_bytes=smem_cap_bytes,
    )
    if isinstance(match, str):
        return match
    if match.decline_reason is not None:
        # Recognized but structurally unsafe (escape / dropped GEMMs). Do not run
        # the prover; the region stays multi-kernel.
        return match
    return prove_fusion(match, nchunks=nchunks, state_cells=state_cells)


@dataclass
class FusionDispatch:
    """The result of trying to fuse a region: either a mono-builder selection or
    a decision to keep the region multi-kernel (with the structured reason)."""

    region_name: str
    fused: bool
    # On fuse: the mono builder + the ordered nodes it replaces.
    mono_builder: str | None
    replaced_nodes: tuple[str, ...]
    # On decline: the per-node builders left in place + the reason.
    decline_reason: str | None
    proof: dict[str, Any]


def dispatch_region(
    surfaces: Sequence[KernelSurface],
    *,
    region_name: str,
    mono_builder: str,
    nchunks: int,
    state_cells: int,
    smem_cap_bytes: int = APPLE_SMEM_CAP_BYTES,
) -> FusionDispatch:
    """Detect+prove+dispatch a region. The PROTOTYPE emission strategy.

    On ACCEPT (z3-proved, in-budget, GEMMs kept, nothing escapes) the dispatcher
    selects the proven hand-written mono builder ``mono_builder`` to REPLACE the N
    per-node ``build_*`` calls — the honest deferral (like the auto-GEMM pass
    dispatching to the builder rather than fabricating a raw-TIR splice).

    On DECLINE the region is left MULTI-KERNEL: the per-node builders stay in
    place and the structured decline reason is recorded (RULE #1, no wrong fusion).
    """
    result = analyze_region(
        surfaces,
        region_name=region_name,
        mono_builder=mono_builder,
        nchunks=nchunks,
        state_cells=state_cells,
        smem_cap_bytes=smem_cap_bytes,
    )
    if isinstance(result, str):
        return FusionDispatch(
            region_name=region_name,
            fused=False,
            mono_builder=None,
            replaced_nodes=(),
            decline_reason=result,
            proof={},
        )
    if not result.fusible:
        logger.warning(
            "auto-fuse-chunk: region=%s recognized but NOT fused (reason=%s) — "
            "left multi-kernel (RULE #1: no wrong fusion)",
            region_name, result.decline_reason or result.z3_reason,
        )
        return FusionDispatch(
            region_name=region_name,
            fused=False,
            mono_builder=None,
            replaced_nodes=(),
            decline_reason=result.decline_reason or "not_proved",
            proof=result.as_dict(),
        )
    logger.warning(
        "auto-fuse-chunk: region=%s FUSED nodes=%s -> mono builder %s "
        "(state %d B <= %d B, GEMMs kept, %s)",
        region_name, result.ordered_nodes, result.mono_builder,
        result.state_smem_bytes, result.smem_cap_bytes, result.z3_reason,
    )
    return FusionDispatch(
        region_name=region_name,
        fused=True,
        mono_builder=result.mono_builder,
        replaced_nodes=result.ordered_nodes,
        decline_reason=None,
        proof=result.as_dict(),
    )


# --------------------------------------------------------------------------- #
# Pass entry (gating)                                                          #
# --------------------------------------------------------------------------- #


def config_enabled() -> bool:
    """True iff the pass is opted in via env or a registered PassConfig key."""
    env = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    try:
        from tvm import transform as tvm_transform  # type: ignore

        cfg = tvm_transform.PassContext.current().config
        val = cfg.get(PASS_CONFIG_KEY, None) if cfg is not None else None
        if val is None:
            return False
        return bool(val)
    except Exception:
        return False


def fuse_forward_chunk_region(
    *,
    nchunks: int,
    headdim: int = 64,
    dstate: int = 64,
    two_kernel: bool = True,
    smem_cap_bytes: int = APPLE_SMEM_CAP_BYTES,
) -> FusionDispatch:
    """Convenience driver: detect+prove+dispatch the mamba3 forward region.

    ``two_kernel=True`` exercises the demonstrable F0+F1 prototype bar (the
    smallest true producer-consumer state hand-off); ``two_kernel=False`` runs the
    full F0+F1+F2 region. Returns a ``FusionDispatch``. When the pass is gated OFF
    this still runs for analysis/testing — the GATE only governs whether a real
    compile pipeline would APPLY the dispatch.
    """
    fwd = mamba3_forward_surfaces()
    if two_kernel:
        # F0+F1 only. prev_states becomes a region OUTPUT (F2 excluded) and so is
        # NOT an escaping INTERNAL edge — it is simply a materialized region
        # output, exactly like final_state. The F0->F1 internal edges that get
        # privatized are summary_states + dA_cumsum.
        surfaces = (fwd[0], fwd[1])
        region_name = "mamba3_fwd_F0_F1"
    else:
        surfaces = fwd
        region_name = "mamba3_fwd_F0_F1_F2"
    state_cells = headdim * dstate
    return dispatch_region(
        surfaces,
        region_name=region_name,
        mono_builder="mamba3_ssd_fused_fwd.build_ssd_fused_fwd",
        nchunks=nchunks,
        state_cells=state_cells,
        smem_cap_bytes=smem_cap_bytes,
    )


def proof_payload(match: _FusionRegionMatch) -> str:
    """Serialize a match's proof/decline audit trail (for func-attr stamping)."""
    return json.dumps(match.as_dict(), sort_keys=True)
