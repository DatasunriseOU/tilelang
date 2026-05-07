"""``tl.extern_intrinsic`` — declare raw ``__device__`` sources as tile-typed
TIR ops (RFC §6).

This module implements the user-facing decorator and the :class:`Frag` data
type that capture the **tile contract** of an externally-authored CUDA / HIP /
Metal kernel snippet so that existing TileLang fusion passes (
``auto_double_buffer``, ``thread_storage_sync``, ``inject_pipeline``,
``layout_inference``) treat the call as a regular TIR block — i.e. without
forcing an HBM round-trip.

The decorator returns a callable that, when invoked from a kernel definition,
emits a TIR ``call_extern`` whose first argument is the registered intrinsic
name and whose subsequent arguments are typed buffer ``access_ptr`` handles in
the declared scopes. The body source string itself is stashed in the global
:mod:`tilelang.language.extern_registry` and is materialised by the codegen
backend at lowering time.

Fusion-safety invariants (the framework does NOT statically verify these — they
are user contract):

1. **No implicit barriers.** The body must not call ``__syncthreads`` /
   ``threadgroup_barrier`` / ``s_barrier``. Barrier insertion is the job of
   ``src/transform/thread_storage_sync.cc``; an implicit barrier inside the
   body breaks the deadlock-freedom invariant the pipeliner relies on.

2. **No undeclared memory access.** The body may only read or write the
   buffers passed in via the declared :class:`Frag` arguments. Reading from a
   neighbouring shared-memory tile not in the contract will not be visible to
   ``layout_inference`` or ``thread_storage_sync``.

3. **No global memory.** Extern bodies must not dereference raw pointers into
   global / device memory. All HBM traffic must remain at the outer fusion
   boundary, otherwise cross-source fusion collapses to two launches.

4. **Pure on declared scope.** Side effects beyond the declared frag writes
   (e.g. mutating ``__shared__`` slots not in the contract, atomic-add to
   global counters) defeat ``auto_double_buffer`` and may corrupt pipelined
   versions of the same op.

5. **No host-only types.** Types in the declared :class:`Frag` must be the
   tile-typed dtypes TileLang itself emits (``"float16"``, ``"bfloat16"``,
   ``"float32"``, ``"int8"``, ``"int32"``, ``"e4m3"``, ``"e5m2"``, ``"e8m0"``,
   etc.). Host-side ``half2`` / ``__nv_bfloat162`` packed types are not
   recognised by ``layout_inference``.

Citations:
    - RFC: ``RFC_unified_fused_kernel.md`` §6 (cross-source extern intrinsic).
    - TIR call entry point: see :func:`tilelang.language.tir.op.call_extern`.
    - Existing pattern reference: :mod:`tilelang.language.customize` (``dp4a``
      uses the same ``call_extern + access_ptr`` shape we emit here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Tuple

from . import extern_registry as _registry

# ---------------------------------------------------------------------------
# Frag: tile-typed contract for one input/output of an extern intrinsic.
# ---------------------------------------------------------------------------

# Layout strings we accept. Anything that affects what `layout_inference.cc`
# would do for a regular TileLang op must be expressible here. The
# ``simdgroup_*`` layouts are the Metal SIMDgroup matrix-fragment layouts
# matching ``src/op/`` and ``tilelang/transform/metal_fragment_to_simdgroup.py``;
# the ``mma_*`` ones map to Volta+ wmma/HMMA fragment layouts.
LayoutKind = Literal[
    "row_major",
    "col_major",
    "swizzled_xor",
    "mma_A",
    "mma_B",
    "mma_C",
    "simdgroup_a",
    "simdgroup_b",
    "simdgroup_c",
]

ScopeKind = Literal["global", "shared", "shared.dyn", "local", "wmma", "simdgroup"]

_VALID_LAYOUTS: frozenset[str] = frozenset(LayoutKind.__args__)  # type: ignore[attr-defined]
_VALID_SCOPES: frozenset[str] = frozenset(ScopeKind.__args__)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Frag:
    """Tile-typed declaration of one extern-intrinsic argument.

    Resolved (RFC §8 Q7) — the simple ``tl.frag(name, shape, scope, dtype)``
    contract is too thin: ``layout_inference`` needs the layout, the codegen
    needs alignment for vectorised loads, and the pipeliner needs a stage
    hint. We add three fields beyond the minimal RFC stub.

    Attributes:
        name: Argument name as it appears in the user's body source.
        shape: Tile shape, all dims static ints (symbolic shapes are deferred).
        scope: Storage scope (``"shared"`` / ``"local"`` / ``"global"`` / etc.).
        dtype: TVM dtype string (e.g. ``"float16"``).
        layout: Hardware-fragment layout. Default ``"row_major"``.
        alignment: Byte-alignment guarantee, used by codegen for vector loads.
        pipeline_stage: Hint for ``inject_pipeline``; ``-1`` means "don't pin".
        is_output: True if extern writes to this frag; False if read-only.
    """

    name: str
    shape: Tuple[int, ...]
    scope: str
    dtype: str
    layout: str = "row_major"
    alignment: int = 16
    pipeline_stage: int = -1
    is_output: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Frag.name must be a valid identifier; got {self.name!r}")
        if not self.shape or any((not isinstance(d, int)) or d <= 0 for d in self.shape):
            raise ValueError(f"Frag.shape must be a non-empty tuple of positive ints; got {self.shape!r}")
        if self.scope not in _VALID_SCOPES:
            raise ValueError(
                f"Frag.scope={self.scope!r} not in {sorted(_VALID_SCOPES)}; "
                "extend extern.py if you need a new scope kind."
            )
        if self.layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"Frag.layout={self.layout!r} not in {sorted(_VALID_LAYOUTS)}"
            )
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(f"Frag.alignment must be a positive power of two; got {self.alignment}")


# ---------------------------------------------------------------------------
# Body validation: scrape the body string for the declared function and check
# its arity against the Frag tuple. Done at registration time so the user gets
# a clear error before they hit the codegen path.
# ---------------------------------------------------------------------------

# Match a __device__ function declaration (CUDA / HIP). Captures the parameter
# list. Metal uses [[kernel]] / inline functions; we additionally accept a bare
# function definition with the registered name.
_DEVICE_FN_RE = re.compile(
    r"(?:__device__|__attribute__\(\(device\)\)|inline\s+__device__|static\s+__device__)\s+"
    r"[\w:\s\*&<>,]+?\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)
# Metal: bodies are typically "void name(threadgroup half *a, ...) { ... }" or
# rely on simdgroup_matrix types. Looser regex; we only sanity-check the arg
# count.
_METAL_FN_RE = re.compile(
    r"(?:inline\s+)?(?:void|[\w:]+)\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)


def _split_args(arglist: str) -> list[str]:
    """Split a C-style argument list, ignoring commas inside angle brackets."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in arglist:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                out.append(arg)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _validate_body(target: str, intrinsic_name: str, body: str, frags: Sequence[Frag]) -> None:
    """Lightly validate the body string against the declared contract.

    We only check arity (parameter count) — full type-matching is impractical
    across CUDA / HIP / Metal. The named function must appear; if it does, its
    arg count must equal ``len(frags)``.
    """
    pat = _METAL_FN_RE if target == "metal" else _DEVICE_FN_RE
    matches = list(pat.finditer(body))
    found = [m for m in matches if m.group(1) == intrinsic_name]
    if not found:
        # Be tolerant: many users may inline the body without a matching name.
        # Only error if no function definition appears at all.
        if not matches:
            raise ValueError(
                f"extern_intrinsic[{target}] body for '{intrinsic_name}' contains "
                f"no recognisable function definition; expected '{intrinsic_name}(...)'."
            )
        return
    args = _split_args(found[0].group(2))
    if len(args) != len(frags):
        raise ValueError(
            f"extern_intrinsic[{target}] '{intrinsic_name}' declares "
            f"{len(args)} body parameter(s), but the signature declares "
            f"{len(frags)} Frag(s). Body args: {args!r}; "
            f"Frags: {[f.name for f in frags]!r}."
        )


# ---------------------------------------------------------------------------
# TIR call attribute scheme.
#
# We emit:
#   tir.call_extern("handle", "tl.extern_intrinsic.<name>", access_ptr(arg0), ...)
#
# The backend codegen looks up the body via
# ``tilelang.language.extern_registry.lookup("<name>")``. The intrinsic prefix
# ``tl.extern_intrinsic.`` lets reviewers grep for every emission site.
# ---------------------------------------------------------------------------

EXTERN_CALL_PREFIX: str = "tl.extern_intrinsic."
"""Symbol prefix for emitted ``call_extern`` ops; grep target for reviewers."""

EXTERN_BLOCK_ATTR: str = "tl.extern_intrinsic_meta"
"""``BlockNode`` annotation key carrying the intrinsic metadata (frags + stage)."""


# ---------------------------------------------------------------------------
# Meta serialisation helpers.
#
# ``layout_inference.cc`` and ``inject_pipeline.cc`` look for the
# :data:`EXTERN_BLOCK_ATTR` block annotation. The Python decorator does not
# create the enclosing block itself (the user is responsible for the
# ``T.block_attr({EXTERN_BLOCK_ATTR: ...})`` call near the ``call_extern``);
# however, we centralise the dict shape here so callers cannot drift from the
# C++ side. The dict carries:
#
#   - ``layouts``: per-frag layout name (``Array<String>``).
#   - ``tile_size``: shape of the *first* output fragment (``Array<Int>``),
#     used by ``layout_inference.cc`` to dispatch onto the tile-parameterised
#     ``makeGemmFragment{A,B,C}`` factories in ``src/layout/gemm_layouts.cc``.
#     For ``mma_*`` fragments we expect a 3-tuple ``(M, N, K)``; for
#     ``simdgroup_*`` fragments we expect a 2-tuple (typically ``(8, 8)``).
#   - ``pipeline_stage``: int hint for ``inject_pipeline.cc``; ``-1`` is a
#     no-op.
#   - ``is_output``: per-frag write/read-only flag (``Array<Int>`` / 0|1).
# ---------------------------------------------------------------------------


def build_meta(frags: Sequence[Frag], pipeline_stage: int = -1) -> dict:
    """Serialise the per-Frag contract into the dict consumed by the C++
    passes via the ``EXTERN_BLOCK_ATTR`` block annotation.

    The returned dict is shape-stable; callers should pass it verbatim to
    ``T.block_attr({EXTERN_BLOCK_ATTR: build_meta(frags)})``.

    The ``tile_size`` entry is derived from ``Frag.shape`` of the first
    *output* Frag (or the first Frag if none is marked output) — this is the
    accumulator/result tile, which is the natural anchor for the
    ``makeGemmFragment{A,B,C}`` factories in
    :file:`src/layout/gemm_layouts.cc`.
    """
    layouts = [f.layout for f in frags]
    is_output = [int(f.is_output) for f in frags]
    anchor = next((f for f in frags if f.is_output), frags[0] if frags else None)
    tile_size = list(anchor.shape) if anchor is not None else []
    return {
        "layouts": layouts,
        "tile_size": tile_size,
        "pipeline_stage": int(pipeline_stage),
        "is_output": is_output,
    }


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def _check_signature(name: str, frags: Sequence[Frag]) -> None:
    if not frags:
        raise ValueError(f"extern_intrinsic '{name}' signature returned no Frags.")
    seen: set[str] = set()
    for f in frags:
        if f.name in seen:
            raise ValueError(f"extern_intrinsic '{name}' has duplicate Frag name {f.name!r}.")
        seen.add(f.name)


def extern_intrinsic(
    name: str,
    signature: Callable[..., Iterable[Frag]],
    bodies: Mapping[str, str],
) -> Callable[..., Any]:
    """Decorator that registers ``bodies`` as a tile-typed extern intrinsic.

    Args:
        name: Globally-unique intrinsic name (becomes the ``call_extern`` symbol).
        signature: Callable that, given runtime shape parameters, returns the
            tuple of :class:`Frag` describing the contract.
        bodies: Mapping ``"cuda"`` / ``"hip"`` / ``"metal"`` -> body source.

    Returns:
        A callable that, when invoked from inside a TileLang kernel body,
        emits a TIR ``call_extern`` with typed buffer arguments.

    Per-Frag metadata is serialised by :func:`build_meta` for consumption
    by ``layout_inference.cc`` / ``inject_pipeline.cc``. The serialised dict
    additionally carries ``tile_size`` derived from ``Frag.shape`` of the
    output (or first) frag — this lets the C++ side dispatch onto the
    tile-parameterised ``makeGemmFragment{A,B,C}`` factories. The decorator
    itself only emits the ``call_extern``; the user is responsible for
    wrapping the call site in a ``T.block_attr({EXTERN_BLOCK_ATTR: ...})``
    block when fragment-layout inference is desired.

    Example (see ``poc/extern_intrinsic_examples/simdgroup_mma.py``)::

        @tl.extern_intrinsic(
            name="simdgroup_mma_8x8",
            signature=lambda: (
                Frag("a", (8, 8), "simdgroup", "float16", layout="simdgroup_a"),
                Frag("b", (8, 8), "simdgroup", "float16", layout="simdgroup_b"),
                Frag("c", (8, 8), "simdgroup", "float32", layout="simdgroup_c",
                     is_output=True),
            ),
            bodies={"metal": MSL_SOURCE},
        )
        def simdgroup_mma_8x8(): ...
    """
    if not isinstance(name, str) or not name:
        raise ValueError("extern_intrinsic: 'name' must be a non-empty string.")
    if not callable(signature):
        raise TypeError("extern_intrinsic: 'signature' must be callable.")
    if not isinstance(bodies, Mapping) or not bodies:
        raise ValueError("extern_intrinsic: 'bodies' must be a non-empty mapping.")

    valid = _registry.valid_targets()
    for tgt in bodies:
        if tgt not in valid:
            raise ValueError(
                f"extern_intrinsic '{name}': unknown target {tgt!r}; "
                f"valid targets are {sorted(valid)}."
            )

    # Eagerly probe the signature with no args if it accepts none, just to
    # surface obvious typos at registration time.
    probe_frags: tuple[Frag, ...] | None = None
    try:
        # Best-effort zero-arg probe — only if the signature supports it.
        probe = signature()
        probe_frags = tuple(probe)
        _check_signature(name, probe_frags)
        for tgt, body in bodies.items():
            _validate_body(tgt, name, body, probe_frags)
    except TypeError:
        # signature requires runtime shape args — defer validation to call time.
        probe_frags = None

    intrinsic = _registry.ExternIntrinsic(
        name=name,
        signature=lambda *a, **kw: tuple(signature(*a, **kw)),
        bodies=dict(bodies),
    )
    _registry.register(intrinsic)

    def _emit(*runtime_args: Any, **runtime_kwargs: Any) -> Any:
        """Resolve the signature with shape args and emit the TIR call."""
        frags = tuple(intrinsic.signature(*runtime_args, **runtime_kwargs))
        _check_signature(name, frags)
        if probe_frags is None:
            for tgt, body in bodies.items():
                _validate_body(tgt, name, body, frags)
        return _emit_tir_call(intrinsic.name, frags, runtime_args, runtime_kwargs)

    _emit.__name__ = f"extern_intrinsic_{name}"
    _emit.__doc__ = f"Emit TIR call_extern for registered intrinsic {name!r}."
    _emit.intrinsic = intrinsic  # type: ignore[attr-defined]
    # Helper: resolve the signature with shape args and return the
    # ``EXTERN_BLOCK_ATTR`` payload (see :func:`build_meta`). Users wire this
    # into a sibling ``T.block_attr({EXTERN_BLOCK_ATTR: emit.meta(...)})``.
    def _meta(*runtime_args: Any, pipeline_stage: int = -1, **runtime_kwargs: Any) -> dict:
        frags = tuple(intrinsic.signature(*runtime_args, **runtime_kwargs))
        return build_meta(frags, pipeline_stage=pipeline_stage)
    _emit.meta = _meta  # type: ignore[attr-defined]
    return _emit


def _emit_tir_call(
    name: str,
    frags: Sequence[Frag],
    runtime_args: tuple[Any, ...],
    runtime_kwargs: Mapping[str, Any],
) -> Any:
    """Build the TIR ``call_extern`` node. TVM imports are lazy so this module
    can be imported without TVM installed (registration-only flows)."""
    # Lazy imports: keep top-level import-free for testability.
    from tvm import tir  # noqa: WPS433

    import tilelang.language as T  # noqa: WPS433  # circular-safe at call time

    # Buffers come either positionally after the shape args or by keyword.
    # We accept both: ``intrin(M, N, A, B, C)`` or ``intrin(A, B, C)`` (no shape
    # args needed when the signature is zero-arg).
    buffers = [
        a for a in runtime_args
        if hasattr(a, "data") and hasattr(a, "dtype") and hasattr(a, "shape")
    ]
    buffers += [
        v for v in runtime_kwargs.values()
        if hasattr(v, "data") and hasattr(v, "dtype") and hasattr(v, "shape")
    ]
    if len(buffers) != len(frags):
        raise ValueError(
            f"extern_intrinsic '{name}': received {len(buffers)} buffer arg(s) "
            f"but contract declares {len(frags)} Frag(s) ({[f.name for f in frags]})."
        )

    access_ptrs = []
    for buf, frag in zip(buffers, frags):
        mode = "rw" if frag.is_output else "r"
        access_ptrs.append(T.access_ptr(buf, mode))

    # Symbol name embeds the registered intrinsic name; codegen greps for
    # ``EXTERN_CALL_PREFIX`` to dispatch on the registry.
    symbol = f"{EXTERN_CALL_PREFIX}{name}"
    return tir.call_extern("handle", symbol, *access_ptrs)


__all__ = [
    "Frag",
    "LayoutKind",
    "ScopeKind",
    "extern_intrinsic",
    "build_meta",
    "EXTERN_CALL_PREFIX",
    "EXTERN_BLOCK_ATTR",
]
