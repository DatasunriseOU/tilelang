"""``tl.extern_intrinsic`` — declare raw ``__device__`` sources as tile-typed
TIR ops (RFC §6).

This module implements the user-facing decorator and the :class:`Frag` data
type that capture the **tile contract** of an externally-authored CUDA / HIP /
Metal / CuTeDSL kernel snippet so that existing TileLang fusion passes (
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

.. warning::

   **Trusted-bodies-only contract.** Body strings supplied via ``bodies=`` are
   emitted verbatim into the final GPU kernel; downstream codegen MUST treat
   them as literal text without further string interpolation. They must come
   from trusted developer code. Arbitrary or attacker-controlled bodies can
   compromise the entire kernel (RCE on the GPU, driver crashes, silent data
   corruption). The framework deliberately does NOT sandbox or sanitise
   bodies — that is the price of letting the DSL drop down to raw device
   code. See RFC §6 for the discussion behind this decision.

Citations:
    - RFC: ``RFC_unified_fused_kernel.md`` §6 (cross-source extern intrinsic).
    - TIR call entry point: see :func:`tilelang.language.tir.op.call_extern`.
    - Existing pattern reference: :mod:`tilelang.language.customize` (``dp4a``
      uses the same ``call_extern + access_ptr`` shape we emit here).
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
    "simdgroup_a_fp8",
    "simdgroup_b_fp8",
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
    shape: tuple[int, ...]
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


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    path: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        path.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        path.append(cur.id)
    return tuple(reversed(path))


def _cutedsl_kernel_decorator_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    module_aliases: set[str] = {"cutlass.cute"}
    kernel_aliases: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cutlass.cute":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "cutlass":
                for alias in node.names:
                    if alias.name == "cute":
                        module_aliases.add(alias.asname or alias.name)
            elif node.module == "cutlass.cute":
                for alias in node.names:
                    if alias.name == "kernel":
                        kernel_aliases.add(alias.asname or alias.name)

    return module_aliases, kernel_aliases


def _is_cutedsl_kernel_decorator(
    node: ast.AST,
    module_aliases: set[str],
    kernel_aliases: set[str],
) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    path = _attribute_path(node)
    if len(path) == 1:
        return path[0] in kernel_aliases
    if len(path) >= 2 and path[-1] == "kernel":
        return ".".join(path[:-1]) in module_aliases
    return False


def _cutedsl_kernel_functions(
    body: str,
    intrinsic_name: str,
    target: str,
) -> list[ast.FunctionDef]:
    try:
        tree = ast.parse(body)
    except SyntaxError as err:
        raise ValueError(
            f"extern_intrinsic[{target}] body for '{intrinsic_name}' is not valid "
            f"Python CuTeDSL source: {err.msg}."
        ) from err

    module_aliases, kernel_aliases = _cutedsl_kernel_decorator_names(tree)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            _is_cutedsl_kernel_decorator(
                decorator,
                module_aliases=module_aliases,
                kernel_aliases=kernel_aliases,
            )
            for decorator in node.decorator_list
        )
    ]


def _validate_cutedsl_body(
    target: str,
    intrinsic_name: str,
    body: str,
    frags: Sequence[Frag],
) -> None:
    matches = _cutedsl_kernel_functions(body, intrinsic_name, target)
    found = [node for node in matches if node.name == intrinsic_name]
    if not found:
        if not matches:
            raise ValueError(
                f"extern_intrinsic[{target}] body for '{intrinsic_name}' contains "
                f"no recognisable CuTeDSL @kernel definition; expected "
                f"'@kernel\\ndef {intrinsic_name}(...)' or "
                f"'@cute.kernel\\ndef {intrinsic_name}(...)'."
            )
        raise ValueError(
            f"extern_intrinsic[{target}] body for '{intrinsic_name}' defines "
            f"CuTeDSL kernel(s) {[node.name for node in matches]!r}, but lowering "
            f"will call '{intrinsic_name}(...)'. Expected '@cute.kernel\\ndef "
            f"{intrinsic_name}(...)' or '@kernel\\ndef {intrinsic_name}(...)'."
        )

    fn = found[0]
    args = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
    if len(args) != len(frags):
        raise ValueError(
            f"extern_intrinsic[{target}] '{intrinsic_name}' declares "
            f"{len(args)} body parameter(s), but the signature declares "
            f"{len(frags)} Frag(s). Body args: {[arg.arg for arg in args]!r}; "
            f"Frags: {[f.name for f in frags]!r}."
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


# Strip out comments + string literals before the parameter-name scan so a
# stray ``// uses A_frag`` / ``"a"`` doesn't satisfy the contract check.
#
# Order matters: raw strings (``R"tag(...)tag"``) must be removed BEFORE the
# regular-string regex sees them, otherwise the embedded ``"`` confuses the
# escape-aware matcher and we'd leak fragment-name-looking identifiers from
# the raw-string body. Raw strings are common in Apple MSL bodies that embed
# multi-line shader source.
_C_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_C_STRING_RE = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
# C++11 raw string literals: R"delim(...)delim" or u8R"...". The delimiter is
# captured (back-referenced) so we match the matching closing ``)delim"``.
_C_RAW_STRING_RE = re.compile(r"(?:u8|u|U|L)?R\"([^\s()\\]{0,16})\((?:.|\n)*?\)\1\"")


def _scrub_body_for_name_scan(body: str) -> str:
    """Drop comments + raw-string + string literals before the name scan.

    Raw strings are scrubbed first because their internal ``"`` would otherwise
    desync the escape-aware regular-string matcher and cause it to either miss
    a closing quote or eat unrelated source. Without this, an extern body with
    a raw-string-embedded MSL shader could hide / fake fragment-name tokens.
    """
    body = _C_RAW_STRING_RE.sub(" ", body)
    body = _C_COMMENT_RE.sub(" ", body)
    body = _C_STRING_RE.sub(" ", body)
    return body


def _validate_body(target: str, intrinsic_name: str, body: str, frags: Sequence[Frag]) -> None:
    """Lightly validate the body string against the declared contract.

    Two checks:

    1. **Arity** — if the named function definition appears, its parameter
       count must equal ``len(frags)``. Full C type matching is impractical
       across CUDA / HIP / Metal, so we only count.
    2. **Parameter-name matching** (best-effort) — every declared
       :class:`Frag` ``name`` must appear at least once as a word-boundary
       token in the body source (after stripping comments and string
       literals). A missing Frag name is the most common typo failure mode
       and is reported via :class:`UserWarning`; we don't escalate to an
       error because some users hand-roll the body in an unusual style
       (e.g. typedef'd structs, macro-expanded names) where a literal name
       match would false-fire.

    We don't do the reverse "names that look like a fragment but aren't
    declared" check because it would false-fire on every helper local
    variable in the body.
    """
    if target == "cutedsl":
        _validate_cutedsl_body(target, intrinsic_name, body, frags)
    else:
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
        else:
            args = _split_args(found[0].group(2))
            if len(args) != len(frags):
                raise ValueError(
                    f"extern_intrinsic[{target}] '{intrinsic_name}' declares "
                    f"{len(args)} body parameter(s), but the signature declares "
                    f"{len(frags)} Frag(s). Body args: {args!r}; "
                    f"Frags: {[f.name for f in frags]!r}."
                )

    scrubbed = _scrub_body_for_name_scan(body)
    missing: list[str] = []
    for f in frags:
        if not re.search(rf"\b{re.escape(f.name)}\b", scrubbed):
            missing.append(f.name)
    if missing:
        warnings.warn(
            f"extern_intrinsic[{target}] '{intrinsic_name}': declared Frag name(s) "
            f"{missing!r} do not appear in the body source. This is usually a typo "
            f"between the signature and the body; the kernel will still register "
            f"but the body cannot reference frags whose names it doesn't know.",
            UserWarning,
            stacklevel=3,
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
        bodies: Mapping ``"cuda"`` / ``"hip"`` / ``"metal"`` / ``"cutedsl"`` -> body source.

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
        """Resolve the signature with shape args and emit the TIR call.

        ``signature`` only sees shape args — buffer-like positional args (and
        any buffer-valued kwargs) are stripped first so a user can write the
        natural ``intrin(M, N, A_frag, B_frag, C_frag)`` and the shape factory
        ``lambda M, N: ...`` still receives the right tuple. Without this
        split, the signature factory would receive Buffer objects and either
        raise ``TypeError`` or silently produce wrong frags (perf review
        finding #1).
        """
        # If invoked by Python's `@` syntax as `@extern_intrinsic(...)`, this
        # function receives the decorated function as its sole argument. We
        # return ourselves to replace the user's stub with this emitter.
        if len(runtime_args) == 1 and not runtime_kwargs and callable(runtime_args[0]) and not _looks_like_buffer(runtime_args[0]):
            func = runtime_args[0]
            _emit.__name__ = func.__name__
            _emit.__doc__ = func.__doc__ or _emit.__doc__
            return _emit

        shape_args, buffer_args = _split_shape_and_buffer_args(runtime_args)
        shape_kwargs, buffer_kwargs = _split_shape_and_buffer_kwargs(runtime_kwargs)
        frags = tuple(intrinsic.signature(*shape_args, **shape_kwargs))
        _check_signature(name, frags)
        if probe_frags is None:
            for tgt, body in bodies.items():
                _validate_body(tgt, name, body, frags)
        return _emit_tir_call(intrinsic.name, frags, buffer_args, buffer_kwargs)

    _emit.__name__ = f"extern_intrinsic_{name}"
    _emit.__doc__ = f"Emit TIR call_extern for registered intrinsic {name!r}."
    _emit.intrinsic = intrinsic  # type: ignore[attr-defined]
    # Helper: resolve the signature with shape args and return the
    # ``EXTERN_BLOCK_ATTR`` payload (see :func:`build_meta`). Users wire this
    # into a sibling ``T.block_attr({EXTERN_BLOCK_ATTR: emit.meta(...)})``.
    def _meta(*runtime_args: Any, pipeline_stage: int = -1, **runtime_kwargs: Any) -> dict:
        shape_args, _ = _split_shape_and_buffer_args(runtime_args)
        shape_kwargs, _ = _split_shape_and_buffer_kwargs(runtime_kwargs)
        frags = tuple(intrinsic.signature(*shape_args, **shape_kwargs))
        return build_meta(frags, pipeline_stage=pipeline_stage)
    _emit.meta = _meta  # type: ignore[attr-defined]
    return _emit


def _looks_like_buffer(obj: Any) -> bool:
    """Heuristic: True if ``obj`` quacks like a TIR buffer.

    Used to separate shape-parameter args (ints, sym vars) from buffer args
    in the signature/emit split. We cannot ``isinstance(obj, tir.Buffer)``
    here without forcing a TVM import at module load — keep duck-typing.

    The check requires TIR-specific attributes (``access_ptr`` or ``scope``)
    beyond the generic ``(data, dtype, shape)`` triple to avoid false
    positives on numpy arrays, torch tensors, and pandas DataFrames — all
    of which also carry ``data``, ``dtype``, and ``shape``.
    """
    if not (hasattr(obj, "data") and hasattr(obj, "dtype") and hasattr(obj, "shape")):
        return False
    # TIR-specific discriminators: tir.Buffer exposes access_ptr() and scope().
    if hasattr(obj, "access_ptr") or hasattr(obj, "scope"):
        return True
    # Fallback: if .data carries .type_annotation it's a TVM Var, not a
    # numpy/torch data attribute.
    data = getattr(obj, "data", None)
    return data is not None and hasattr(data, "type_annotation")


def _split_shape_and_buffer_args(
    runtime_args: tuple[Any, ...],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Partition positional args into (shape_args, buffer_args).

    Order is preserved within each group so the user-facing call shape
    ``intrin(M, N, A_frag, B_frag, C_frag)`` produces ``shape=(M, N)`` and
    ``buffers=(A_frag, B_frag, C_frag)``. Mixed orderings (buffers between
    shape args) preserve the buffer-vs-shape classification but lose the
    original interleaving, which is fine because shape factories don't see
    buffers and ``_emit_tir_call`` matches buffers by Frag order.
    """
    shape_args = tuple(a for a in runtime_args if not _looks_like_buffer(a))
    buffer_args = tuple(a for a in runtime_args if _looks_like_buffer(a))
    return shape_args, buffer_args


def _split_shape_and_buffer_kwargs(
    runtime_kwargs: Mapping[str, Any],
) -> tuple[dict, dict]:
    """Same partition for kwargs. Returns ``(shape_kwargs, buffer_kwargs)``."""
    shape_kwargs: dict = {}
    buffer_kwargs: dict = {}
    for k, v in runtime_kwargs.items():
        if _looks_like_buffer(v):
            buffer_kwargs[k] = v
        else:
            shape_kwargs[k] = v
    return shape_kwargs, buffer_kwargs


def _emit_tir_call(
    name: str,
    frags: Sequence[Frag],
    buffer_args: tuple[Any, ...],
    buffer_kwargs: Mapping[str, Any],
) -> Any:
    """Build the TIR ``call_extern`` node.

    Inputs are already pre-filtered by the caller:

    - ``buffer_args``: positional buffer-likes in user-supplied order.
    - ``buffer_kwargs``: keyword buffers; matched by ``Frag.name``.

    Mixing the two is allowed but kwargs win when a frag-name appears in both
    (this disambiguates the otherwise-brittle positional zip flagged by the
    security review). TVM imports are lazy so this module can be imported
    without TVM installed (registration-only flows).
    """
    # Lazy imports: keep top-level import-free for testability.
    from tvm import tir  # noqa: WPS433

    import tilelang.language as T  # noqa: WPS433  # circular-safe at call time

    # Resolve the per-frag buffer:
    #   1. If a kwarg matches the frag name, use it.
    #   2. Else fall back to the next positional buffer (preserving order).
    by_name: dict = dict(buffer_kwargs)
    positional = list(buffer_args)
    resolved: list[Any] = []
    for frag in frags:
        if frag.name in by_name:
            resolved.append(by_name.pop(frag.name))
        elif positional:
            resolved.append(positional.pop(0))
        else:
            raise ValueError(
                f"extern_intrinsic '{name}': missing buffer for Frag {frag.name!r} "
                f"(declared frags: {[f.name for f in frags]!r})."
            )
    if positional or by_name:
        leftover = [type(b).__name__ for b in positional] + sorted(by_name)
        raise ValueError(
            f"extern_intrinsic '{name}': received unexpected buffer arg(s) "
            f"{leftover!r}; contract declares {[f.name for f in frags]!r}."
        )

    access_ptrs = []
    for buf, frag in zip(resolved, frags):
        mode = "rw" if frag.is_output else "r"
        access_ptrs.append(T.access_ptr(buf, mode))

    # Symbol name embeds the registered intrinsic name; codegen greps for
    # ``EXTERN_CALL_PREFIX`` to dispatch on the registry.
    symbol = f"{EXTERN_CALL_PREFIX}{name}"
    return tir.call_extern("handle", symbol, *access_ptrs)


# ---------------------------------------------------------------------------
# Canonical Frag factories for hardware-specific MMA fragment shapes.
#
# The grok security review (#08, "other bugs" #3) flagged that hand-rolling
# every Frag at the call site is error-prone — the same Apple SIMDgroup MMA
# tile shows up in every Metal MMA op and reviewers were drifting on default
# scope/dtype/alignment. The factories below pin the canonical defaults so
# users only override what they actually mean to change.
#
# Layout strings stay as the canonical ``simdgroup_a/b/c`` enum entries
# resolved by ``layout_inference.cc`` -> ``src/layout/gemm_layouts.cc``;
# concrete register-tile layouts are owned by the C++ side. The factories
# here only make the *contract* unambiguous.
#
# Note on the missing Fragment factory for ``simdgroup_*``: Apple's Metal
# Shading Language Specification §6.7.2 defines ``simdgroup_matrix<T,8,8>``
# as an opaque type whose per-thread element decomposition is
# implementation-defined. Loads/stores go through ``simdgroup_load`` /
# ``simdgroup_store``, which themselves manage the thread-element mapping;
# user code never indexes ``matrix[lane][elt]`` directly. Consequently
# ``layout_inference.cc`` returns an empty ``Layout()`` for the three
# ``simdgroup_*`` strings — there is no canonical thread/element mapping
# to encode. The Fragment factories below pin the *call-site contract*
# (shape (8, 8), scope=``"simdgroup"``, dtype defaults) but do NOT attempt
# to fabricate a per-lane Layout; that decision is correct and matches
# Apple's documented opacity. See also ``src/op/utils.h:61`` for the
# matching ``"metal.simdgroup"`` scope check that codegen relies on.
# ---------------------------------------------------------------------------


def _simdgroup_factory(
    layout: str,
    default_dtype: str,
    default_is_output: bool,
) -> Callable[..., Frag]:
    def _make(
        name: str,
        shape: tuple[int, int] = (8, 8),
        dtype: str = default_dtype,
        *,
        scope: str = "simdgroup",
        alignment: int = 16,
        pipeline_stage: int = -1,
        is_output: bool = default_is_output,
    ) -> Frag:
        if len(shape) != 2:
            raise ValueError(
                f"{layout} expects a 2-D tile shape; got {shape!r}"
            )
        return Frag(
            name=name,
            shape=tuple(shape),
            scope=scope,
            dtype=dtype,
            layout=layout,
            alignment=alignment,
            pipeline_stage=pipeline_stage,
            is_output=is_output,
        )
    _make.__name__ = layout
    _make.__qualname__ = layout
    return _make


simdgroup_a = _simdgroup_factory("simdgroup_a", default_dtype="float16", default_is_output=False)
"""Apple Metal SIMDgroup matrix-A operand factory.

>>> a = simdgroup_a("a")
>>> a.layout, a.shape, a.dtype, a.scope, a.is_output
('simdgroup_a', (8, 8), 'float16', 'simdgroup', False)
"""

simdgroup_b = _simdgroup_factory("simdgroup_b", default_dtype="float16", default_is_output=False)
"""Apple Metal SIMDgroup matrix-B operand factory.

>>> b = simdgroup_b("b")
>>> b.layout, b.shape, b.dtype, b.scope, b.is_output
('simdgroup_b', (8, 8), 'float16', 'simdgroup', False)
"""

simdgroup_c = _simdgroup_factory("simdgroup_c", default_dtype="float32", default_is_output=True)
"""Apple Metal SIMDgroup C-accumulator operand factory.

Defaults to ``is_output=True`` since SIMDgroup MMA always writes the C tile.

>>> c = simdgroup_c("c")
>>> c.layout, c.shape, c.dtype, c.scope, c.is_output
('simdgroup_c', (8, 8), 'float32', 'simdgroup', True)
"""


def _simdgroup_doctest() -> None:
    """End-to-end doctest for the 8-field Frag contract via the simdgroup_*
    factories. No execution; just exercises the metadata path that
    layout_inference.cc / inject_pipeline.cc consume.

    >>> a = simdgroup_a("a", pipeline_stage=0)
    >>> b = simdgroup_b("b", pipeline_stage=0)
    >>> c = simdgroup_c("c", pipeline_stage=1)
    >>> meta = build_meta((a, b, c), pipeline_stage=1)
    >>> meta["layouts"]
    ['simdgroup_a', 'simdgroup_b', 'simdgroup_c']
    >>> meta["tile_size"]
    [8, 8]
    >>> meta["is_output"]
    [0, 0, 1]
    >>> meta["pipeline_stage"]
    1
    """


# ---------------------------------------------------------------------------
# FP8 SIMDgroup MMA fragment factories — FORWARD-COMPATIBLE PLACEHOLDERS
# ---------------------------------------------------------------------------
# Apple has not (as of 2026-05) shipped ``simdgroup_matrix<float8_e4m3>`` /
# ``simdgroup_matrix<float8_e5m2>`` types in the MSL specification. Current
# Apple silicon (M3/M4) exposes simdgroup MMA only for fp16/bf16/fp32. The
# factories below pin the *call-site contract* — shape (8, 8), scope
# ``"simdgroup"``, default dtype ``"float8_e4m3"`` — so kernels in
# ``cppmega.mlx/_tilelang/{fp8_msl_kernels,sparse_mla_fp8,fp8_vecmat_path_c}.py``
# can drop in cleanly the moment Apple ships. Until then:
#
#  - the layout string ``simdgroup_a_fp8`` / ``simdgroup_b_fp8`` is unknown
#    to ``layout_inference.cc`` and falls through to the same empty
#    ``Layout()`` path that the existing fp16 factories already use — see
#    the comment block above ``_simdgroup_factory`` for why opacity is the
#    correct answer for SIMDgroup MMA in general (Apple's per-thread element
#    decomposition is implementation-defined per MSL §6.7.2).
#  - codegen will *not* synthesise an fp8 MMA call from a TileLang
#    intrinsic; FP8 paths still go through extern_intrinsic with an opaque
#    MSL body that includes whatever fp8 emulation Apple ships
#    (``mx_get_fp8_dot4`` / ``simd_sum`` software emulation today). The
#    factory just labels the operand role for the metadata pipeline.
#
# When Apple ships native FP8 simdgroup MMA the precise edits required:
#   1. Confirm dtype tokens in MSL spec (likely ``float8_e4m3`` / ``float8_e5m2``
#      to match Metal's official type names) — adjust ``default_dtype`` here.
#   2. Add register-tile decomposition to ``src/layout/gemm_layouts.cc`` and
#      have ``layout_inference.cc`` map ``simdgroup_{a,b}_fp8`` to it.
#   3. Add ``builtin::simdgroup_mma_fp8`` to ``src/op/builtin.cc`` and a
#      Metal codegen path in ``src/target/codegen_metal.cc``.
#   4. Flip the ``xfail`` marker on
#      ``test_simdgroup_fp8_factories_produce_canonical_frags_fp8_runtime``
#      in ``poc/extern_intrinsic_examples/test_extern_smoke.py``.
# ---------------------------------------------------------------------------

simdgroup_a_fp8 = _simdgroup_factory(
    "simdgroup_a_fp8", default_dtype="float8_e4m3", default_is_output=False,
)
"""Apple Metal SIMDgroup matrix-A operand factory — FP8 forward-compat.

Forward-compatible placeholder pending Apple FP8 silicon (see module
docstring). Layout is opaque (empty ``Layout()``), matching the canonical
fp16 ``simdgroup_a`` factory. ``dtype`` defaults to ``"float8_e4m3"``;
override to ``"float8_e5m2"`` for the unsigned-zero variant.

>>> a = simdgroup_a_fp8("a")
>>> a.layout, a.shape, a.dtype, a.scope, a.is_output
('simdgroup_a_fp8', (8, 8), 'float8_e4m3', 'simdgroup', False)
"""

simdgroup_b_fp8 = _simdgroup_factory(
    "simdgroup_b_fp8", default_dtype="float8_e4m3", default_is_output=False,
)
"""Apple Metal SIMDgroup matrix-B operand factory — FP8 forward-compat.

Forward-compatible placeholder pending Apple FP8 silicon (see module
docstring). Pair with ``simdgroup_c`` (fp32 accumulator) for the typical
FP8 → FP32 GEMM contract.

>>> b = simdgroup_b_fp8("b")
>>> b.layout, b.shape, b.dtype, b.scope, b.is_output
('simdgroup_b_fp8', (8, 8), 'float8_e4m3', 'simdgroup', False)
"""


__all__ = [
    "Frag",
    "LayoutKind",
    "ScopeKind",
    "extern_intrinsic",
    "build_meta",
    "EXTERN_CALL_PREFIX",
    "EXTERN_BLOCK_ATTR",
    "simdgroup_a",
    "simdgroup_b",
    "simdgroup_c",
    "simdgroup_a_fp8",
    "simdgroup_b_fp8",
]
