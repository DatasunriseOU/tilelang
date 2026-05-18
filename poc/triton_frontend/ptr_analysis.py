"""Python facade over microsoft/triton-shared ``mlir::tts::PtrAnalysis``.

Architecture decision (pure mlir-python-bindings vs C++ shim)
=============================================================
We use a thin **C++ pybind11 shim** (``poc/triton_frontend/_cxx/``) rather
than pure ``mlir-python-bindings``. Reasons:

1. ``PtrAnalysis`` is a stateful C++ class (``DenseMap<Value, PtrState>``,
   ``IRMapping``, ``OpBuilder``, recursion across ``scf.for`` regions) that
   is *not* a ``mlir::Pass`` -- it cannot be invoked through PassManager.
2. ``mlir-python-bindings`` exposes only upstream MLIR core + in-tree
   dialects. triton-shared's ``tts.*`` ops, ``GetStructuredStateOp``, and
   ``PtrState`` have no Python surface.
3. The vendored implementation (``vendored/triton_shared/...``, MIT-licensed,
   commit 08684f9, 2025-12-05) is meant to be re-used verbatim per RFC
   section 3. A C++ shim preserves correctness fixes from upstream.
4. The shim ABI is small (5 entry points, MLIR text in / text out), so the
   Python side stays import-friendly without a hard build dependency.

Vendoring policy (RFC section 8 question 1 + section 7 phase 1.1):
- Source path: ``poc/triton_frontend/vendored/triton_shared/``.
- License: MIT (Microsoft Corporation, Meta Platforms). Compatible with
  TileLang's Apache-2.0 license.
- Sibling integration #5 vendors the TritonStructured dialect that
  ``PtrAnalysis`` emits. Until that lands, ``rewrite()`` raises
  ``RuntimeError`` from the shim's ``TL_PA_ERR_INTERNAL`` path.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PtrState",
    "StridedLayout",
    "PtrAnalysis",
    "shim_available",
    "dialects_available",
    "run_ptr_analysis",
    "extract_ptr_states",
    "run_ptr_analysis_with_states",
    "run_ptr_analysis_with_states_generic",
    "SHIM_MODULE_NAME",
]


SHIM_MODULE_NAME = "_triton_frontend_cxx"
"""Name of the pybind11 extension built from ``poc/triton_frontend/_cxx/``."""


# Build directories the cmake configuration may have produced the shim into.
# Order matters: the orchestrator's "port" build (out-of-tree, cmake -B build-port)
# is checked first because that is where freshly-produced .so files land in the
# current pipeline; ``_cxx/build/`` is the legacy in-tree location used by
# ``build_cxx.ensure_built``.
_THIS_DIR = Path(__file__).resolve().parent
_SHIM_BUILD_DIRS: Tuple[Path, ...] = (
    _THIS_DIR / "_cxx" / "build-port",
    _THIS_DIR / "_cxx" / "build",
)


def _shim_dir_has_extension(d: Path) -> bool:
    if not d.is_dir():
        return False
    for entry in d.iterdir():
        name = entry.name
        if not name.startswith(SHIM_MODULE_NAME):
            continue
        if name.endswith((".so", ".dylib", ".pyd")):
            return True
    return False


def _ensure_shim_on_syspath() -> bool:
    """If a built shim exists in any known build dir, prepend it to sys.path.

    Returns True when a directory was added (or was already present) and the
    shim should now be importable.
    """
    for d in _SHIM_BUILD_DIRS:
        if _shim_dir_has_extension(d):
            ds = str(d)
            if ds not in sys.path:
                sys.path.insert(0, ds)
            importlib.invalidate_caches()
            return True
    return False


# Module-level latch so the "shim unavailable" warning fires at most once per
# process, no matter how many call sites poke shim_available(). The build_cxx
# helper is consulted lazily so importing this module never triggers cmake.
_SHIM_WARNED = False


def _shim_unavailable_warn_once() -> None:
    global _SHIM_WARNED
    if _SHIM_WARNED:
        return
    _SHIM_WARNED = True
    warnings.warn(
        "C++ PtrAnalysis shim unavailable; falling back to MVP scalar path. "
        "Run `python -m poc.triton_frontend.build_cxx --build` to enable "
        "multi-element tile loads.",
        RuntimeWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Cross-extension LLVM/MLIR conflict guard
# ---------------------------------------------------------------------------
# Both the ``_triton_frontend_cxx`` shim and upstream ``triton._C.libtriton``
# statically link their own copy of LLVM. Loading both into one Python process
# triggers ``LLVM ERROR: Option '<name>' already exists!`` (cl::opt duplicate
# registration) and aborts with SIGABRT mid-collection. Until the shim is
# rebuilt against a shared LLVM, we refuse to load the second of the two and
# emit a clean skip-warning instead of crashing the interpreter.
#
# Tests that depend on the shim are wired through ``dialects_available()`` /
# ``shim_available()``, so returning ``False`` here makes pytest skip the
# affected tests cleanly. Callers that go through ``_load_shim()`` directly
# (i.e. the runtime ``PtrAnalysis`` rewrite path) get a ``NotImplementedError``
# with a diagnostic instead of a hard abort.
_LLVM_CONFLICT_WARNED = False


def _triton_already_loaded() -> bool:
    """Return True iff Triton's native C extension is already in this process.

    We check for the LLVM-bearing C extension specifically (``triton._C`` /
    ``triton._C.libtriton``) rather than the pure-Python ``triton`` shell,
    because importing ``triton`` does not by itself pull in the conflicting
    LLVM static-init -- that only happens once ``libtriton`` is touched.
    """
    for mod_name in ("triton._C.libtriton", "triton._C"):
        if mod_name in sys.modules:
            return True
    return False


def _shim_conflict_warn_once() -> None:
    global _LLVM_CONFLICT_WARNED
    if _LLVM_CONFLICT_WARNED:
        return
    _LLVM_CONFLICT_WARNED = True
    warnings.warn(
        "PtrAnalysis C++ shim disabled in this process: "
        "Triton's libtriton (with its own LLVM) is already loaded. "
        "Both link LLVM statically and registering cl::opts twice aborts "
        "the interpreter, so the shim is skipped here. Run shim-dependent "
        "tests in a fresh Python process (or via pytest-forked / a "
        "subprocess wrapper) to exercise them.",
        RuntimeWarning,
        stacklevel=3,
    )


def shim_available() -> bool:
    """Return True if the C++ shim is importable from ``sys.path``.

    If a pre-built extension exists under ``_cxx/build/`` but isn't yet on
    ``sys.path``, this prepends the build dir so the next ``import`` succeeds.
    The cmake/ninja build itself is NEVER triggered from here -- callers must
    invoke ``python -m poc.triton_frontend.build_cxx --build`` explicitly.

    Emits a one-shot :class:`RuntimeWarning` when the shim is missing so the
    fallback to the MVP scalar path is observable in test logs without
    drowning them in repeated messages.

    When upstream Triton's ``libtriton`` is already loaded in this process
    we return ``False`` (with a one-shot warning) instead of letting the
    duplicate LLVM static-init abort the interpreter; the shim is still
    physically present on disk, just unsafe to import here.
    """
    # If the shim itself is already loaded we are safe regardless of triton.
    if SHIM_MODULE_NAME in sys.modules:
        return True
    if _triton_already_loaded():
        _shim_conflict_warn_once()
        return False
    if importlib.util.find_spec(SHIM_MODULE_NAME) is not None:
        return True
    # First try our local build-dir candidates (build-port, build).
    if _ensure_shim_on_syspath() and importlib.util.find_spec(SHIM_MODULE_NAME) is not None:
        return True
    # Fall back to the cmake build helper without shelling out to cmake itself.
    try:
        from . import build_cxx as _build_cxx
    except ImportError:  # pragma: no cover - build_cxx ships alongside us
        _build_cxx = None  # type: ignore[assignment]
    if _build_cxx is not None and _build_cxx.ensure_built(build=False, verbose=False):
        return True
    _shim_unavailable_warn_once()
    return False


def dialects_available() -> bool:
    """Return True if the shim was built against TritonStructured + Triton.

    When False, the shim still imports but ``rewrite()`` will raise because
    ``tl_pa_run_rewrite`` returns ``TL_PA_ERR_INTERNAL`` in stub mode. Tests
    that need a working rewrite should skip on this rather than just on
    :func:`shim_available`.
    """
    if not shim_available():
        return False
    try:
        mod = importlib.import_module(SHIM_MODULE_NAME)
    except ImportError:  # pragma: no cover
        return False
    return bool(getattr(mod, "dialects_available", False))


def _load_shim() -> Any:
    """Lazy-import the C++ shim or raise a NotImplementedError with build hint."""
    # Make sure local build-port / build dirs are on sys.path before importing.
    # This mirrors the discovery logic in :func:`shim_available` so callers
    # that go straight through the load helper don't trip over a stale
    # negative cache from an early ``find_spec`` miss.
    _ensure_shim_on_syspath()
    if SHIM_MODULE_NAME not in sys.modules and _triton_already_loaded():
        _shim_conflict_warn_once()
        raise NotImplementedError(
            "PtrAnalysis C++ shim cannot be loaded in this process: "
            "Triton's libtriton (with its own LLVM) is already loaded and "
            "registering cl::opts twice would abort the interpreter. "
            "Run shim-dependent tests in a fresh Python process."
        )
    try:
        return importlib.import_module(SHIM_MODULE_NAME)
    except ImportError as exc:  # pragma: no cover - exercised only when unbuilt
        raise NotImplementedError(
            "PtrAnalysis C++ shim is not built.\n"
            "Build it with:\n"
            "  cmake -S poc/triton_frontend/_cxx -B build/triton_frontend_cxx \\\n"
            "    -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR \\\n"
            "    -DTRITON_INSTALL_DIR=$TRITON_INSTALL_DIR\n"
            "  cmake --build build/triton_frontend_cxx\n"
            "Then add build/triton_frontend_cxx to PYTHONPATH.\n"
            f"Original ImportError: {exc!r}"
        ) from exc


@dataclass(frozen=True)
class PtrState:
    """Strided-pointer descriptor recovered by ``mlir::tts::PtrAnalysis``.

    Mirrors the public fields of the C++ ``mlir::tts::PtrState`` struct
    declared in ``vendored/triton_shared/include/triton-shared/
    AnalysisStructured/PtrAnalysis.h``. Values are kept as opaque strings
    (the printed form of the corresponding ``OpFoldResult``) so this
    dataclass does not require an ``mlir.ir`` import.

    Attributes
    ----------
    offsets, sizes, strides:
        The recovered ``OpFoldResult`` lists (printed form). Strings are
        either SSA names like ``"%i"`` or integer literals like ``"16"``.
    source:
        Printed form of the base pointer SSA value, or ``None`` if absent.
    modulos:
        Per-axis modulo information (or ``None`` for axes without one).
        May be empty when not surfaced by the shim.
    shape:
        Original tensor shape for block-pointer loads, or ``None`` for
        plain pointer loads.
    op:
        Printed form of the op the state was extracted from (typically a
        ``tts.make_tptr``). Useful for diagnostics and for matching the
        emitted state back to the rewritten module.
    result_ssa:
        SSA name of the result the state describes (e.g. ``"%2"``). Used
        as the key when threading the state map through the walker.
    """

    offsets: Tuple[str, ...] = ()
    sizes: Tuple[str, ...] = ()
    strides: Tuple[str, ...] = ()
    source: Optional[str] = None
    modulos: Tuple[Optional[str], ...] = ()
    shape: Optional[Tuple[str, ...]] = None
    op: Optional[str] = None
    result_ssa: Optional[str] = None


# Backwards-compatible alias for the prior scaffold name.
#
# DEPRECATED: scheduled for removal once external callers migrate to PtrState.
# We emit a single DeprecationWarning per process on first instantiation
# (cheaper than per-instance) so noisy logs don't drown out real output but
# the deprecation is impossible to miss in CI.
_STRIDED_LAYOUT_WARNED = False


@dataclass
class StridedLayout:
    """Legacy alias of :class:`PtrState`. Kept so the scaffold imports do not
    break for external callers that grew up against the stub.

    .. deprecated::
        Use :class:`PtrState` (immutable, populated by :meth:`PtrAnalysis.extract_states`).
        Scheduled for removal in the next release; first instantiation emits
        a :class:`DeprecationWarning`.
    """

    base: Any = None
    offsets: List[Any] = field(default_factory=list)
    sizes: List[Any] = field(default_factory=list)
    strides: List[Any] = field(default_factory=list)
    order: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        global _STRIDED_LAYOUT_WARNED
        if not _STRIDED_LAYOUT_WARNED:
            _STRIDED_LAYOUT_WARNED = True
            warnings.warn(
                "poc.triton_frontend.ptr_analysis.StridedLayout is deprecated; "
                "use PtrState instead. StridedLayout will be removed in the "
                "next release.",
                DeprecationWarning,
                stacklevel=2,
            )


class PtrAnalysis:
    """Drive ``mlir::tts::PtrAnalysis::rewriteOp`` over an MLIR module.

    The constructor accepts either an ``mlir.ir.Module`` (printed
    immediately to text) or a raw MLIR string. ``rewrite()`` returns the
    rewritten module text; ``extract_states()`` returns the recovered
    ``PtrState`` descriptors.
    """

    def __init__(
        self,
        module: Any,
        *,
        enable_make_gather_scatter_tensor_ptr: bool = False,
        use_unsafe_mask: bool = False,
    ) -> None:
        if isinstance(module, str):
            self._module_text: str = module
        else:
            # Duck-type ``mlir.ir.Module``: anything that ``str()``-ifies to
            # printed MLIR is acceptable.
            self._module_text = str(module)
        self._enable_gs = bool(enable_make_gather_scatter_tensor_ptr)
        self._use_unsafe_mask = bool(use_unsafe_mask)
        self._rewritten_text: Optional[str] = None
        self._states: Optional[List[PtrState]] = None
        self._rewrite_error: Optional[BaseException] = None
        self._shim: Any = None  # lazy

    # ---- public API ------------------------------------------------------

    def rewrite(self) -> str:
        """Run ``PtrAnalysis::rewriteOp`` and return the rewritten module text.

        The return type is ``str`` rather than ``mlir.ir.Module`` to avoid a
        hard dependency on ``mlir-python-bindings`` at the package boundary.
        Callers that already have a Context can re-parse the result.

        Cached: subsequent calls return the previously-rewritten text without
        re-parsing or re-running the analysis.
        """
        if self._rewritten_text is not None:
            return self._rewritten_text
        if self._rewrite_error is not None:
            # Fail-fast on a previously-cached error so we don't redo the work.
            raise self._rewrite_error
        shim = self._ensure_shim()
        # Run rewrite + states extraction in a single ``Module`` lifetime so
        # ``extract_states`` doesn't have to re-parse the module text. We use
        # the Context/Module pybind wrappers directly rather than the
        # ``run_ptr_analysis`` convenience helper for that reason.
        try:
            ctx = shim.Context()
            mod = shim.Module(ctx, self._module_text)
            mod.run_rewrite(self._enable_gs, self._use_unsafe_mask)
            rewritten = mod.to_string()
            states = _parse_states_json(mod.extract_states_json())
        except BaseException as exc:
            self._rewrite_error = exc
            raise
        self._rewritten_text = rewritten
        self._states = states
        return self._rewritten_text

    def extract_states(self) -> List[PtrState]:
        """Return ``PtrState`` descriptors recovered by the analysis.

        Cached alongside :meth:`rewrite`; the first call to either populates
        both caches in a single C++ Module lifetime.
        """
        if self._states is None:
            # ``rewrite`` populates both caches.
            self.rewrite()
        # ``rewrite`` always sets ``_states`` to a list (possibly empty).
        assert self._states is not None
        return self._states

    # ---- internal helpers ------------------------------------------------

    def _ensure_shim(self) -> Any:
        if self._shim is None:
            self._shim = _load_shim()
        return self._shim

    # ---- legacy compatibility (kept so callers built against the stub
    # don't immediately break) --------------------------------------------

    def visit(self, op: Any) -> None:  # pragma: no cover - thin shim path
        raise NotImplementedError(
            "PtrAnalysis.visit() is no longer the entry point; call rewrite()."
        )

    def rebuild_strides(self, value: Any) -> PtrState:  # pragma: no cover
        states = self.extract_states()
        if not states:
            raise RuntimeError("rewrite() produced no PtrStates")
        return states[0]

    def lift_offsets(
        self,
        value: Any,
        loop_ivs: Optional[Sequence[Any]] = None,
    ) -> List[Any]:  # pragma: no cover
        raise NotImplementedError(
            "lift_offsets is folded into rewrite(); use extract_states() to "
            "inspect the recovered offsets."
        )

    def known_layouts(self) -> List[PtrState]:
        return self.extract_states()


_MISSING_FIELD_WARNED: bool = False


def _warn_missing_fields_once(missing: Sequence[str]) -> None:
    global _MISSING_FIELD_WARNED
    if _MISSING_FIELD_WARNED:
        return
    _MISSING_FIELD_WARNED = True
    warnings.warn(
        "PtrAnalysis shim returned PtrState entries missing fields "
        f"{sorted(set(missing))!r}; falling back to printed-op parsing. "
        "Rebuild poc/triton_frontend/_cxx for richer state metadata.",
        RuntimeWarning,
        stacklevel=2,
    )


# Regex that pulls structured fields out of the printed form of a
# ``tts.make_tptr`` op. The shim's minimal JSON contains the printed op text
# in the ``"op"`` key; until the C++ side is updated to emit explicit JSON
# arrays for offsets/sizes/strides we parse it from here. Example input:
#   "%2 = tts.make_tptr %arg0 to sizes: [16], strides: [1], "
#   "offsets: [0], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>"
_TPTR_RE = re.compile(
    r"""
    (?P<result>%[A-Za-z0-9_]+)\s*=\s*tts\.make_tptr\s+
    (?P<source>%[A-Za-z0-9_]+)\s+to\s+
    sizes:\s*\[(?P<sizes>[^\]]*)\]\s*,\s*
    strides:\s*\[(?P<strides>[^\]]*)\]\s*,\s*
    offsets:\s*\[(?P<offsets>[^\]]*)\]\s*,\s*
    shape:\s*\[(?P<shape>[^\]]*)\]
    """,
    re.VERBOSE,
)


def _split_oplist(s: str) -> Tuple[str, ...]:
    """Split a comma-separated MLIR OpFoldResult list, stripping whitespace.

    Empty input -> empty tuple. Keeps each element as a string (caller decides
    if it wants ``int(x)`` -- the dataclass deliberately stays string-typed).
    """
    s = s.strip()
    if not s:
        return ()
    return tuple(part.strip() for part in s.split(",") if part.strip())


def _parse_states_json(raw: str) -> List[PtrState]:
    """Translate the shim's JSON dump into ``PtrState`` instances.

    The shim emits a *minimal* JSON array today: each element is
    ``{"op": "<printed tts.make_tptr>"}``. We parse the printed form to
    recover ``sizes``, ``strides``, ``offsets``, ``shape``, ``source``, and
    the ``result_ssa`` name. Once integration #5 lands the JSON will gain
    explicit array fields and the dict-shaped path below kicks in.
    """
    import json

    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: List[PtrState] = []
    saw_dict_with_explicit_fields = False
    for entry in data:
        if not isinstance(entry, dict):
            continue
        op_text = entry.get("op")
        # ---- Path A: explicit JSON arrays (future-richer schema). ------
        if any(k in entry for k in ("offsets", "sizes", "strides", "source")):
            saw_dict_with_explicit_fields = True
            shape_val = entry.get("shape")
            shape_t: Optional[Tuple[str, ...]]
            if shape_val is None:
                shape_t = None
            else:
                shape_t = tuple(str(s) for s in shape_val)
            modulos_val = entry.get("modulos") or ()
            modulos_t = tuple(
                (None if m is None else str(m)) for m in modulos_val
            )
            out.append(
                PtrState(
                    offsets=tuple(str(o) for o in entry.get("offsets", []) or ()),
                    sizes=tuple(str(s) for s in entry.get("sizes", []) or ()),
                    strides=tuple(str(s) for s in entry.get("strides", []) or ()),
                    source=entry.get("source"),
                    modulos=modulos_t,
                    shape=shape_t,
                    op=op_text,
                    result_ssa=entry.get("result_ssa"),
                )
            )
            continue
        # ---- Path B: minimal {"op": "<printed>"} -- parse from text. ---
        if not isinstance(op_text, str):
            continue
        m = _TPTR_RE.search(op_text)
        if m is None:
            # Not a make_tptr -- record what we can so the caller still sees
            # the op. ``source`` falls back to the op text for compat.
            out.append(PtrState(source=op_text, op=op_text))
            continue
        out.append(
            PtrState(
                offsets=_split_oplist(m.group("offsets")),
                sizes=_split_oplist(m.group("sizes")),
                strides=_split_oplist(m.group("strides")),
                shape=_split_oplist(m.group("shape")) or None,
                source=m.group("source"),
                op=op_text,
                result_ssa=m.group("result"),
            )
        )
    if not saw_dict_with_explicit_fields and out:
        # Only a one-shot warning so we don't flood logs.
        _warn_missing_fields_once(("offsets", "sizes", "strides", "source"))
    return out


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (used by the lowering pipeline).
# ---------------------------------------------------------------------------


def run_ptr_analysis(ttir_text: str) -> str:
    """Run PtrAnalysis on TTIR text and return the rewritten module text.

    Convenience wrapper around the C++ shim's free function of the same name.
    Raises ``NotImplementedError`` (with a build hint) if the shim isn't
    importable.
    """
    shim = _load_shim()
    return shim.run_ptr_analysis(ttir_text)


def extract_ptr_states(ttir_text: str) -> List[PtrState]:
    """Run PtrAnalysis and return the recovered :class:`PtrState` list.

    Wraps the shim's ``extract_ptr_states`` (which returns a JSON string in
    the current shim build) and parses it through :func:`_parse_states_json`.
    """
    shim = _load_shim()
    raw = shim.extract_ptr_states(ttir_text)
    if not isinstance(raw, str):
        # Future-proof: if the shim ever returns a structured list directly
        # we still hand it to the parser via JSON serialization for one
        # consistent code path.
        import json as _json
        raw = _json.dumps(raw)
    return _parse_states_json(raw)


@lru_cache(maxsize=128)
def run_ptr_analysis_with_states(
    ttir_text: str,
) -> Tuple[str, List[PtrState]]:
    """Combined rewrite + extract; one shim invocation.

    Returns ``(rewritten_ttir_text, states)``. Cheaper than calling the two
    helpers separately because the shim parses the input only once.
    """
    shim = _load_shim()
    rewritten, raw_states = shim.run_ptr_analysis_with_states(ttir_text)
    if not isinstance(raw_states, str):
        import json as _json
        raw_states = _json.dumps(raw_states)
    return rewritten, _parse_states_json(raw_states)


@lru_cache(maxsize=128)
def run_ptr_analysis_with_states_generic(
    ttir_text: str,
) -> Tuple[str, List[PtrState]]:
    """Combined rewrite + extract, returning generic-form TTIR.

    This keeps the rewritten text consumed by ``mlir.ir`` and the serialized
    PtrState names in the same C++ rewrite/module lifetime. Callers that
    re-print custom TTIR through a second generic conversion can otherwise
    observe stale or colliding SSA references in dynamic offsets/strides.
    """
    shim = _load_shim()
    if hasattr(shim, "run_ptr_analysis_with_states_generic"):
        rewritten, raw_states = shim.run_ptr_analysis_with_states_generic(ttir_text)
    else:  # pragma: no cover - stale extension fallback
        rewritten, raw_states = shim.run_ptr_analysis_with_states(ttir_text)
    if not isinstance(raw_states, str):
        import json as _json
        raw_states = _json.dumps(raw_states)
    return rewritten, _parse_states_json(raw_states)
