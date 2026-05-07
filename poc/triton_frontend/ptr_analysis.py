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
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

__all__ = [
    "PtrState",
    "StridedLayout",
    "PtrAnalysis",
    "shim_available",
    "SHIM_MODULE_NAME",
]


SHIM_MODULE_NAME = "_triton_frontend_cxx"
"""Name of the pybind11 extension built from ``poc/triton_frontend/_cxx/``."""


def shim_available() -> bool:
    """Return True if the C++ shim is importable from ``sys.path``."""
    return importlib.util.find_spec(SHIM_MODULE_NAME) is not None


def _load_shim() -> Any:
    """Lazy-import the C++ shim or raise a NotImplementedError with build hint."""
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
    """

    offsets: Tuple[str, ...] = ()
    sizes: Tuple[str, ...] = ()
    strides: Tuple[str, ...] = ()
    source: Optional[str] = None
    """Printed form of the base pointer SSA value, or ``None`` if absent."""


# Backwards-compatible alias for the prior scaffold name.
@dataclass
class StridedLayout:
    """Legacy alias of :class:`PtrState`. Kept so the scaffold imports do not
    break for external callers that grew up against the stub.
    """

    base: Any = None
    offsets: List[Any] = field(default_factory=list)
    sizes: List[Any] = field(default_factory=list)
    strides: List[Any] = field(default_factory=list)
    order: Tuple[int, ...] = ()


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
        self._shim: Any = None  # lazy

    # ---- public API ------------------------------------------------------

    def rewrite(self) -> str:
        """Run ``PtrAnalysis::rewriteOp`` and return the rewritten module text.

        The return type is ``str`` rather than ``mlir.ir.Module`` to avoid a
        hard dependency on ``mlir-python-bindings`` at the package boundary.
        Callers that already have a Context can re-parse the result.
        """
        shim = self._ensure_shim()
        self._rewritten_text = shim.run_ptr_analysis(
            self._module_text,
            self._enable_gs,
            self._use_unsafe_mask,
        )
        return self._rewritten_text

    def extract_states(self) -> List[PtrState]:
        """Return ``PtrState`` descriptors recovered by the analysis."""
        shim = self._ensure_shim()
        ctx = shim.Context()
        mod = shim.Module(ctx, self._module_text)
        mod.run_rewrite(self._enable_gs, self._use_unsafe_mask)
        raw_json = mod.extract_states_json()
        return _parse_states_json(raw_json)

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


def _parse_states_json(raw: str) -> List[PtrState]:
    """Translate the shim's JSON dump into ``PtrState`` instances.

    The shim emits a *minimal* JSON array of ``{"op": "<printed-op>"}``
    entries today; once integration #5 lands, the schema will gain
    explicit ``offsets/sizes/strides/source`` arrays. We tolerate both
    shapes here.
    """
    import json

    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: List[PtrState] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        out.append(
            PtrState(
                offsets=tuple(entry.get("offsets", []) or ()),
                sizes=tuple(entry.get("sizes", []) or ()),
                strides=tuple(entry.get("strides", []) or ()),
                source=entry.get("source") or entry.get("op"),
            )
        )
    return out
