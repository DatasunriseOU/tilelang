"""Shared SSA / MLIR-op fixtures for the op-emitter test suite.

Background
----------
Before this module landed, every test file under ``poc/triton_frontend/tests``
that needed a fake SSA value or a fake jaxlib-shape MLIR op rolled its own
local class. Three divergent ``_FakeSSA`` / ``_HashableSSA`` / ``_FakeValue``
variants accumulated:

* ``_FakeSSA`` (control tests)  -- dict subclass, id-based ``__hash__``.
* ``_HashableSSA`` (memory tests) -- dict subclass, id-based ``__hash__``.
* ``_HashableSSA`` (arith tests)  -- ``__slots__`` class with ``name/dtype``
  only.
* ``_HashableSSA`` (op_mapping tests) -- ``__slots__`` class with
  ``name/shape/dtype``, no ``__hash__`` override (default identity hash).
* ``_FakeValue`` (pipeline tests) -- ``__slots__`` class with
  ``name/shape/dtype``, tuple-keyed ``__hash__``, dict-like ``__getitem__``.

That divergence was a Wave D4 review finding: three slightly-different
hash semantics across tests means a refactor that subtly changes how
``WalkerCtx.value_map`` keys are compared can pass on one file and fail on
another -- and reviewers can't easily diff the variants. This module is
the single source of truth.

Public surface
--------------
``FakeSSA``
    Hashable SSA fixture. Constructible from any of the call patterns the
    existing tests use:

    * ``FakeSSA("name", "dtype")`` -- positional ``name``, ``dtype``.
    * ``FakeSSA(name=..., shape=..., dtype=...)`` -- all kwargs.
    * ``FakeSSA({"name": ..., "shape": ..., "dtype": ...})`` -- dict-shaped
      positional (legacy ``_FakeSSA(dict)`` callers).

    Subclasses ``dict`` so the ``isinstance(value, dict)`` branches in
    :mod:`poc.triton_frontend.op_mapping` helpers (``_dtype_of``,
    ``_shape_of``, ``_ssa_name``, etc.) still take the dict path.
    Equality and ``__hash__`` are tuple-based on
    ``(name, shape, dtype)`` so two fakes with the same payload compare
    equal -- matching the pipeline-test expectation -- while still being
    a usable ``WalkerCtx.value_map`` key.

``FakeMlirOp``
    Minimal stand-in for a jaxlib ``mlir.ir.Operation`` whose inherent
    attrs live in MLIR Properties storage. ``op.attributes`` is empty
    (matching jaxlib under ``allow_unregistered_dialects=True``); the
    printed op text still carries the ``<{...}>`` block so the
    ``_attrs_with_properties_shared`` regex fallback in op_mapping can
    extract the predicate / rmw_op / transpose_A / transpose_B fields.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence


__all__ = ["FakeSSA", "FakeMlirOp"]


class FakeSSA(dict):
    """Unified hashable SSA fixture.

    See module docstring for the rationale; this class subsumes the
    historical ``_FakeSSA`` / ``_HashableSSA`` / ``_FakeValue`` shapes.
    """

    __slots__ = ()

    def __init__(
        self,
        name: Any = None,
        dtype: Optional[str] = None,
        *,
        shape: Sequence[int] = (),
    ) -> None:
        # Accept three legacy construction shapes:
        #   FakeSSA({"name": ..., ...})       -- dict-positional
        #   FakeSSA("name", "dtype")          -- arith-style positional
        #   FakeSSA(name=..., shape=..., dtype=...)  -- memory/op_mapping kwargs
        if isinstance(name, Mapping):
            payload = dict(name)
            payload.setdefault("name", payload.get("name"))
            payload.setdefault("shape", tuple(payload.get("shape", ())))
            payload.setdefault("dtype", payload.get("dtype", "float32"))
            super().__init__(payload)
            return
        if dtype is None:
            dtype = "float32"
        super().__init__(name=name, shape=tuple(shape), dtype=dtype)

    # ---- attribute-style access (slots-class compatibility) -----------

    @property
    def name(self) -> Any:  # type: ignore[override]
        return self.get("name")

    @property
    def shape(self) -> tuple:
        return tuple(self.get("shape", ()))

    @property
    def dtype(self) -> str:
        return str(self.get("dtype", "float32"))

    # ---- hashing -------------------------------------------------------

    def _key(self) -> tuple:
        return (self.get("name"), tuple(self.get("shape", ())), self.get("dtype"))

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self._key())

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, FakeSSA):
            return self._key() == other._key()
        return NotImplemented

    def __ne__(self, other: object) -> bool:  # type: ignore[override]
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __repr__(self) -> str:  # type: ignore[override]
        return f"FakeSSA(name={self.get('name')!r}, shape={tuple(self.get('shape', ()))}, dtype={self.get('dtype')!r})"


class FakeMlirOp:
    """Minimal stand-in for a jaxlib ``mlir.ir.Operation`` Properties path.

    Under ``allow_unregistered_dialects=True`` jaxlib hides the dialect's
    Properties from ``op.attributes`` (it stays empty), but the printed
    op text still includes the ``<{...}>`` block. This fake mirrors that
    shape so the emitter is forced through the
    ``_attrs_with_properties_shared`` fallback in ``op_mapping.py``.
    """

    __slots__ = ("name", "operands", "results", "attributes", "_printed", "attrs")

    def __init__(
        self,
        name: str,
        operands: Iterable[Any],
        results: Iterable[Any],
        printed: str = "",
        *,
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name = name
        self.operands = list(operands)
        self.results = list(results)
        self.attributes = []  # empty -- jaxlib Properties path
        self._printed = printed
        # ``attrs`` is the dict-shape mirror used by the test path that
        # constructs op records as plain dicts; we keep both surfaces so
        # either ``op.attrs[...]`` or the regex parse over ``str(op)``
        # works.
        self.attrs = dict(attrs) if attrs else {}

    def __str__(self) -> str:  # what _parse_generic_properties_shared reads
        return self._printed
