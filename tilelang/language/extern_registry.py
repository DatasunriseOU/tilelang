"""Module-level registry for ``tl.extern_intrinsic`` declarations.

Implements the registry side of RFC §6 (cross-source extern intrinsic
mechanism). Holds a thread-safe global table keyed by the user-visible
intrinsic name; each entry stores the :class:`Frag` signature factory and the
per-target body strings.

The registry is decoupled from the decorator module so that codegen passes
(CUDA / HIP / Metal) can do a lookup-by-name without importing the user-facing
DSL surface (which lazily depends on TVM).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

# Public targets (matches RFC §6 backend priority — Metal, CUDA, HIP).
_VALID_TARGETS: frozenset[str] = frozenset({"cuda", "hip", "metal"})


@dataclass(frozen=True)
class ExternIntrinsic:
    """A registered extern intrinsic.

    Attributes:
        name: Globally-unique intrinsic name, used as the ``call_extern`` symbol.
        signature: Callable returning the tuple of :class:`Frag` for given shape args.
        bodies: Mapping target -> raw ``__device__`` source string.
    """

    name: str
    signature: Callable[..., tuple]  # returns tuple[Frag, ...]
    bodies: Mapping[str, str]

    def has_target(self, target: str) -> bool:
        """Return True iff a body is registered for ``target``."""
        return target in self.bodies


class _Registry:
    """Thread-safe singleton registry of extern intrinsics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._table: Dict[str, ExternIntrinsic] = {}

    def register(self, intrinsic: ExternIntrinsic) -> None:
        """Insert ``intrinsic``; raise if name is already taken."""
        with self._lock:
            if intrinsic.name in self._table:
                raise KeyError(
                    f"extern_intrinsic '{intrinsic.name}' already registered "
                    f"(targets={list(self._table[intrinsic.name].bodies)})"
                )
            for target in intrinsic.bodies:
                if target not in _VALID_TARGETS:
                    raise ValueError(
                        f"extern_intrinsic '{intrinsic.name}' has unknown target "
                        f"'{target}'; valid targets: {sorted(_VALID_TARGETS)}"
                    )
            self._table[intrinsic.name] = intrinsic

    def unregister(self, name: str) -> None:
        """Remove an entry. Used in tests; raises KeyError if absent."""
        with self._lock:
            del self._table[name]

    def lookup(self, name: str) -> Optional[ExternIntrinsic]:
        """Return the entry for ``name`` or None."""
        with self._lock:
            return self._table.get(name)

    def keys(self) -> tuple[str, ...]:
        """Snapshot of registered names."""
        with self._lock:
            return tuple(self._table.keys())

    def clear(self) -> None:
        """Drop all entries. Test-only helper."""
        with self._lock:
            self._table.clear()


_REGISTRY = _Registry()


def register(intrinsic: ExternIntrinsic) -> None:
    """Register an :class:`ExternIntrinsic` in the global registry."""
    _REGISTRY.register(intrinsic)


def lookup(name: str) -> Optional[ExternIntrinsic]:
    """Return the registered intrinsic for ``name`` or ``None``."""
    return _REGISTRY.lookup(name)


def keys() -> tuple[str, ...]:
    """Return a snapshot of registered intrinsic names."""
    return _REGISTRY.keys()


def clear() -> None:
    """Clear the registry (test-only)."""
    _REGISTRY.clear()


def valid_targets() -> frozenset[str]:
    """Return the set of supported codegen targets."""
    return _VALID_TARGETS


__all__ = [
    "ExternIntrinsic",
    "register",
    "lookup",
    "keys",
    "clear",
    "valid_targets",
]
