"""FX graph validation for the TileLang Dynamo backend.

Split out from ``__init__`` so that ``aot_autograd_glue`` can import it without
circular-import gymnastics. The original ``_validate_graph`` lived in
``__init__.py`` and was imported via ``from . import _validate_graph`` —
which resolved to the package, not the function, raising ``ImportError`` and
breaking the entire AOT autograd path (grok wave-3 #09 HIGH bug #1).
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.fx


__all__ = ["UnsupportedFXOpError", "validate_graph"]


class UnsupportedFXOpError(NotImplementedError):
    """Raised when an FX node target has no TileLang lowering rule.

    The ``tilelang`` Dynamo backend is fail-fast: rather than silently falling
    back to eager, we surface a clear error pointing at the missing emitter
    in ``fx_to_tilelang.ATEN_DISPATCH``.
    """


def validate_graph(gm: "torch.fx.GraphModule") -> None:
    """Inspect ``gm`` and raise :class:`UnsupportedFXOpError` for unknown ops.

    This runs before lowering so the user gets a single descriptive error
    listing every offending node, instead of one error per node from inside
    the dispatch loop.
    """
    from .fx_to_tilelang import ATEN_DISPATCH, _node_op_key  # noqa: WPS433

    unsupported: List[str] = []
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        if node.op == "call_function":
            key = _node_op_key(node.target)
            if key is not None and key in ATEN_DISPATCH:
                continue
            unsupported.append(f"{node.format_node()} (target={node.target!r})")
        elif node.op == "call_method":
            unsupported.append(
                f"{node.format_node()} (method={node.target!r}; "
                "call_method lowering not implemented in POC)"
            )
        else:
            unsupported.append(
                f"{node.format_node()} (submodule={node.target!r}; "
                "call_module inlining not implemented in POC)"
            )
    if unsupported:
        joined = "\n  - ".join(unsupported)
        raise UnsupportedFXOpError(
            "tilelang backend cannot lower the following FX nodes "
            "(see RFC §7 Phase 2.2 — extend `ATEN_DISPATCH`):\n  - "
            + joined
        )
