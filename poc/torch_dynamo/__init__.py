"""POC: ``torch.compile(backend="tilelang")`` Dynamo backend registration.

RFC reference: ``RFC_unified_fused_kernel.md`` §1 (Goal), §3 (OSS landscape —
``torch.fx → TileLang custom backend`` row), §7 Phase 2 (2.1 skeleton, 2.2 FX
op map, 2.3 ``torch.library.custom_op`` wrap).

This module registers a real (forward-only) ``tilelang`` Dynamo backend.
Backward integration is deferred to integration #10 — see
``aot_autograd_glue.py``.

Op-naming open question (resolution
====================================
PyTorch 2.11+ ``torch.library.custom_op`` expects each registered op qualname
to be globally unique within a process. Repeated registration of the same
qualname raises ``RuntimeError: Tried to register an operator (...) we
already had a Python operator``. Two designs were considered:

  (A) ``tilelang::fused_<content_hash>`` per FX subgraph — clean
      FakeTensor caching, every artifact has a stable identity, but inflates
      the global op registry on workloads with many recompiles.
  (B) one generic ``tilelang::fused`` with the artifact passed as an opaque
      non-tensor arg — flat namespace, but ``torch.library.custom_op``
      requires schema-typed args; opaque Python handles complicate
      ``FakeTensor`` caching and AOTAutograd plumbing.

We adopt **(A)** in this POC. PyTorch 2.11+ ships an LRU on the op registry
behind ``torch._library.custom_ops._OP_REGISTRY`` and the registry is process
local — this is the path the upstream Inductor backend takes for its fused
ops. Caching by content hash also dovetails with TileLang's existing
``cached(...)`` JIT cache (``tilelang/jit/__init__.py``). Idempotent
re-registration is guarded inside ``custom_op_wrapper.wrap_as_custom_op``.

Layout::

    poc/torch_dynamo/
        __init__.py              <- this file (backend registration)
        fx_to_tilelang.py        <- FX graph traversal + dispatch table
        custom_op_wrapper.py     <- torch.library.custom_op wrap
        aot_autograd_glue.py     <- backward via aot_autograd (integration #10)
        examples/torch_compile_smoke.py
        README.md

This package is intentionally outside the production ``tilelang/`` tree until
Phase 2 stabilises.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    import torch.fx

__all__ = [
    "tilelang_backend",
    "register",
    "UnsupportedFXOpError",
]


_REGISTERED: bool = False


class UnsupportedFXOpError(NotImplementedError):
    """Raised when an FX node target has no TileLang lowering rule.

    The ``tilelang`` Dynamo backend is fail-fast: rather than silently falling
    back to eager, we surface a clear error pointing at the missing emitter
    in ``fx_to_tilelang.ATEN_DISPATCH``.
    """


def _validate_graph(gm: "torch.fx.GraphModule") -> None:
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


def tilelang_backend(
    gm: "torch.fx.GraphModule",
    example_inputs: Sequence["torch.Tensor"],
) -> Callable[..., Any]:
    """Dynamo backend entry point — RFC §7 Phase 2.1 / 2.2 / 2.3.

    The backend:
      1. Lowers the joint fwd+bwd FX graph via ``aot_autograd`` (integration
         #10). Both halves go through the same FX -> TileLang walker; the
         single forward-only path is preserved as a fallback when
         ``aot_autograd`` is unavailable (very old PyTorch / lint env).
      2. Each side compiles to a fused TileLang artifact wrapped as
         ``torch.library.custom_op`` (``tilelang::fused_<hash>_fwd`` and
         ``tilelang::fused_<hash>_bwd``).
      3. Validates ``gm`` has only ops we know how to lower (fail-fast)
         when running the forward-only fallback path.
    """
    # Lazy import — module must be importable without torch / tvm installed.
    try:
        from .aot_autograd_glue import (  # noqa: WPS433
            make_aot_backend,
            tilelang_fw_compiler,
            tilelang_bw_compiler,
        )
        backend = make_aot_backend(tilelang_fw_compiler, tilelang_bw_compiler)
        return backend(gm, example_inputs)
    except ImportError:
        # aot_autograd / functorch unavailable — fall back to the legacy
        # forward-only path so the smoke test still runs in lint envs.
        from .fx_to_tilelang import FXToTileLang  # noqa: WPS433
        from .custom_op_wrapper import wrap_as_custom_op  # noqa: WPS433

        _validate_graph(gm)
        lowerer = FXToTileLang(gm, list(example_inputs))
        artifact = lowerer.run()
        return wrap_as_custom_op(artifact, lowerer.fx_signature())


def register() -> None:
    """Register ``tilelang`` with ``torch._dynamo`` (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    # Lazy import: the module must be importable in environments without torch
    # (e.g. doc builds, lint, wheel-less CI stages).
    from torch._dynamo import register_backend  # type: ignore[import-not-found]

    # Canonical 2026 API: register_backend can be used as decorator or callable.
    # Signature contract: (gm, example_inputs) -> Callable[..., List[Tensor]].
    register_backend(name="tilelang", compiler_fn=tilelang_backend)
    _REGISTERED = True
