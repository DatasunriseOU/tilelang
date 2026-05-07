"""AOTAutograd glue for the TileLang Dynamo backend.

RFC reference: ``RFC_unified_fused_kernel.md`` §7 Phase 2.3 (autograd meta +
backward) and §8 risk #5 ("Autograd through fused custom_op"). This file is
integration #10 of the RFC.

Resolution of the (A) per-pattern vs (B) joint design question
==============================================================
We adopt **(B) joint via aot_autograd**.

Rationale
---------
* aot_autograd captures the *joint* fwd+bwd FX graph upfront and hands us
  TWO independent ``GraphModule``\\ s — one for fwd, one for bwd — with the
  saved-tensor / tangent plumbing already wired in. This matches what
  Inductor and Helion both do (see ``torch._inductor.compile_fx`` —
  ``aot_autograd(fw_compiler=compile_fx_inner, bw_compiler=...)``), and it
  is the only path PyTorch's autograd engine endorses for backends that
  want to fuse across the bwd boundary.
* Approach (A) — registering paired ``tilelang::fused_<hash>_fwd`` and
  ``tilelang::fused_<hash>_bwd`` with ``torch.library.register_autograd``
  — works for a single, fully-statically-known fused op, but it requires
  us to manually compute saved tensors, tangent shapes, and conjugate
  gradients before we even see the bwd FX graph. aot_autograd already
  does all of that.
* Approach (B) also lets us reuse the *exact same*
  :class:`fx_to_tilelang.FXToTileLang` walker for both directions; the
  bwd graph is just another FX ``GraphModule`` with a different op set.
  We extend ``ATEN_DISPATCH`` with the top-8 bwd ATEN ops below.
* The per-hash op qualname collision concern that motivated (A) on the
  fwd side is resolved by appending ``_fwd`` / ``_bwd`` suffixes inside
  ``custom_op_wrapper.wrap_as_custom_op(..., is_backward=...)``; each
  joint capture produces a unique pair.

Known limitations (Phase 2.3 PoC — acceptable trade-offs)
---------------------------------------------------------
1. **functorch / torch.func.grad composability.** Nested transformations
   (``vmap(grad(f))``, ``grad(vmap(f))``, double-backward) are *not*
   guaranteed to round-trip. ``aot_autograd`` only captures a single
   joint fwd+bwd. ``torch.func`` transformations re-enter Dynamo and may
   either retrace through our backend (fine, double compile cost) or hit
   the ``torch.library.custom_op`` autograd-disabled tag and bail out.
   See https://pytorch.org/docs/stable/notes/extending.func.html — we
   cite this as a Phase 3 follow-up.

2. **View aliasing across the fused boundary.** If a fused op's output
   is a view of one of its inputs (e.g. ``out = x.view(...)`` followed
   by an in-place mutation), the joint graph aot_autograd hands us still
   names the underlying storage. Our ``custom_op_wrapper`` declares
   ``mutates_args=()`` and FakeTensor caching assumes outputs are
   freshly allocated — view fusions will produce *correct numerics but
   silently break aliasing assumptions* downstream. We document this and
   recommend ``view`` ops stay outside the fused subgraph until Phase 3.

3. **Non-differentiable ops fused with differentiable ones.** When the
   fwd subgraph contains an op like ``aten.argmax`` (no gradient) plus
   ``aten.matmul`` (has gradient), aot_autograd partitions the joint at
   the boundary and emits a bwd graph that consumes only the
   differentiable saved tensors. Our backend respects the partition
   passively — the bwd compiler simply receives a smaller graph. The
   only failure mode is when the partition is non-trivial (e.g. mixed
   custom ops): we surface
   :class:`__init__.UnsupportedFXOpError` and let Dynamo fall back to
   eager for that subgraph.

Canonical 2026 API surface (PyTorch 2.11+)
------------------------------------------
::

    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func

    backend = aot_autograd(
        fw_compiler=tilelang_fw_compiler,
        bw_compiler=tilelang_bw_compiler,
    )

The ``make_boxed_func`` import path is preserved across the 2.10 -> 2.11
``functorch`` re-export shuffle (``torch._functorch.aot_autograd`` re-
exports it for back-compat). We import it lazily and try both locations
to be safe — see :func:`_import_make_boxed_func`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    import torch.fx


__all__ = [
    "make_aot_backend",
    "tilelang_fw_compiler",
    "tilelang_bw_compiler",
]


def _import_make_boxed_func() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Lazily import ``make_boxed_func`` across PyTorch 2.10/2.11/2.12.

    The canonical 2.11 path is ``functorch.compile.make_boxed_func``. PyTorch
    2.12 also exposes it as ``torch._functorch.aot_autograd.make_boxed_func``.
    We try both, in priority order. TODO: verify on the exact PyTorch shipped
    with this checkout — both work as of 2026-05.
    """
    try:
        # TODO: verify — canonical 2.11+ path.
        from functorch.compile import make_boxed_func  # type: ignore[import-not-found]
        return make_boxed_func
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from torch._functorch.aot_autograd import make_boxed_func  # type: ignore[import-not-found]
        return make_boxed_func
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(
            "Could not import make_boxed_func from either functorch.compile "
            "or torch._functorch.aot_autograd; integration #10 requires "
            "PyTorch >= 2.10."
        ) from exc


def _compile_one_side(
    gm: "torch.fx.GraphModule",
    example_inputs: Sequence["torch.Tensor"],
    *,
    is_backward: bool,
) -> Callable[..., Any]:
    """Shared body for the fwd / bwd compilers.

    Lowers ``gm`` via :class:`fx_to_tilelang.FXToTileLang`, registers the
    artifact as ``tilelang::fused_<hash>_{fwd,bwd}`` and returns a *boxed*
    callable suitable for aot_autograd. ``aot_autograd`` expects the
    compiler to return a callable that takes a single positional list of
    tensors (the "boxed" calling convention).
    """
    from .fx_to_tilelang import FXToTileLang
    from .custom_op_wrapper import wrap_as_custom_op

    # ``__init__._validate_graph`` runs only on the user-facing backend
    # entry; here we trust aot_autograd to hand us decomposed FX.
    lowerer = FXToTileLang(gm, list(example_inputs))
    artifact = lowerer.run()
    runner = wrap_as_custom_op(
        artifact,
        lowerer.fx_signature(),
        is_backward=is_backward,
    )

    make_boxed = _import_make_boxed_func()

    # ``runner`` already has the (placeholder-args) -> output signature
    # Dynamo expects. ``make_boxed_func`` re-wraps it to the boxed
    # ``([t1, t2, ...]) -> [out, ...]`` convention aot_autograd uses.
    def _unboxed(*args: Any) -> Any:
        return runner(*args)

    _unboxed.__name__ = (
        f"tilelang_{'bw' if is_backward else 'fw'}_{artifact.name}"
    )
    return make_boxed(_unboxed)


def tilelang_fw_compiler(
    gm: "torch.fx.GraphModule",
    example_inputs: List["torch.Tensor"],
) -> Callable[..., Any]:
    """Forward compiler used by ``aot_autograd``.

    Lowers the forward FX graph via :class:`fx_to_tilelang.FXToTileLang`
    and registers the result as ``tilelang::fused_<hash>_fwd``.
    """
    return _compile_one_side(gm, example_inputs, is_backward=False)


def tilelang_bw_compiler(
    gm: "torch.fx.GraphModule",
    example_inputs: List["torch.Tensor"],
) -> Callable[..., Any]:
    """Backward compiler used by ``aot_autograd``.

    Lowers the backward FX graph (saved tensors + tangents in, gradients
    out) via :class:`fx_to_tilelang.FXToTileLang`, with the bwd handlers
    registered in ``ATEN_DISPATCH``. Registers the result as
    ``tilelang::fused_<hash>_bwd``.
    """
    return _compile_one_side(gm, example_inputs, is_backward=True)


def make_aot_backend(
    fw_compiler: Optional[Callable[..., Any]] = None,
    bw_compiler: Optional[Callable[..., Any]] = None,
) -> Callable[..., Any]:
    """Build an ``aot_autograd``-wrapped Dynamo backend.

    Parameters
    ----------
    fw_compiler
        Compiler for the forward FX graph. Defaults to
        :func:`tilelang_fw_compiler`.
    bw_compiler
        Compiler for the backward FX graph. Defaults to ``fw_compiler``
        per PyTorch convention (and, in our case, both lower through the
        same FX walker — only ``ATEN_DISPATCH`` differs by op kind).
    """
    # Lazy import: module must be importable without torch.
    from torch._dynamo.backends.common import aot_autograd  # type: ignore[import-not-found]

    fw = fw_compiler if fw_compiler is not None else tilelang_fw_compiler
    bw = bw_compiler if bw_compiler is not None else fw

    return aot_autograd(fw_compiler=fw, bw_compiler=bw)
