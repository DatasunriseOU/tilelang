"""Wrap a fused TileLang kernel as a ``torch.library.custom_op``.

RFC reference: ``RFC_unified_fused_kernel.md`` §1 (final box of the diagram —
``torch.library.custom_op``) and §7 Phase 2.3 (autograd meta + backward —
integration #10 wires the backward through aot_autograd; see
``aot_autograd_glue.py``).

Implementation notes
--------------------
We use scheme **(A)** from the open question in ``__init__``: each FX
subgraph gets its own ``tilelang::fused_<content_hash>`` op. The hash is
stable across runs of the same graph, so subsequent compiles in the same
process hit a registry-level cache — guarded by ``_REGISTRY`` below since
``torch.library.custom_op`` raises on duplicate registration.

Integration #10 augments this with a ``is_backward`` flag: when True, the
op qualname suffix becomes ``_bwd`` (paired with ``_fwd`` for the forward
op), and the autograd-disabled tag stays in place — aot_autograd handles
gradient routing externally so the bwd op itself does NOT need a
``register_autograd`` block (option (B) from the RFC).

A forward-only call (``is_backward=False`` and any input has
``requires_grad=True``) still raises through ``_check_no_grad`` because it
indicates a missed aot_autograd capture.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Sequence, Tuple, TYPE_CHECKING  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover
    import torch


# Process-wide cache so re-registering the same fused op is idempotent.
_REGISTRY: Dict[str, Callable[..., Any]] = {}
_REGISTRY_LOCK = threading.Lock()

# Wave-2 fix-pack (grok #02 security note 2 + killer scenarios):
# warn-once cache for non-contiguous / aliased tensors at the custom_op
# boundary. Keyed by (op_qualname, position-in-args, contiguity-flag) so
# the warning fires once per op×slot×stride-pattern.
_CONTIGUITY_WARN_SEEN: FrozenSet = frozenset()
_CONTIGUITY_LOCK = threading.Lock()


def _ensure_contiguous_inputs(
    op_qualname: str,
    tensors: Sequence[Any],
) -> Tuple[Any, ...]:
    """Return ``tensors`` with each non-contiguous / aliasing input replaced
    by a fresh contiguous copy, warning once per ``(op, slot, reason)`` triple.

    Wave-2 #09 expansion: in addition to non-contiguous strides, we also
    detect view-aliased inputs (``t._base is not None``). The fused TileLang
    kernel writes through pointer arithmetic that assumes the input owns its
    storage; a view backed by another live tensor would silently overwrite
    the parent's memory once the launcher believes the slot is exclusive.
    """
    global _CONTIGUITY_WARN_SEEN
    fixed: list = []
    new_seen = set(_CONTIGUITY_WARN_SEEN)
    changed = False
    for i, t in enumerate(tensors):
        try:
            is_tensor = hasattr(t, "is_contiguous")
        except Exception:  # pragma: no cover
            is_tensor = False
        if not is_tensor:
            fixed.append(t)
            continue
        try:
            ok = t.is_contiguous()
        except Exception:  # pragma: no cover
            ok = True
        # Wave-2 #09 view-aliasing guard: ``t._base is not None`` signals the
        # tensor shares storage with another live tensor (slice / select /
        # narrow / unbind / expand). Re-materialise so the kernel writes do
        # not corrupt the parent.
        try:
            aliased = getattr(t, "_base", None) is not None
        except Exception:  # pragma: no cover
            aliased = False
        if not ok or aliased:
            reason = "non-contiguous" if not ok else "view-aliased"
            key = (op_qualname, i, reason)
            if key not in new_seen:
                new_seen.add(key)
                changed = True
                warnings.warn(
                    f"tilelang custom_op {op_qualname!r}: input #{i} "
                    f"(shape={tuple(getattr(t, 'shape', ()))}) is "
                    f"{reason}; auto-materialising a fresh contiguous "
                    f"copy. For best perf insert an explicit .contiguous() "
                    f"upstream of this op.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            # Wave-3 perf fix (grok review #02 perf §1): only the *aliased+
            # already-contiguous* case forces ``clone()`` — that is the slot
            # where ``.contiguous()`` is a no-op and would still share storage
            # with the parent tensor. For non-contiguous inputs (the common
            # view-heavy LLM-forward case) ``.contiguous()`` already allocates
            # fresh storage, so the extra ``clone()`` is wasted ~2× cost.
            if aliased and ok:
                fixed.append(t.contiguous().clone())
            else:
                fixed.append(t.contiguous())
        else:
            fixed.append(t)
    if changed:
        with _CONTIGUITY_LOCK:
            _CONTIGUITY_WARN_SEEN = frozenset(new_seen)
    return tuple(fixed)


@dataclass
class FusedKernelArtifact:
    """Opaque handle returned by :class:`FXToTileLang.run`.

    Attributes
    ----------
    name
        Suffix used in the op qualname (``tilelang::<name>``). Comes from
        :py:meth:`FXToTileLang.content_hash`.
    launcher
        Callable that takes the runtime input tensors (placeholders +
        param tensors in FX order) and returns the output tensor(s).
    input_specs / output_specs
        Static shape / dtype specs derived from FX example_inputs.
    param_tensors
        nn.Parameters / buffers captured via ``get_attr`` — bound at compile
        time and re-played on every call.
    prim_func
        The *first* TileLang ``tvm.tir.PrimFunc`` from the fusable-region
        chain if real lowering succeeded, else ``None``. Back-compat field;
        new code should consult :attr:`prim_funcs` for the full chain.
    prim_funcs
        Tuple of one ``tvm.tir.PrimFunc`` per fusable region (RFC §3
        partitioning). When non-empty the wrapper exposes a single
        ``tilelang::fused_<hash>_fwd`` op whose impl chains the compiled
        regions in order. Empty if every region fell back to the per-op
        extern slot (eager-replay equivalent).
    source
        Free-form info string (e.g. ``"region#0 ... ok | region#1 ... ok"``)
        carrying per-region compile status for debug / logging.
    """

    name: str
    launcher: Callable[..., Any]
    input_specs: Sequence[Any]
    output_specs: Sequence[Any]
    param_tensors: Sequence[Any] = ()
    prim_func: Any = None
    prim_funcs: Tuple[Any, ...] = ()
    source: str = ""
    # Wave-4 #09 fix #6: atomic-accumulator flag lives on the artifact so
    # the bwd compile can read it back (was previously sniffed inline in
    # aot_autograd_glue, which made the metadata orthogonal to the
    # artifact's identity and impossible to surface in tooling).
    has_atomic_accumulator: bool = False


def _spec_to_torch_dtype(dtype_name: str) -> Any:
    """Map a TileLang dtype string back to ``torch.dtype``."""
    import torch  # type: ignore[import-not-found]

    M = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return M.get(dtype_name, torch.float32)


def _check_no_grad(tensors: Sequence[Any], *, allow_grad: bool = False) -> None:
    """Raise if any input requires grad — unless ``allow_grad`` is True.

    For the *backward* op registered by integration #10 we explicitly
    accept ``requires_grad=True`` tensors: aot_autograd has already taken
    over autograd routing, and the tangents it hands us may carry
    requires_grad through the joint graph. ``allow_grad=True`` short-
    circuits the legacy guard.
    """
    if allow_grad:
        return
    for t in tensors:
        if getattr(t, "requires_grad", False):
            raise NotImplementedError(
                "tilelang forward-only path saw an input with "
                "requires_grad=True. Use the aot_autograd-wrapped backend "
                "(see poc/torch_dynamo/aot_autograd_glue.py) — this guard "
                "fires when the joint capture was bypassed.")


def wrap_as_custom_op(
    artifact: FusedKernelArtifact,
    fx_signature: Dict[str, Any],
    *,
    is_backward: bool = False,
) -> Callable[..., Any]:
    """Register ``artifact`` as ``tilelang::<artifact.name>{_fwd,_bwd}``
    and return a plain Python callable Dynamo can splice back into the
    optimized graph.

    Parameters
    ----------
    artifact
        The compiled :class:`FusedKernelArtifact`. Its ``prim_funcs`` field
        may carry **multiple** ``tvm.tir.PrimFunc``s (one per fusable region
        — RFC §3 partitioning). The artifact's ``launcher`` is already
        responsible for chaining the compiled regions; this wrapper only
        exposes a single ``tilelang::fused_<hash>_fwd`` op qualname so the
        Dynamo / FakeTensor caches still see one op even when the underlying
        compiled artifact is a chain of kernels.
    fx_signature
        Diagnostic-only signature dict from
        :py:meth:`FXToTileLang.fx_signature`.
    is_backward
        Integration #10 (RFC §7 Phase 2.3): when True the op qualname is
        suffixed ``_bwd``, requires_grad inputs are accepted, and the
        autograd-disabled tag stays set (aot_autograd routes the bwd
        externally, so the bwd op itself stays a leaf in the autograd
        graph — option (B) from the RFC). When False the qualname is
        suffixed ``_fwd``.
    """
    suffix = "_bwd" if is_backward else "_fwd"
    op_qualname = f"tilelang::{artifact.name}{suffix}"

    with _REGISTRY_LOCK:
        cached = _REGISTRY.get(op_qualname)
        if cached is not None:
            return _bind_runtime(cached, artifact, is_backward=is_backward)

        # Lazy import — module must import without torch.
        from torch.library import custom_op, register_fake  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]

        param_tensors = tuple(artifact.param_tensors)

        # We don't have a static schema string handy — let custom_op infer
        # one from the typed wrapper. The wrapper takes a flat tensor list
        # and returns a flat tensor (or tuple). Concrete shapes / dtypes are
        # tracked via the meta function.
        n_outputs = len(artifact.output_specs)
        allow_grad = is_backward

        # Hot-path optimisation (grok #09 perf review): captures
        # ``param_tensors`` as a frozen tuple and avoids the per-call
        # ``list(args) + list(param_tensors)`` copy. If there are no
        # captured params we route straight to ``artifact.launcher``.
        _has_params = bool(param_tensors)
        _bound_launcher = artifact.launcher

        @custom_op(op_qualname, mutates_args=())
        def _impl(args: List[torch.Tensor]) -> Any:  # type: ignore[name-defined]
            _check_no_grad(args, allow_grad=allow_grad)
            # Wave-2 fix-pack: contiguity guard at the custom_op boundary.
            args = _ensure_contiguous_inputs(op_qualname, args)
            if _has_params:
                return _bound_launcher(*args, *param_tensors)
            return _bound_launcher(*args)

        @register_fake(op_qualname)
        def _fake(args: List[torch.Tensor]) -> Any:  # type: ignore[name-defined]
            # Shape inference flows for both fwd and bwd: aot_autograd
            # consults this when partitioning the joint graph and when
            # caching FakeTensor results across recompiles.
            outs = []
            for spec in artifact.output_specs:
                outs.append(torch.empty(
                    spec.shape,
                    dtype=_spec_to_torch_dtype(spec.dtype),
                    device=args[0].device if len(args) else "meta",
                ))
            if n_outputs == 1:
                return outs[0]
            return tuple(outs)

        # Phase 2.3 / integration #10:
        #   * autograd-disabled tag stays set on the *bwd* op — aot_autograd
        #     handles gradient routing externally (option (B)). Re-enabling
        #     autograd on the bwd op would cause double differentiation.
        #   * the bwd op deliberately does NOT call
        #     ``torch.library.register_autograd`` — the joint capture
        #     supplies that wiring.
        _impl._tilelang_autograd_disabled = True  # type: ignore[attr-defined]
        _impl._tilelang_is_backward = is_backward  # type: ignore[attr-defined]
        _impl._tilelang_artifact = artifact  # type: ignore[attr-defined]

        _REGISTRY[op_qualname] = _impl
        return _bind_runtime(_impl, artifact, is_backward=is_backward)


def _bind_runtime(
    op: Callable[..., Any],
    artifact: FusedKernelArtifact,
    *,
    is_backward: bool = False,
) -> Callable[..., Any]:
    """Wrap a registered op into the (placeholder-args)->output callable
    Dynamo expects from a backend.

    Dynamo's contract: the backend returns a callable with the *same*
    signature as ``gm.forward``. FX placeholders (model inputs) are passed
    positionally; parameters / buffers were captured via ``get_attr`` at
    trace time and live inside the artifact. Integration #10 plumbs
    ``is_backward`` so the runtime guard accepts grad-bearing tangents.
    """

    allow_grad = is_backward

    def _runner(*runtime_inputs: Any) -> Any:
        _check_no_grad(runtime_inputs, allow_grad=allow_grad)
        return op(list(runtime_inputs))

    side = "bw" if is_backward else "fw"
    _runner.__name__ = f"tilelang_{side}_runner_{artifact.name}"
    _runner._tilelang_op = op  # type: ignore[attr-defined]
    _runner._tilelang_artifact = artifact  # type: ignore[attr-defined]
    _runner._tilelang_is_backward = is_backward  # type: ignore[attr-defined]
    return _runner


def disable_autograd_stub(op: Callable[..., Any]) -> Callable[..., Any]:
    """Tag ``op`` autograd-disabled. Real autograd lands in integration #10."""
    op._tilelang_autograd_disabled = True  # type: ignore[attr-defined]
    return op
