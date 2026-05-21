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

A legacy forward-only call (``is_backward=False`` with default
``allow_grad_inputs`` and any input has ``requires_grad=True``) still raises
through ``_check_no_grad`` because it indicates a missed aot_autograd capture.
AOTAutograd-managed forward calls pass ``allow_grad_inputs=True`` explicitly.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# Process-wide cache so re-registering the same fused op is idempotent.
_REGISTRY: Dict[str, Callable[..., Any]] = {}
_REGISTRY_LOCK = threading.Lock()

def _iter_tensors(value: Any) -> list[Any]:
    """Return tensor-like leaves from a flat/nested runtime value."""
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_iter_tensors(item))
        return out
    if hasattr(value, "is_contiguous") and hasattr(value, "clone"):
        return [value]
    return []


def _storage_key(tensor: Any) -> tuple[int, int] | None:
    """Best-effort storage identity for PyTorch alias checks."""
    try:
        storage = tensor.untyped_storage()
        return (int(storage.data_ptr()), int(tensor.storage_offset()))
    except Exception:
        try:
            return (int(tensor.data_ptr()), 0)
        except Exception:
            return None


def _ensure_non_aliased_outputs(
    args: Sequence[Any],
    result: Any,
) -> Any:
    """Reject outputs that violate ``torch.library.custom_op`` alias rules.

    AOTAutograd forward graphs may return saved primals alongside computed
    tensors, and metadata ops such as ``aten.detach`` may produce a second
    result that aliases an earlier output. ``torch.library.custom_op`` does
    not allow either form. The wrapper reports the alias instead of hiding a
    copy so alias/view handling stays visible to partitioning and lowering.
    """
    input_keys: dict[tuple[int, int], int] = {}
    for input_idx, tensor in enumerate(_iter_tensors(args)):
        key = _storage_key(tensor)
        if key is not None:
            input_keys[key] = input_idx
    seen_output_keys: dict[tuple[int, int], int] = {}

    def _check_one(value: Any, output_idx: int) -> Any:
        key = _storage_key(value)
        if key is None:
            return value
        if key in input_keys:
            input_idx = input_keys[key]
            raise RuntimeError(
                f"tilelang custom_op output #{output_idx} aliases input "
                f"#{input_idx}. Keep alias/view ops outside the fused region "
                "or materialise them explicitly in the captured graph."
            )
        if key in seen_output_keys:
            first_idx = seen_output_keys[key]
            raise RuntimeError(
                f"tilelang custom_op output #{output_idx} aliases output "
                f"#{first_idx}. Keep alias/view ops outside the fused region "
                "or materialise them explicitly in the captured graph."
            )
        seen_output_keys[key] = output_idx
        return value

    if isinstance(result, tuple):
        return tuple(_check_one(value, i) for i, value in enumerate(result))
    if isinstance(result, list):
        return [_check_one(value, i) for i, value in enumerate(result)]
    return _check_one(result, 0)


def _ensure_contiguous_inputs(
    op_qualname: str,
    tensors: Sequence[Any],
) -> Tuple[Any, ...]:
    """Return ``tensors`` only when every tensor is contiguous and unaliased.

    The fused TileLang runtime boundary must not hide layout repairs. If a
    caller wants a materialised tensor, it must put that operation in the FX
    graph before fusion so allocation/copy cost is visible to compilation.
    """
    for i, t in enumerate(tensors):
        try:
            is_tensor = hasattr(t, "is_contiguous")
        except Exception:  # pragma: no cover
            is_tensor = False
        if not is_tensor:
            continue
        try:
            ok = t.is_contiguous()
        except Exception:  # pragma: no cover
            ok = True
        # Wave-2 #09 view-aliasing guard: ``t._base is not None`` signals the
        # tensor shares storage with another live tensor (slice / select /
        # narrow / unbind / expand). Reject it here so any materialisation is
        # a visible graph operation rather than a wrapper-side copy.
        try:
            aliased = getattr(t, "_base", None) is not None
        except Exception:  # pragma: no cover
            aliased = False
        if not ok or aliased:
            reason = "non-contiguous" if not ok else "view-aliased"
            raise RuntimeError(
                f"tilelang custom_op {op_qualname!r}: input #{i} "
                f"(shape={tuple(getattr(t, 'shape', ()))}) is {reason}. "
                "Materialise layout explicitly before this fused op, e.g. "
                "with .contiguous() in the captured graph, so the copy is "
                "visible to compilation."
            )
    return tuple(tensors)


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
    # Per-output source for tensors that must be passed through outside the
    # custom_op because torch.library forbids returning aliases of inputs.
    # Entries are ``None`` for real fused outputs, ``("input", i)`` /
    # ``("param", i)`` for graph outputs that are exactly a placeholder or
    # get_attr parameter, or ``("output", i)`` for duplicate saved outputs
    # that reuse an earlier graph output.
    output_passthrough_sources: Tuple[Optional[Tuple[str, int]], ...] = ()
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


def _normalise_passthrough_sources(
    artifact: FusedKernelArtifact,
) -> Tuple[Optional[Tuple[str, int]], ...]:
    n_outputs = len(artifact.output_specs)
    raw = tuple(getattr(artifact, "output_passthrough_sources", ()) or ())
    if not raw:
        return tuple(None for _ in range(n_outputs))
    if len(raw) != n_outputs:
        raise RuntimeError(
            f"tilelang custom_op {artifact.name!r}: "
            f"output_passthrough_sources has {len(raw)} entries for "
            f"{n_outputs} output specs"
        )
    out: List[Optional[Tuple[str, int]]] = []
    for i, source in enumerate(raw):
        if source is None:
            out.append(None)
            continue
        if (
            not isinstance(source, tuple)
            or len(source) != 2
            or source[0] not in {"input", "param", "output"}
            or not isinstance(source[1], int)
        ):
            raise RuntimeError(
                f"tilelang custom_op {artifact.name!r}: invalid pass-through "
                f"source at output #{i}: {source!r}"
            )
        out.append(source)
    return tuple(out)


def _as_output_list(result: Any) -> List[Any]:
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return list(result)
    return [result]


def _select_custom_op_outputs(
    result: Any,
    *,
    custom_output_indices: Tuple[int, ...],
    full_output_count: int,
    op_qualname: str,
) -> List[Any]:
    values = _as_output_list(result)
    if len(values) == full_output_count:
        return [values[i] for i in custom_output_indices]
    if len(values) == len(custom_output_indices):
        return values
    raise RuntimeError(
        f"{op_qualname}: launcher returned {len(values)} values, expected "
        f"either {full_output_count} full graph outputs or "
        f"{len(custom_output_indices)} custom-op outputs"
    )


def _passthrough_value(
    source: Tuple[str, int],
    runtime_inputs: Sequence[Any],
    param_tensors: Sequence[Any],
    *,
    op_qualname: str,
) -> Any:
    kind, index = source
    if kind == "output":
        raise RuntimeError(
            f"{op_qualname}: output pass-through sources are resolved from "
            "the reconstructed output list"
        )
    pool = runtime_inputs if kind == "input" else param_tensors
    try:
        return pool[index]
    except IndexError as exc:
        raise RuntimeError(
            f"{op_qualname}: pass-through output references missing "
            f"{kind} #{index}"
        ) from exc


def _reconstruct_full_outputs(
    op_result: Any,
    *,
    passthrough_sources: Tuple[Optional[Tuple[str, int]], ...],
    custom_output_indices: Tuple[int, ...],
    runtime_inputs: Sequence[Any],
    param_tensors: Sequence[Any],
    op_qualname: str,
) -> Any:
    custom_values = _as_output_list(op_result)
    if len(custom_values) != len(custom_output_indices):
        raise RuntimeError(
            f"{op_qualname}: custom op returned {len(custom_values)} values "
            f"for {len(custom_output_indices)} non-pass-through outputs"
        )
    custom_by_index = dict(zip(custom_output_indices, custom_values))
    full: List[Any] = []
    for output_idx, source in enumerate(passthrough_sources):
        if source is None:
            full.append(custom_by_index[output_idx])
        elif source[0] == "output":
            source_index = source[1]
            try:
                full.append(full[source_index])
            except IndexError as exc:
                raise RuntimeError(
                    f"{op_qualname}: output #{output_idx} references "
                    f"unavailable earlier output #{source_index}"
                ) from exc
        else:
            full.append(_passthrough_value(
                source,
                runtime_inputs,
                param_tensors,
                op_qualname=op_qualname,
            ))
    if len(full) == 1:
        return full[0]
    return full


def _bind_passthrough_only_runtime(
    artifact: FusedKernelArtifact,
    passthrough_sources: Tuple[Optional[Tuple[str, int]], ...],
    *,
    is_backward: bool = False,
    allow_grad_inputs: Optional[bool] = None,
) -> Callable[..., Any]:
    allow_grad = is_backward if allow_grad_inputs is None else allow_grad_inputs
    op_qualname = f"tilelang::{artifact.name}{'_bwd' if is_backward else '_fwd'}"
    param_tensors = tuple(artifact.param_tensors)

    def _runner(*runtime_inputs: Any) -> Any:
        _check_no_grad(runtime_inputs, allow_grad=allow_grad)
        return _reconstruct_full_outputs(
            (),
            passthrough_sources=passthrough_sources,
            custom_output_indices=(),
            runtime_inputs=runtime_inputs,
            param_tensors=param_tensors,
            op_qualname=op_qualname,
        )

    side = "bw" if is_backward else "fw"
    _runner.__name__ = f"tilelang_{side}_passthrough_{artifact.name}"
    _runner._tilelang_artifact = artifact  # type: ignore[attr-defined]
    _runner._tilelang_is_backward = is_backward  # type: ignore[attr-defined]
    return _runner


def wrap_as_custom_op(
    artifact: FusedKernelArtifact,
    fx_signature: Dict[str, Any],
    *,
    is_backward: bool = False,
    allow_grad_inputs: Optional[bool] = None,
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
    allow_grad_inputs
        Explicit autograd ownership flag.  ``None`` preserves the legacy
        behavior: backward ops accept grad-bearing inputs and forward-only ops
        reject them.  AOTAutograd-managed forward ops pass ``True`` because
        model parameters remain ``requires_grad=True`` even when runtime
        execution is under ``torch.no_grad()``.
    """
    suffix = "_bwd" if is_backward else "_fwd"
    op_qualname = f"tilelang::{artifact.name}{suffix}"
    allow_grad_runtime = is_backward if allow_grad_inputs is None else allow_grad_inputs
    passthrough_sources = _normalise_passthrough_sources(artifact)
    custom_output_indices = tuple(
        i for i, source in enumerate(passthrough_sources) if source is None
    )
    op_output_specs = tuple(artifact.output_specs[i] for i in custom_output_indices)

    if not custom_output_indices:
        return _bind_passthrough_only_runtime(
            artifact,
            passthrough_sources,
            is_backward=is_backward,
            allow_grad_inputs=allow_grad_runtime,
        )

    with _REGISTRY_LOCK:
        cached = _REGISTRY.get(op_qualname)
        if cached is not None:
            return _bind_runtime(
                cached,
                artifact,
                passthrough_sources=passthrough_sources,
                custom_output_indices=custom_output_indices,
                is_backward=is_backward,
                allow_grad_inputs=allow_grad_runtime,
            )

        # Lazy import — module must import without torch.
        from torch.library import custom_op, register_fake  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]

        param_tensors = tuple(artifact.param_tensors)

        # We don't have a static schema string handy — let custom_op infer
        # one from the typed wrapper. The wrapper takes a flat tensor list
        # and returns a flat tensor (or tuple). Concrete shapes / dtypes are
        # tracked via the meta function.
        n_outputs = len(op_output_specs)
        allow_grad = allow_grad_runtime

        # Hot-path optimisation (grok #09 perf review): captures
        # ``param_tensors`` as a frozen tuple and avoids the per-call
        # ``list(args) + list(param_tensors)`` copy. If there are no
        # captured params we route straight to ``artifact.launcher``.
        _has_params = bool(param_tensors)
        _bound_launcher = artifact.launcher

        # Wave-9 #1: unify _impl / _fake return contract per n_outputs.
        # Old code annotated both as ``-> List[Tensor]`` (to satisfy
        # ``torch.library.infer_schema``, wave-7 #4) but the bodies
        # returned a Tensor for n=1 and a tuple for n>1 — a runtime/type
        # mismatch flagged HIGH by grok wave-7/8 review (rev_38ff59759f).
        # Multi-output regions (FA 9-tuple wired in wave-8 #4) made this
        # latent bug actively brittle.
        #
        # New contract — branch at registration time so runtime and
        # annotation always agree:
        #   n=1  → -> Tensor       (returns the launcher's single output)
        #   n>1  → -> List[Tensor] (always normalises to ``list(outs)``)
        if n_outputs == 1:

            @custom_op(op_qualname, mutates_args=())
            def _impl(args: List[torch.Tensor]) -> torch.Tensor:  # type: ignore[name-defined]
                _check_no_grad(args, allow_grad=allow_grad)
                checked_args = _ensure_contiguous_inputs(op_qualname, args)
                if _has_params:
                    result = _bound_launcher(*checked_args, *param_tensors)
                else:
                    result = _bound_launcher(*checked_args)
                custom_values = _select_custom_op_outputs(
                    result,
                    custom_output_indices=custom_output_indices,
                    full_output_count=len(artifact.output_specs),
                    op_qualname=op_qualname,
                )
                result = _ensure_non_aliased_outputs(checked_args, custom_values[0])
                # Launcher may return a 1-tuple/1-list around the single
                # tensor — unwrap to match the declared schema.
                if isinstance(result, (list, tuple)):
                    if len(result) != 1:
                        raise RuntimeError(
                            f"{op_qualname}: n_outputs=1 launcher returned "
                            f"{len(result)} elements")
                    return result[0]
                return result

            @register_fake(op_qualname)
            def _fake(args: List[torch.Tensor]) -> torch.Tensor:  # type: ignore[name-defined]
                spec = op_output_specs[0]
                return torch.empty(
                    spec.shape,
                    dtype=_spec_to_torch_dtype(spec.dtype),
                    device=args[0].device if len(args) else "meta",
                )

        else:

            @custom_op(op_qualname, mutates_args=())
            def _impl(args: List[torch.Tensor]) -> List[torch.Tensor]:  # type: ignore[name-defined]
                _check_no_grad(args, allow_grad=allow_grad)
                checked_args = _ensure_contiguous_inputs(op_qualname, args)
                if _has_params:
                    result = _bound_launcher(*checked_args, *param_tensors)
                else:
                    result = _bound_launcher(*checked_args)
                result = _select_custom_op_outputs(
                    result,
                    custom_output_indices=custom_output_indices,
                    full_output_count=len(artifact.output_specs),
                    op_qualname=op_qualname,
                )
                result = _ensure_non_aliased_outputs(checked_args, result)
                # Multi-output: launcher may return tuple (FA 9-tuple) or
                # list — coerce to the declared list[Tensor] schema.
                if isinstance(result, (list, tuple)):
                    return list(result)
                # Single-Tensor return on a multi-output op_qualname is a
                # bug in the fused emitter; fail loudly rather than silently
                # corrupt downstream meta propagation.
                raise RuntimeError(
                    f"{op_qualname}: n_outputs={n_outputs} launcher "
                    f"returned a non-iterable {type(result).__name__}")

            @register_fake(op_qualname)
            def _fake(args: List[torch.Tensor]) -> List[torch.Tensor]:  # type: ignore[name-defined]
                outs = []
                for spec in op_output_specs:
                    outs.append(torch.empty(
                        spec.shape,
                        dtype=_spec_to_torch_dtype(spec.dtype),
                        device=args[0].device if len(args) else "meta",
                    ))
                return outs

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
        return _bind_runtime(
            _impl,
            artifact,
            passthrough_sources=passthrough_sources,
            custom_output_indices=custom_output_indices,
            is_backward=is_backward,
            allow_grad_inputs=allow_grad_runtime,
        )


def _bind_runtime(
    op: Callable[..., Any],
    artifact: FusedKernelArtifact,
    *,
    passthrough_sources: Optional[Tuple[Optional[Tuple[str, int]], ...]] = None,
    custom_output_indices: Optional[Tuple[int, ...]] = None,
    is_backward: bool = False,
    allow_grad_inputs: Optional[bool] = None,
) -> Callable[..., Any]:
    """Wrap a registered op into the (placeholder-args)->output callable
    Dynamo expects from a backend.

    Dynamo's contract: the backend returns a callable with the *same*
    signature as ``gm.forward``. FX placeholders (model inputs) are passed
    positionally; parameters / buffers were captured via ``get_attr`` at
    trace time and live inside the artifact. Integration #10 plumbs
    ``allow_grad_inputs`` so the AOTAutograd-managed path accepts grad-bearing
    primals/tangents while the forward-only fallback keeps its guard.
    """

    allow_grad = is_backward if allow_grad_inputs is None else allow_grad_inputs
    param_tensors = tuple(artifact.param_tensors)
    if passthrough_sources is None:
        passthrough_sources = _normalise_passthrough_sources(artifact)
    if custom_output_indices is None:
        custom_output_indices = tuple(
            i for i, source in enumerate(passthrough_sources) if source is None
        )
    has_passthrough = any(source is not None for source in passthrough_sources)
    op_qualname = f"tilelang::{artifact.name}{'_bwd' if is_backward else '_fwd'}"

    def _runner(*runtime_inputs: Any) -> Any:
        _check_no_grad(runtime_inputs, allow_grad=allow_grad)
        result = op(list(runtime_inputs))
        if not has_passthrough:
            return result
        return _reconstruct_full_outputs(
            result,
            passthrough_sources=passthrough_sources,
            custom_output_indices=custom_output_indices,
            runtime_inputs=runtime_inputs,
            param_tensors=param_tensors,
            op_qualname=op_qualname,
        )

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
