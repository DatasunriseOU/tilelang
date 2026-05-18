"""Run the Triton-IR -> TileLang TIR reducer over a corpus of real kernels.

This script measures (does not modify) the current state of the
reducer in :mod:`poc.triton_frontend`. For each kernel in the corpus
it tries to:

  1. capture textual TTIR via :func:`jit_to_ttir.triton_jit_to_ttir`
     (skipped when Triton isn't importable; falls back to the canned
     fixture text in :mod:`canned_ttir`),
  2. invoke :func:`poc.triton_frontend.from_ttir` (or, when ``mlir.ir``
     bindings are present, the full MLIR-walker path),
  3. tag the kernel with one of:

       * ``LOWERED_FULL``     -- no exception, ``ctx.value_map`` populated
       * ``LOWERED_DEGRADED`` -- text walker only (no MLIR / TVM)
       * ``FAILED_OPS``       -- ``NotImplementedError`` from a specific op
       * ``FAILED_PARSE``     -- couldn't even capture TTIR
       * ``FAILED_OTHER``     -- unexpected exception (type + msg captured)

Output: a markdown report at ``--report`` (default
``/tmp/triton_reducer_baseline.md``). The most valuable section is the
"Ops needed for full coverage" frequency table -- it tells the
OP_TABLE-emitter agents which ops to prioritise.

Usage::

    python -m poc.triton_frontend._test_harness.run_corpus
    python -m poc.triton_frontend._test_harness.run_corpus --report=path.md

Re-run after each emitter lands to measure uplift. Honest reporting:
all exceptions are captured, never silenced.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Make sure we can import poc.triton_frontend regardless of how the
# script is invoked (``python -m`` vs. plain ``python path/to/file``).
_FRONTEND_ROOT = Path(__file__).resolve().parents[3]
if str(_FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_ROOT))

from poc.triton_frontend import OP_TABLE  # noqa: E402  -- after sys.path tweak
from poc.triton_frontend import _walk_text_ttir, from_ttir  # noqa: E402
from poc.triton_frontend._test_harness import canned_ttir as _canned  # noqa: E402
from poc.triton_frontend._test_harness import jit_to_ttir as _jit  # noqa: E402

__all__ = [
    "KernelResult",
    "Status",
    "run_one",
    "run_corpus",
    "render_markdown",
    "main",
]


# ---------------------------------------------------------------------------
# Status taxonomy + result rows
# ---------------------------------------------------------------------------


class Status:
    """String constants for the status column. Defined as class attrs (not
    an Enum) so the markdown report keeps the human-readable spelling
    without a ``.value`` indirection.
    """

    LOWERED_FULL = "LOWERED_FULL"
    LOWERED_DEGRADED = "LOWERED_DEGRADED"
    FAILED_OPS = "FAILED_OPS"
    FAILED_PARSE = "FAILED_PARSE"
    FAILED_OTHER = "FAILED_OTHER"


@dataclass
class KernelResult:
    """One row in the baseline report."""

    name: str
    source: str
    description: str
    status: str
    visited_ops: List[str] = field(default_factory=list)
    missing_ops: List[str] = field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    ttir_path: Optional[str] = None
    walker_used: str = "text"  # "text" | "mlir"
    primfunc_kind: Optional[str] = None  # e.g. "tvm.tir.PrimFunc" when LOWERED_FULL


# ---------------------------------------------------------------------------
# Helpers: pick the best walker available
# ---------------------------------------------------------------------------


_NOT_IMPL_OP_RE = re.compile(
    r"TTIR op '([^']+)' is not in OP_TABLE", re.MULTILINE
)


def _extract_missing_op(exc: BaseException) -> Optional[str]:
    """Pull the offending op name out of an OP_TABLE NotImplementedError."""
    msg = str(exc)
    m = _NOT_IMPL_OP_RE.search(msg)
    if m:
        return m.group(1)
    return None


def _has_mlir_walker() -> bool:
    """Return True iff the MLIR walker module imports its bindings."""
    try:
        from poc.triton_frontend import mlir_walker  # noqa: WPS433
        return bool(getattr(mlir_walker, "MLIR_WALKER_AVAILABLE", False))
    except Exception:
        return False


def _has_tvm() -> bool:
    """Whether ``import tvm`` succeeds (gates LOWERED_FULL classification)."""
    try:
        importlib.import_module("tvm")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Run one kernel through the reducer
# ---------------------------------------------------------------------------


def _try_text_walker(ttir_text: str) -> Tuple[List[str], Optional[BaseException]]:
    """Run the regex/text walker, return (visited_ops, exception)."""
    try:
        visited = _walk_text_ttir(ttir_text)
        return visited, None
    except Exception as exc:  # noqa: BLE001 -- we surface in the report
        return [], exc


def _try_mlir_walker(
    ttir_text: str,
) -> Tuple[List[str], Optional[BaseException], Optional[str], Optional[Any]]:
    """Run the full MLIR walker. Returns (ops, err, primfunc_kind, ctx).

    ``primfunc_kind`` is the type-name of the returned PrimFunc when the
    walker succeeded; ``None`` when an exception was raised. ``ctx`` (the
    populated :class:`WalkerCtx`) is returned for inspection.
    """
    try:
        from poc.triton_frontend import mlir_walker  # noqa: WPS433
    except Exception as exc:
        return [], exc, None, None

    try:
        # Use the documented public from_ttir entry point so we exercise
        # the same path a real user would. It runs PtrAnalysis (when the
        # shim is available) plus the MLIR walker.
        prim = from_ttir(ttir_text)
        kind = type(prim).__module__ + "." + type(prim).__name__
        return _enumerate_all_ops(ttir_text), None, kind, prim
    except Exception as exc:  # noqa: BLE001
        return [], exc, None, None


_OP_NAME_RE = re.compile(
    r"^\s*(?:%[\w\d_]+(?:,\s*%[\w\d_]+)*\s*=\s*)?"
    r"(?P<op>[a-zA-Z_][\w\.]*)",
)


def _enumerate_all_ops(ttir_text: str) -> List[str]:
    """Best-effort enumeration of *every* dialect op in the TTIR text.

    The reducer's text walker uses a stricter regex (``tt.*`` /
    ``async_copy`` / ``mbarrier``) and silently skips anything else.
    For honest reporting we want a parallel pass that catches
    ``arith.*``, ``math.*``, ``scf.*``, ``llvm.*`` ops too -- those are
    real ops the survey says appear and the reducer cannot route them
    via OP_TABLE today.
    """
    ops: List[str] = []
    for line in ttir_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("module") or stripped in {"{", "}", "}}"}:
            continue
        if stripped.startswith("tt.func") or stripped.startswith("tt.return"):
            ops.append(stripped.split()[0])
            continue
        m = _OP_NAME_RE.match(line)
        if not m:
            continue
        cand = m.group("op")
        # Filter MLIR-style dotted op names only (e.g. "tt.load",
        # "arith.addf"). Plain identifiers without a dot are either
        # SSA names or syntax noise.
        if "." not in cand:
            continue
        ops.append(cand)
    return ops


def run_one(kernel: _canned.CannedKernel) -> KernelResult:
    """Run the reducer on one corpus entry; return the tagged result row."""
    result = KernelResult(
        name=kernel.name,
        source=kernel.source,
        description=kernel.description,
        status=Status.FAILED_OTHER,
    )

    # --- Step 1: obtain TTIR text. ---
    # Prefer live Triton compile when available (more authentic). Fall
    # back to the canned fixture text otherwise.
    ttir_text: Optional[str] = None
    if _jit.triton_available() and getattr(kernel, "live_kernel", None) is not None:
        try:
            ttir_text = _jit.triton_jit_to_ttir(
                kernel.live_kernel,  # type: ignore[arg-type]
                constexprs=kernel.constexprs,
            )
        except _jit.TritonUnavailable:
            ttir_text = None
        except _jit.TTIRCaptureError as exc:
            result.status = Status.FAILED_PARSE
            result.error_type = type(exc).__name__
            result.error_message = str(exc)
            return result

    if ttir_text is None:
        ttir_text = kernel.ttir_text  # canned fallback

    # --- Step 2: prefer the MLIR walker. ---
    ops, err, kind, _ctx = _try_mlir_walker(ttir_text)
    if err is None:
        result.status = (
            Status.LOWERED_FULL if _has_tvm() else Status.LOWERED_DEGRADED
        )
        result.visited_ops = ops
        result.primfunc_kind = kind
        result.walker_used = "mlir"
        return result
    # MLIR walker failed -- fall through to the text walker so we can
    # at least report op coverage. Capture the MLIR exception for the
    # diagnostics column when the text path also fails.
    mlir_err: Optional[BaseException] = err

    # --- Step 3: text walker (degraded path, but always available). ---
    visited, exc = _try_text_walker(ttir_text)
    result.visited_ops = visited
    result.walker_used = "text"

    # Independent pass: enumerate *every* dotted dialect op the TTIR
    # mentions and record the ones not in OP_TABLE. The text walker's
    # regex only matches ``tt.*``/``async_copy``/``mbarrier`` so it
    # silently skips ``arith.*`` / ``math.*`` / ``scf.*`` / ``llvm.*``;
    # we surface those gaps here so the missing-ops table reflects what
    # an honest MLIR walker would have raised on.
    extra_missing: List[str] = []
    structural = {"tt.func", "tt.return"}
    for op_name in _enumerate_all_ops(ttir_text):
        if op_name in structural or op_name in OP_TABLE:
            continue
        if op_name not in extra_missing:
            extra_missing.append(op_name)

    if exc is None:
        # Text walker happy: every ``tt.*`` op was in OP_TABLE. But if
        # ``extra_missing`` is non-empty, then the kernel mentions ops
        # from other dialects (arith / math / scf) that the reducer
        # cannot route -- mark as FAILED_OPS so the missing-ops table
        # picks them up.
        if extra_missing:
            result.status = Status.FAILED_OPS
            result.missing_ops = extra_missing
            result.error_type = "NotImplementedError"
            result.error_message = (
                "OP_TABLE missing entries for: " + ", ".join(extra_missing)
            )
            return result
        # All ops in OP_TABLE; reducer can at least *route* every op. The
        # PrimFunc shell is empty so we mark this DEGRADED, not FULL --
        # the MLIR walker is the only way to populate value_map.
        result.status = Status.LOWERED_DEGRADED
        if mlir_err is not None:
            # Surface MLIR walker failure for diagnostics, even though the
            # text walker happily passed (e.g. an emitter raised inside an
            # OP_TABLE entry that the text path never invokes).
            result.error_type = type(mlir_err).__name__
            result.error_message = str(mlir_err)
        return result

    # Text walker raised. Classify.
    if isinstance(exc, NotImplementedError):
        missing = _extract_missing_op(exc)
        all_missing: List[str] = []
        if missing:
            all_missing.append(missing)
        for op in extra_missing:
            if op not in all_missing:
                all_missing.append(op)
        result.status = Status.FAILED_OPS
        result.missing_ops = all_missing
        result.error_type = "NotImplementedError"
        result.error_message = str(exc)
        return result
    # Anything else is FAILED_OTHER.
    result.status = Status.FAILED_OTHER
    result.error_type = type(exc).__name__
    result.error_message = str(exc)
    return result


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


_SURVEY_HEADER_RE = re.compile(
    r"^###\s+\d+\.\s+(?P<name>[^\s—-]+)\s*[—–-]\s*(?P<src>.+?)\s*$"
)
_SURVEY_OPS_RE = re.compile(r"^- Ops:\s*(?P<ops>.+?)\s*$", re.IGNORECASE)
_SURVEY_NOTES_RE = re.compile(r"^- Notes:\s*(?P<notes>.+?)\s*$", re.IGNORECASE)


def _try_load_survey(path: str) -> Optional[List[Dict[str, str]]]:
    """Parse the survey markdown produced by the parallel agent.

    The survey at ``/tmp/triton_kernel_survey.md`` uses a descriptive
    layout::

        ### 1. _kernel_name — repo/path/to/file.py:LINE
        - LOC: ...
        - Ops: load, broadcast, store, ...
        - Notes: ...

    We extract one corpus row per ``###`` header. Rows we can't parse
    are silently skipped so a malformed survey doesn't break the
    harness. Returns ``None`` when the file is missing entirely.
    """
    if not os.path.exists(path):
        return None
    rows: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip()
                m = _SURVEY_HEADER_RE.match(line)
                if m:
                    if current is not None:
                        rows.append(current)
                    current = {
                        "name": m.group("name"),
                        "source": m.group("src").strip(),
                        "description": "",
                        "ops": "",
                    }
                    continue
                if current is None:
                    continue
                ops_m = _SURVEY_OPS_RE.match(line)
                if ops_m:
                    current["ops"] = ops_m.group("ops")
                    continue
                notes_m = _SURVEY_NOTES_RE.match(line)
                if notes_m:
                    current["description"] = notes_m.group("notes")
                    continue
        if current is not None:
            rows.append(current)
    except Exception as exc:  # pragma: no cover -- malformed survey
        print(
            f"[run_corpus] WARN: failed to parse {path}: {exc!r}",
            file=sys.stderr,
        )
        return None
    return rows or None


# Mapping of free-form op tokens that appear in the survey's ``- Ops: ...``
# lines onto TTIR / arith / scf / math op names that the OP_TABLE walker
# would have to recognise. This lets us synthesise a tiny TTIR snippet
# per survey row so the harness can probe coverage without owning the
# original Python source.
_SURVEY_OP_TOKENS: Dict[str, List[str]] = {
    "load": ["tt.load"],
    "store": ["tt.store"],
    "dot": ["tt.dot"],
    "reduce": ["tt.reduce"],
    "sum": ["tt.reduce"],
    "broadcast": ["tt.broadcast"],
    "splat": ["tt.splat"],
    "where": ["tt.where"],
    "mask": ["tt.where"],
    "atomic_add": ["tt.atomic_rmw"],
    "atomic": ["tt.atomic_rmw"],
    "tanh": ["math.tanh"],
    "exp": ["math.exp"],
    "exp2": ["math.exp2"],
    "log": ["math.log"],
    "log2": ["math.log2"],
    "rsqrt": ["math.rsqrt"],
    "sqrt": ["math.sqrt"],
    "cos": ["math.cos"],
    "sin": ["math.sin"],
    "cast": ["arith.extf"],
    "math": ["arith.addf", "arith.mulf"],
    "mul": ["arith.mulf"],
    "add": ["arith.addf"],
    "sub": ["arith.subf"],
    "div": ["arith.divf"],
    "min": ["arith.minimumf"],
    "max": ["arith.maximumf"],
    "loop": ["scf.for"],
    "while": ["scf.while"],
    "if": ["scf.if"],
    "select": ["arith.select"],
    "fma": ["tt.fma"],
    "make_range": ["tt.make_range"],
    "program_id": ["tt.get_program_id"],
    "expand_dims": ["tt.expand_dims"],
    "reshape": ["tt.reshape"],
    "trans": ["tt.trans"],
    "scan": ["tt.scan"],
    # Inline asm: we deliberately surface this as a still-unsupported op
    # name so the missing-ops table flags it.
    "asm": ["llvm.inline_asm"],
}


def _synthesize_ttir_from_ops(name: str, ops_blob: str) -> str:
    """Produce a tiny TTIR snippet listing the ops the survey claims appear.

    The text walker only checks op names, so a synthetic snippet is
    sufficient to surface OP_TABLE coverage gaps. Tokens we don't
    recognise are passed through verbatim with a ``tt.`` prefix so the
    walker still flags them via the OP_TABLE membership check.
    """
    seen: List[str] = []
    blob = (ops_blob or "").lower()
    # Split on commas/spaces/parens; be very loose.
    tokens = re.split(r"[\s,()/]+", blob)
    seen_set = set()
    for tok in tokens:
        tok = tok.strip().rstrip(".,;:")
        if not tok:
            continue
        mapped = _SURVEY_OP_TOKENS.get(tok)
        if mapped is None:
            # Unknown free-form term; only forward if it looks like a real
            # op (contains a dot or matches a triton-style identifier).
            if "." in tok and any(p in tok for p in ("tt.", "arith.", "math.", "scf.")):
                mapped = [tok]
            else:
                continue
        for op in mapped:
            if op not in seen_set:
                seen_set.add(op)
                seen.append(op)
    body = "\n    ".join(seen) + "\n" if seen else ""
    return (
        "module {\n"
        f"  tt.func @{re.sub(r'[^A-Za-z0-9_]', '_', name)}() {{\n"
        f"    {body}    tt.return\n"
        "  }\n"
        "}\n"
    )


def _build_corpus(survey_path: str) -> List[_canned.CannedKernel]:
    """Return the corpus to run, blending the survey with canned fixtures.

    Strategy:
      * Always include the canned fixtures (they cover the Tier-1
        ladder + a few real-world flavours, with full TTIR text).
      * If a survey file exists at ``survey_path``, synthesise a TTIR
        snippet from each survey row's ``- Ops: ...`` line. This gives
        us coverage probes for ~10 real kernels without the harness
        owning their full source. Survey rows whose name matches a
        canned fixture are skipped (canned wins).
    """
    corpus: List[_canned.CannedKernel] = list(_canned.CANNED_TTIR_FIXTURES)
    survey_rows = _try_load_survey(survey_path)
    if survey_rows is None:
        return corpus
    canned_names = {k.name for k in corpus}
    for row in survey_rows:
        if row["name"] in canned_names:
            continue  # canned takes priority
        ttir = _synthesize_ttir_from_ops(row["name"], row.get("ops", ""))
        corpus.append(
            _canned.CannedKernel(
                name=row["name"],
                description=row["description"] or "(survey entry; TTIR synthesised from ops list)",
                source=row["source"] or "(survey)",
                ttir_text=ttir,
            )
        )
    return corpus


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _probe_unregistered_overlays() -> Dict[str, List[str]]:
    """Find op-emitter dispatch tables that exist but aren't in OP_TABLE.

    The op_emitters package exposes overlay dicts (CONTROL_EMITTERS,
    MEMORY_EMITTERS, etc.); some are auto-merged into OP_TABLE at
    op_mapping import time, others require an explicit ``register_into``
    call that may not have happened yet. Reporting the un-registered
    overlays tells the OP_TABLE-emitter agents which keys are *available*
    but not currently routed -- often a one-line fix.
    """
    out: Dict[str, List[str]] = {}
    candidates = [
        ("op_emitters.control", "CONTROL_EMITTERS"),
        ("op_emitters.memory", "MEMORY_EMITTERS"),
        ("op_emitters.arith", "ARITH_EMITTERS"),
        ("op_emitters.reduction", "REDUCTION_EMITTERS"),
    ]
    for mod_name, attr in candidates:
        try:
            m = importlib.import_module(f"poc.triton_frontend.{mod_name}")
        except Exception:
            continue
        overlay = getattr(m, attr, None)
        if not isinstance(overlay, dict):
            continue
        unregistered = sorted(k for k in overlay.keys() if k not in OP_TABLE)
        if unregistered:
            out[f"{mod_name}.{attr}"] = unregistered
    return out


def run_corpus(
    survey_path: str = "/tmp/triton_kernel_survey.md",
) -> Tuple[List[KernelResult], Dict[str, Any]]:
    """Run every kernel in the corpus and return (rows, env_summary)."""
    corpus = _build_corpus(survey_path)
    env = _jit.describe_triton_env()
    env["mlir_walker_available"] = _has_mlir_walker()
    env["op_table_size"] = len(OP_TABLE)
    env["op_table_keys"] = sorted(OP_TABLE.keys())
    env["unregistered_overlays"] = _probe_unregistered_overlays()
    env["survey_path"] = survey_path
    env["survey_present"] = os.path.exists(survey_path)
    rows: List[KernelResult] = []
    for kernel in corpus:
        if not kernel.ttir_text:
            rows.append(
                KernelResult(
                    name=kernel.name,
                    source=kernel.source,
                    description=kernel.description,
                    status=Status.FAILED_PARSE,
                    error_type="MissingTTIR",
                    error_message=(
                        "Survey lists this kernel but no canned TTIR exists "
                        "and Triton isn't available to compile from source."
                    ),
                )
            )
            continue
        try:
            rows.append(run_one(kernel))
        except Exception as exc:  # pragma: no cover -- harness bug
            # The harness itself blew up -- record verbatim so the user
            # can see we did not silently swallow the failure.
            rows.append(
                KernelResult(
                    name=kernel.name,
                    source=kernel.source,
                    description=kernel.description,
                    status=Status.FAILED_OTHER,
                    error_type=type(exc).__name__,
                    error_message=f"harness internal error: {exc!r}\n"
                    + traceback.format_exc(),
                )
            )
    return rows, env


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _aggregate_missing_ops(rows: List[KernelResult]) -> List[Tuple[str, int]]:
    """Return missing-op frequency table sorted by descending count.

    Most-needed ops come first, which is what the OP_TABLE-emitter
    agents care about.
    """
    c: Counter[str] = Counter()
    for r in rows:
        for op in r.missing_ops:
            c[op] += 1
    return c.most_common()


def _aggregate_status(rows: List[KernelResult]) -> Counter:
    return Counter(r.status for r in rows)


def render_markdown(rows: List[KernelResult], env: Dict[str, Any]) -> str:
    """Render a single markdown report string."""
    lines: List[str] = []
    lines.append("# Triton frontend reducer baseline\n")
    lines.append(
        "Per-kernel status of `poc.triton_frontend.from_ttir` against a\n"
        "corpus of real kernels. Re-run "
        "`python -m poc.triton_frontend._test_harness.run_corpus` after\n"
        "each emitter agent lands to measure uplift.\n"
    )

    # ---- Environment summary (so failures are reproducible) ----
    lines.append("## Environment\n")
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key in (
        "triton",
        "torch",
        "tvm",
        "tilelang",
        "mlir.ir",
        "mlir_walker_available",
        "op_table_size",
        "survey_path",
        "survey_present",
    ):
        val = env.get(key, "")
        lines.append(f"| `{key}` | `{val}` |")
    lines.append("")

    # ---- Status totals ----
    counts = _aggregate_status(rows)
    n = len(rows)
    lines.append("## Status totals\n")
    lines.append("| status | count | fraction |")
    lines.append("|---|---|---|")
    for status in (
        Status.LOWERED_FULL,
        Status.LOWERED_DEGRADED,
        Status.FAILED_OPS,
        Status.FAILED_PARSE,
        Status.FAILED_OTHER,
    ):
        c = counts.get(status, 0)
        frac = f"{(c / n * 100):.0f}%" if n else "n/a"
        lines.append(f"| `{status}` | {c} | {frac} |")
    lines.append(f"\nTotal kernels: **{n}**\n")

    # ---- Top missing ops ----
    missing = _aggregate_missing_ops(rows)
    lines.append("## Ops needed for full coverage\n")
    if missing:
        lines.append(
            "Frequency table of TTIR ops that raised "
            "`NotImplementedError(\"...not in OP_TABLE\")` during the run. "
            "Highest count first -- prioritise these in the OP_TABLE-emitter agents.\n"
        )
        lines.append("| op | count | kernels |")
        lines.append("|---|---|---|")
        for op, count in missing:
            kernels = [r.name for r in rows if op in r.missing_ops]
            lines.append(f"| `{op}` | {count} | {', '.join(kernels)} |")
    else:
        lines.append("_No ops missing from OP_TABLE in this run._")
    lines.append("")

    # ---- Per-kernel rows ----
    lines.append("## Per-kernel rows\n")
    lines.append(
        "| name | status | walker | source | error |"
    )
    lines.append("|---|---|---|---|---|")
    for r in rows:
        err_cell = ""
        if r.error_type or r.error_message:
            short_msg = (r.error_message or "").splitlines()[0][:160]
            err_cell = f"`{r.error_type}`: {short_msg}"
        lines.append(
            "| `{name}` | `{status}` | `{walker}` | {source} | {err} |".format(
                name=r.name,
                status=r.status,
                walker=r.walker_used,
                source=r.source.replace("|", "/"),
                err=err_cell.replace("|", "/"),
            )
        )
    lines.append("")

    # ---- Visited ops per kernel (for triage) ----
    lines.append("## Visited ops (per kernel)\n")
    for r in rows:
        if r.visited_ops:
            ops_str = ", ".join(f"`{o}`" for o in r.visited_ops)
        else:
            ops_str = "_(none)_"
        lines.append(f"- **{r.name}** ({r.status}): {ops_str}")
    lines.append("")

    # ---- Unregistered overlays ----
    overlays = env.get("unregistered_overlays") or {}
    if overlays:
        lines.append("## Unregistered op-emitter overlays\n")
        lines.append(
            "Dispatch tables that exist under `poc/triton_frontend/op_emitters/` "
            "but whose keys are not (yet) merged into `OP_TABLE`. Calling each "
            "module's `register_into(OP_TABLE)` would route these ops without "
            "any additional emitter work.\n"
        )
        for src, keys in sorted(overlays.items()):
            lines.append(f"### `{src}` ({len(keys)} keys)\n")
            lines.append("```")
            for k in keys:
                lines.append(k)
            lines.append("```\n")

    # ---- OP_TABLE snapshot ----
    lines.append("## OP_TABLE snapshot\n")
    lines.append(
        "Current keys registered in `poc.triton_frontend.op_mapping.OP_TABLE` "
        "at the time this report was generated. `ARITH_EMITTERS` and "
        "`REDUCTION_EMITTERS` are auto-merged at module import time; other "
        "overlays (see the 'Unregistered op-emitter overlays' section above) "
        "still need `register_into(OP_TABLE)` to be called.\n"
    )
    keys = env.get("op_table_keys", [])
    lines.append("```")
    for k in keys:
        lines.append(k)
    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Triton-IR reducer over a baseline corpus.",
    )
    parser.add_argument(
        "--report",
        default="/tmp/triton_reducer_baseline.md",
        help="Path to write the markdown report (default: %(default)s).",
    )
    parser.add_argument(
        "--survey",
        default="/tmp/triton_kernel_survey.md",
        help="Path to the survey markdown produced by the parallel agent.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Render the report to stdout instead of writing a file.",
    )
    args = parser.parse_args(argv)

    rows, env = run_corpus(survey_path=args.survey)
    md = render_markdown(rows, env)

    if args.print_only:
        sys.stdout.write(md)
        return 0

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    counts = _aggregate_status(rows)
    summary_bits = " ".join(
        f"{s}={counts.get(s, 0)}"
        for s in (
            Status.LOWERED_FULL,
            Status.LOWERED_DEGRADED,
            Status.FAILED_OPS,
            Status.FAILED_PARSE,
            Status.FAILED_OTHER,
        )
    )
    print(f"[run_corpus] wrote {out} ({len(rows)} kernels: {summary_bits})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
