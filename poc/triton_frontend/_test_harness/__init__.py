"""Triton-frontend reducer test harness.

Sub-package that *measures* (does not modify) the current state of the
TTIR -> TileLang TIR reducer in :mod:`poc.triton_frontend`. The two
public entry points are:

* :func:`jit_to_ttir.triton_jit_to_ttir` -- thin wrapper that takes a
  ``@triton.jit`` Python function and runs ``triton.compiler.compile``
  to obtain the textual TTIR (or returns ``None`` if Triton is missing).
* :mod:`run_corpus` -- orchestrator that walks a corpus of real
  kernels and writes a baseline markdown report to
  ``/tmp/triton_reducer_baseline.md``.

The harness is intentionally side-effect-free w.r.t. the reducer: we
only call ``poc.triton_frontend.from_ttir`` and capture its return
value or exception. Any reducer bug surfaces as a status row in the
report, never a silent fix.

Re-runnable: a parallel agent is filling in stubs under
``poc/triton_frontend/op_emitters/``; rerun the orchestrator after
each emitter lands to measure uplift.
"""
from __future__ import annotations

__all__: list[str] = []
