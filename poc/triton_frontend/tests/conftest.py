"""Cross-extension load-order guard for the triton_frontend test suite.

The ``_triton_frontend_cxx`` shim and upstream Triton's ``libtriton`` each
statically link their own copy of LLVM. Loading both into one Python process
triggers ``LLVM ERROR: Option '<name>' already exists!`` (cl::opt duplicate
registration) and aborts the interpreter with SIGABRT mid-collection. The
abort cannot be caught from Python because it comes from ``abort(3)`` inside
LLVM's command-line subsystem.

We fence the conflict in two places:

1. ``poc.triton_frontend.ptr_analysis.shim_available()`` /
   ``dialects_available()`` return ``False`` (with a one-shot warning) when
   triton's libtriton is already loaded. That keeps shim-gated tests as
   clean skips instead of aborts.
2. Test files that pull triton in at module scope (currently
   ``test_standalone.py``) detect a pre-loaded shim and ``pytest.skip``
   the whole module before the triton import line runs.

The net effect: an aggregate ``pytest poc/triton_frontend/tests/`` run
completes cleanly. Whichever side loaded first wins; the other side's
tests skip with a pointer at how to re-run them in a fresh process.
Running each file in isolation still exercises every test.
"""

from __future__ import annotations
