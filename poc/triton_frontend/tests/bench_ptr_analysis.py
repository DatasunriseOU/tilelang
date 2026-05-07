"""Microbench: ``run_ptr_analysis_with_states`` vs the legacy two-call
sequence (``rewrite()`` followed by ``extract_states()`` over separate
parses).

Run as::

    python -m poc.triton_frontend.tests.bench_ptr_analysis

Skips with a clear message when the C++ shim is not built.
"""
from __future__ import annotations

import sys
import time

from poc.triton_frontend.ptr_analysis import (
    PtrAnalysis,
    dialects_available,
    shim_available,
)

_FIXTURE = """\
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  ) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: !tt.ptr<bf16>
    tt.store %1, %10 : !tt.ptr<bf16>
    tt.return
  }
}
"""

_REPEATS = 50


def _bench_single_pass() -> float:
    t0 = time.perf_counter()
    for _ in range(_REPEATS):
        pa = PtrAnalysis(_FIXTURE)
        pa.rewrite()           # populates both caches in one shim Module lifetime
        _ = pa.extract_states()  # cached, free
    return time.perf_counter() - t0


def _bench_legacy_two_pass() -> float:
    """Simulate the pre-d047f31e behaviour: separate PtrAnalysis instances so
    each call performs its own parse + rewrite. Mirrors the worst case the
    cache landed in d047f31e was designed to eliminate.
    """
    t0 = time.perf_counter()
    for _ in range(_REPEATS):
        PtrAnalysis(_FIXTURE).rewrite()
        PtrAnalysis(_FIXTURE).extract_states()
    return time.perf_counter() - t0


def main() -> int:
    if not shim_available():
        print("SKIP: _triton_frontend_cxx not on PYTHONPATH (build the shim "
              "first; see _cxx/README.md)", file=sys.stderr)
        return 0
    if not dialects_available():
        print("SKIP: shim built without TritonStructured/Triton dialects "
              "(rewrite would error in stub mode)", file=sys.stderr)
        return 0
    cached = _bench_single_pass()
    legacy = _bench_legacy_two_pass()
    print(f"single-pass + cached extract: {cached:.4f}s "
          f"({cached / _REPEATS * 1e3:.2f} ms/iter)")
    print(f"legacy two-parse two-rewrite: {legacy:.4f}s "
          f"({legacy / _REPEATS * 1e3:.2f} ms/iter)")
    if cached > 0:
        print(f"speedup: {legacy / cached:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
