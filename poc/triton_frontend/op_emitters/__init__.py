"""Op-emitter sub-package for the Triton frontend.

Modules under this package register additional entries into the dispatch
table built by :mod:`poc.triton_frontend.op_mapping`. Splitting them out
keeps merge conflicts manageable when several agents extend the frontend
simultaneously.

Submodules
----------
* :mod:`.arith` -- ``arith.*`` / ``math.*`` / ``tt.fma`` (26 ops).
* :mod:`.memory` -- ``tt.load`` / ``tt.store`` / ``tt.{make_range,broadcast,
  splat,expand_dims,view,reshape,addptr}`` / ``tts.make_tptr`` (10 ops).
* :mod:`.reduction` -- ``tt.reduce`` / ``tt.scan`` / ``tt.dot`` /
  ``tt.atomic_*`` (8 ops).
* :mod:`.control` -- control flow (``scf.for`` / ``scf.if``), ``arith``
  casts, ``arith.bitcast``, ``arith.select``, and ``tt.advance`` (15 ops).
"""
from __future__ import annotations

from .arith import ARITH_EMITTERS
from .control import CONTROL_EMITTERS, EmitError, register_into
from .memory import MEMORY_EMITTERS
from .reduction import REDUCTION_EMITTERS

__all__ = [
    "ARITH_EMITTERS",
    "CONTROL_EMITTERS",
    "EmitError",
    "MEMORY_EMITTERS",
    "REDUCTION_EMITTERS",
    "register_into",
]
