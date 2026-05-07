"""FP8 scaled matmul intrinsic exposed on the TileLang language surface.

This module provides ``T.fp8_scaled_matmul`` — a TileLang macro that mirrors
the audiohacking/fp8-mps-metal scaled-matmul kernel signature:

    fp8_scaled_matmul(A_fp8, A_scale, B_fp8, B_scale, C_out)
        # Equivalent to: C_out += (A_fp8.float() * A_scale) @
        #                          (B_fp8.float() * B_scale)
        # with A_fp8 / B_fp8 stored as uchar (e4m3 / e5m2) and the scales
        # broadcast either per-tensor (shape == (1,)) or per-row.

Design
------

The intrinsic is a hygienic ``@T.macro`` that expands inline to the
audiohacking pattern: a scalar K-loop over a dequantize-multiply-accumulate
body, then one post-dot multiply by the per-tensor (or per-row/column)
scales. E8M0 block-scaled operands are the exception: their scale varies by
contracted-K block, so the block scale is applied inside the K loop.
Generic lowering emits scalar FP8 dequantization through ``T.cast``. Metal
targets can instead select direct packed-dot4 or SIMD-group vecmat fast paths
for the e4m3 transpose-B cases where the buffer layout matches.

The reference kernel that this op mirrors is the
``fp8_scaled_matmul_kernel`` published in the audiohacking project:

    https://github.com/audiohacking/fp8-mps-metal
    commit d4fbd40c48aa2a243e600d06627c7dd818150636
    license: MIT

A LUT-decoded variant of the same algorithm ships in
``cppmega_mlx.nn._tilelang.fp8_msl_kernels`` (port of
``AppMana/mps-fp8-for-torch-and-comfyui-python-package`` commit
``a902571eca5362f5e2496cf33dcce52c8bac6a15``, Apache 2.0). Both upstream
projects are credited in the patch comment header.

Why a macro and not a registered TIR op
---------------------------------------

A registered ``tl.fp8_scaled_matmul`` op would buy us:

* a stable IR-level representation (legible in IR-dump traces, addressable
  by passes),
* a single point at which to switch lowering between scalar-emulation,
  cuTe FP8 GEMM (CUDA/Hopper/Blackwell), and any future Metal cooperative
  tensor instruction (Apple has no native FP8 ALU through the M5
  generation — see the Apple WWDC 2025 cooperative-tensors session).

It would cost a C++ rebuild and a parallel scheduler-pass extension. The
hygienic macro form gives us the same user-facing surface today
(``T.fp8_scaled_matmul(...)`` parses cleanly inside ``@T.prim_func``) and
the same MSL output as the C++ approach would, because all the lowering
work (FP8 storage allocation, scalar dequant cast, simdgroup-buffer
exclusion) is already done by the patches that landed earlier:

* ``docs/upstream/tilelang_metal_fp8/`` (Agent C) — storage-only FP8 in
  ``codegen_metal.cc``.
* ``docs/upstream/tilelang_metal_fp8_vector/`` (Agent F-1) — vector FP8
  cast lowering.
* ``docs/upstream/tilelang_metal_fp8_gemm/`` (Agent E) — Metal scalar
  fallback dispatcher for FP8 ``T.gemm``.

Scaled GEMM differs from plain ``T.gemm(fp8, fp8, fp32)`` only by the
final post-dot multiply by ``A_scale * B_scale`` for ordinary floating
scales; the dispatching and codegen path is identical. Mirroring the
audiohacking scalar K-loop therefore reduces to: take the scalar
dequantize-and-dot body that Agent E already validated and apply the
broadcast scale once to the completed dot product. E8M0 block scales keep
their multiply inside the K-loop because the scale index is ``k // 32``.

Behaviour
---------

Within ``@T.prim_func`` the call expands to::

    for i, j in T.grid(M, N):
        base = C[i, j]
        for k in T.serial(K):
            a_val = T.cast(A_fp8[i, k], accum_dtype)   # FP8 -> fp32
            b_val = T.cast(B_fp8[k, j], accum_dtype)   # FP8 -> fp32
            C[i, j] = C[i, j] + a_val * b_val
        sa = A_scale[0] if A_scale.shape == (1,) else A_scale[i]
        sb = B_scale[0] if B_scale.shape == (1,) else B_scale[j]
        C[i, j] = base + (C[i, j] - base) * sa * sb

Per-tensor vs per-row dispatch happens at macro-expansion time based on
the static shape of the scale operand; the resulting MSL has no runtime
branch.

Public attribution
------------------

* audiohacking/fp8-mps-metal (MIT) — algorithm: scalar dequant, fp32 fma,
  post-dot per-tensor / per-row scale broadcast.
* AppMana/mps-fp8-for-torch-and-comfyui-python-package (Apache 2.0) — the
  cppmega.mlx vendor ``mx.fast.metal_kernel`` port that uses a 256-entry
  LUT instead of bit-extraction; functionally equivalent.
"""

from __future__ import annotations

import os
from typing import Optional

from tilelang import tvm as _tvm  # noqa: F401
import tilelang.language as T
from tilelang._typing import BufferLikeType
from tvm import tir
from tvm.target import Target

# CPPMEGA Z3 idea #10 (fp4/fp8 dot4 packed legality): optional Z3 import. We
# fail closed if z3-solver is not installed -- no auto-promotion to the
# packed-dot4 fast path -- so symbolic-shape callers always keep the legacy
# scalar / simd_sum path. Constant-shape callers don't need Z3 at all (the
# static fast path discharges the predicate in plain Python).
try:
    import z3 as _z3  # type: ignore
    _Z3_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when z3-solver isn't installed
    _z3 = None  # type: ignore
    _Z3_AVAILABLE = False

from .blockscaled_layout import (
    BlockScaledLayout,
    E8M0_BLOCK_K32,
    E8M0_BLOCK_SIZE,
    e8m0_to_float,
)

__all__ = [
    "fp8_scaled_matmul",
    "metal_fp8_e4m3_dot4",
    "FP8_DTYPES",
]


# Storage-level FP8 dtype tags accepted by this intrinsic. Any other dtype
# in the A / B operands raises a TypeError at parse time. ``float8_e8m0fnu``
# is the block-scale-factor format and is intentionally excluded — it is
# carried by the sf_a / sf_b operands of the block-scaled GEMM, not by A / B.
FP8_DTYPES: tuple[str, ...] = ("float8_e4m3", "float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz")


def metal_fp8_e4m3_dot4(a_ptr, b_ptr, a_word_idx, b_word_idx):
    """Metal packed FP8 e4m3 dot4 intrinsic.

    ``a_word_idx`` / ``b_word_idx`` are uint32 word indices into byte buffers,
    not byte offsets. Metal lowers this to one LUT-decoded 4-byte dot product.
    """
    return T.call_intrin(
        "float32",
        "tir.metal.fp8_e4m3_dot4",
        a_ptr,
        b_ptr,
        a_word_idx,
        b_word_idx,
    )


def _is_fp8_dtype(dt) -> bool:
    """Return True if a dtype string / object names an FP8 storage variant."""
    s = str(dt or "")
    return any(s.startswith(t) for t in ("float8", "fp8"))


def _shape_extent(buffer, axis: int) -> int:
    """Return a constant integer extent for ``buffer.shape[axis]``.

    Used at macro-expansion time to dispatch per-tensor vs per-row
    behaviour. Falls back to ``-1`` if the extent is symbolic, which the
    caller treats as "assume per-row".
    """
    shape = getattr(buffer, "shape", None)
    if shape is None or len(shape) <= axis:
        return -1
    extent = shape[axis]
    if isinstance(extent, int):
        return extent
    if hasattr(extent, "value"):
        try:
            return int(extent.value)
        except (TypeError, ValueError):
            return -1
    if isinstance(extent, tir.IntImm):
        return int(extent.value)
    return -1


def _resolve_target(target: Optional[Target]) -> Optional[Target]:
    """Return an explicit target, or the target active during reparsing."""
    if target is None:
        return Target.current(allow_none=True)
    if isinstance(target, str):
        return Target(target)
    return target


def _is_metal_target(target: Optional[Target]) -> bool:
    """Return True when the caller requests Metal lowering."""
    target = _resolve_target(target)
    if target is None:
        return False
    kind = getattr(target, "kind", None)
    kind_name = getattr(kind, "name", None)
    if kind_name is not None:
        return str(kind_name).lower() == "metal"
    return "metal" in str(target).lower()


def _target_thread_warp_size(target: Optional[Target]) -> int:
    """Return the target SIMD-group width used by Metal warp intrinsics."""
    target = _resolve_target(target)
    if target is None:
        return 32
    if _is_metal_target(target):
        # Apple's simdgroup reduction intrinsics operate on 32 lanes.  TVM's
        # generic Metal target may expose thread_warp_size=16, but using that
        # here would split one hardware simdgroup across two output columns.
        return 32
    attrs = getattr(target, "attrs", None)
    if attrs is None:
        return 32
    try:
        value = attrs.get("thread_warp_size")
    except AttributeError:
        value = None
    if value is None:
        return 32
    if hasattr(value, "value"):
        return int(value.value)
    return int(value)


# CPPMEGA Z3 idea #10: env-controlled disable knob for the auto-promote
# behaviour. Set TILELANG_DISABLE_FP8_DOT4_AUTO=1 to force the legacy/scalar
# path even when the static / Z3 prover proves the fast-path legal (useful for
# A/B benchmarking and bisection).
def _fp8_dot4_auto_disabled() -> bool:
    return os.environ.get("TILELANG_DISABLE_FP8_DOT4_AUTO", "0") not in ("0", "", "false", "False")


# CPPMEGA Z3 idea #10: probe whether the Metal dot4 / SIMD-position TIR ops
# are registered. The dot4 vecmat macro emits intrinsics that exist in the
# stack-c branch of the Metal codegen but aren't always registered in apache-tvm
# trees. The auto-promote dispatcher must skip the dot4 path when they're
# missing, otherwise lowering raises ``AttributeError: Operator ... is not
# registered`` and we regress every previously-passing kernel.
_DOT4_INTRINSICS = (
    "tir.metal.thread_position_in_grid_x",
    "tir.metal.thread_index_in_simdgroup",
    "tir.metal.fp8_e4m3_dot4",
)
_dot4_intrinsics_registered_cache: Optional[bool] = None


def _dot4_intrinsics_registered() -> bool:
    """Return True iff every dot4 intrinsic op is registered (cached)."""
    global _dot4_intrinsics_registered_cache
    if _dot4_intrinsics_registered_cache is not None:
        return _dot4_intrinsics_registered_cache
    try:
        from tvm.ir import Op  # type: ignore
    except Exception:
        _dot4_intrinsics_registered_cache = False
        return False
    for name in _DOT4_INTRINSICS:
        try:
            Op.get(name)
        except Exception:
            _dot4_intrinsics_registered_cache = False
            return False
    _dot4_intrinsics_registered_cache = True
    return True


def _buffer_innermost_stride(buffer) -> Optional[int]:
    """Return the innermost stride of a buffer if statically known, else None.

    A row-major buffer with no explicit strides has innermost stride 1 by
    construction; if explicit strides are present we read the last one.
    """
    strides = getattr(buffer, "strides", None)
    if not strides:
        # Empty / None means contiguous row-major -> innermost stride == 1.
        return 1
    last = strides[-1]
    if isinstance(last, int):
        return last
    if hasattr(last, "value"):
        try:
            return int(last.value)
        except (TypeError, ValueError):
            return None
    if isinstance(last, tir.IntImm):
        return int(last.value)
    return None


def _buffer_element_offset(buffer) -> Optional[int]:
    """Return ``elem_offset`` if statically known, else None (assume 0)."""
    eo = getattr(buffer, "elem_offset", None)
    if eo is None:
        return 0
    if isinstance(eo, int):
        return int(eo)
    if hasattr(eo, "value"):
        try:
            return int(eo.value)
        except (TypeError, ValueError):
            return None
    if isinstance(eo, tir.IntImm):
        return int(eo.value)
    return None


# ---------------------------------------------------------------------------
# CPPMEGA Z3 idea #10: fp4/fp8 dot4 packed legality
# ---------------------------------------------------------------------------
# The Metal ``metal_fp8_e4m3_dot4`` intrinsic packs 4 fp8 bytes into a uint32
# word and emits one LUT-decoded dot4 per word. For this to be safe we need a
# linear-arith predicate to hold:
#
#     K_a == K_b
#     K  % 4 == 0       (whole number of u32 words)
#     stride_a == 1     (contiguous K)
#     stride_b == 1
#     addr_a % 4 == 0   (each row starts on a u32-word boundary)
#     addr_b % 4 == 0
#
# We discharge the predicate with a TWO-tier strategy:
#
#  * Static fast path: when every numeric arg is a Python int / IntImm we
#    evaluate the predicate inline, which avoids the ~5-10 ms of Z3 setup
#    that previous reviewers flagged as overly heavy for the constant case.
#  * Z3 fallback: when any arg is symbolic we ask Z3 to prove ``Not(legality)``
#    is unsat under non-negativity / range bounds. The bit-bound encoding
#    ``EnterConstraint(0 <= addr < (1<<32))`` emulates the upcoming
#    ``Z3Prover::SetBitVectorMode(32)`` (branch z3-bv-mode); without it Z3
#    is happy to satisfy ``addr % 4 != 0`` with a giant negative witness on
#    free Int sort, which is sound but useless. Timeout is 50 ms; UNKNOWN /
#    timeout / exceptions all return False (conservative).
#
# Either path returning False keeps the caller on the legacy macro. That is
# the conservative-by-default rule -- a wrong proof here would silently emit
# packed-dot4 against an unaligned buffer and trigger a Metal crash on first
# dispatch.
_Z3_DOT4_TIMEOUT_MS = 50

# Address bit width used by the symbolic ``0 <= addr < (1 << _BITS)`` bound.
# 32 matches the eventual ``Z3Prover::SetBitVectorMode(32)`` semantics.
_Z3_DOT4_ADDR_BITS = 32

# CPPMEGA Z3 idea #10 audit-fix: bound the symbolic stride search space. Without
# an upper bound the solver is free to consider absurd strides (the Int sort is
# unbounded), which inflates the search and hits UNKNOWN/timeout when combined
# with the addr % 4 modular obligation. Real kernels never use innermost
# strides above this knob (one stride-1024 row is already 4 KiB / 8 KiB which
# exceeds typical Metal threadgroup-buffer slabs); legality outside [1, 1024)
# is conservative-False by construction. Configurable for future kernels with
# wider K, but the constant is a deliberate "tightening" rather than a hint.
_DOT4_MAX_STRIDE = 1024


def _is_int_imm_or_int(value) -> bool:
    """Return True if ``value`` is a Python int or ``tir.IntImm``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, tir.IntImm):
        return True
    return False


def _const_int_value(value) -> Optional[int]:
    """Coerce ``value`` to a Python int when it's a constant; else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, tir.IntImm):
        return int(value.value)
    if hasattr(value, "value"):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            return None
    return None


def _z3_prove_dot4_legal(
    K_a,
    K_b,
    stride_a,
    stride_b,
    addr_a,
    addr_b,
    A_buf=None,
    B_buf=None,
) -> tuple[bool, str]:
    """Prove the FP8 packed-dot4 fast path is legal for these args.

    Args:
        K_a, K_b: Contracted-dim extents on A and B (``IntImm`` / ``int`` /
            ``tir.PrimExpr``).
        stride_a, stride_b: Innermost (contiguous-K) strides.
        addr_a, addr_b: Buffer element offsets (used as proxies for the start
            addresses; alignment is verified modulo 4).
        A_buf, B_buf: Buffer objects accepted for diagnostic context only.

    The predicate proven is exactly::

        K_a == K_b && K_a % 4 == 0
            && stride_a == 1 && stride_b == 1
            && addr_a % 4 == 0 && addr_b % 4 == 0

    Conservative-on-stride!=1 (audit finding 2026-05-06):
        We intentionally reject any ``stride != 1`` even when the buffer
        layout is contiguous in some other algebraic sense (e.g. row-major
        linearised). The packed-dot4 intrinsic loads a u32 word at
        ``ptr[word_idx]``; if the FP8 byte stream is anything other than
        4-byte contiguous along K, the four bytes packed into one u32 word
        would no longer correspond to four consecutive K elements. So
        ``stride==1`` is the safe, target-independent invariant the proof
        pins. This matches the audit recommendation that "conservative
        rejection on non-unit stride" is the correct posture for the
        Metal dot4 fast path and any future CUDA / ROCm port.

    Symbolic-negative-address invariant (audit finding 2026-05-06):
        When ``addr_a`` / ``addr_b`` are symbolic and the caller has NOT
        added a ``addr >= 0`` constraint, the prover returns False. The
        BV32 emulation (``0 <= addr < (1<<32)``) inside this function only
        clamps the *named* free variables; if the caller substitutes a
        ``tir.Var`` that the prover sees as opaque, the conservative
        outcome is False. Once the C++-side ``Z3Prover::SetBitVectorMode(32)``
        FFI bridge lands, signed wrap-around will be encoded properly via
        BV sort and this invariant becomes part of the BV semantics.

    Returns:
        ``(proved, reason)``. ``proved=True`` means every conjunct is
        established; the caller can route to ``metal_fp8_e4m3_dot4``.
        ``proved=False`` always means the caller MUST keep the legacy path.
        UNKNOWN / timeout / solver exceptions all map to
        ``proved=False`` -- conservative by construction.
    """
    # ----- Static fast path -------------------------------------------------
    # Common case: M=1 vecmat with concrete shapes baked in. Skip Z3 entirely.
    if all(_is_int_imm_or_int(v) for v in (K_a, K_b, stride_a, stride_b, addr_a, addr_b)):
        ka = _const_int_value(K_a)
        kb = _const_int_value(K_b)
        sa = _const_int_value(stride_a)
        sb = _const_int_value(stride_b)
        aa = _const_int_value(addr_a)
        ab = _const_int_value(addr_b)
        if None in (ka, kb, sa, sb, aa, ab):
            return False, "static path could not coerce to int"
        if ka <= 0 or kb <= 0 or ka != kb:
            return False, f"static: K mismatch / non-positive ({ka}, {kb})"
        if ka % 4 != 0:
            return False, f"static: K={ka} not divisible by 4"
        if sa != 1 or sb != 1:
            return False, f"static: stride != 1 ({sa}, {sb})"
        if aa % 4 != 0 or ab % 4 != 0:
            return False, f"static: addr not 4-aligned ({aa}, {ab})"
        # CPPMEGA Z3 idea #5: the alignment proof above is necessary but not
        # sufficient -- the Metal dot4 lane accumulates in int24, so we
        # additionally need ``K * 127 * 127 < 2^23``. Without this gate the
        # fast path silently overflows for K beyond 520. See
        # ``tilelang.analysis.int24_overflow_proof`` for the full obligation;
        # the int8/e4m3 max-abs of 127 is the worst case for the LUT-decoded
        # path on Apple Silicon.
        from tilelang.analysis.int24_overflow_proof import prove_dot4_int24_safe
        if not prove_dot4_int24_safe(ka):
            return False, f"static: int24 overflow at K={ka} (K*127*127 >= 2^23)"
        return True, (
            f"static fast path: K={ka}, strides=(1,1), addrs=({aa}%4=0,{ab}%4=0), "
            "int24 safe"
        )

    # ----- Symbolic / Z3 fallback -------------------------------------------
    # If z3 isn't installed in this interpreter we MUST fall back to the
    # legacy macro -- otherwise we'd silently emit packed-dot4 against a
    # buffer whose alignment we can't prove.
    if not _Z3_AVAILABLE:
        return False, "z3 not available; symbolic dot4 legality cannot be proven"

    # Note: the Python-side ``tvm.arith.Z3Prover`` bridge described in the
    # roadmap is not yet exposed by this TVM submodule. The C++-side prover
    # ships in ``src/transform/vendored/z3_prover.{h,cc}`` but its API
    # (CanProve / EnterConstraint / SetTimeoutMs / SetBitVectorMode) is not
    # FFI-registered. We fall back to the standalone z3-solver Python package
    # and emulate ``EnterConstraint`` / ``SetTimeoutMs`` / BV-mode via the
    # solver's native API, which is sufficient for the linear-arith fragment
    # this proof needs. When the FFI bridge lands, swap the body here for an
    # ``arith.Z3Prover(analyzer)`` call.
    try:
        s = _z3.Solver()
        s.set("timeout", int(_Z3_DOT4_TIMEOUT_MS))

        ka_v = _z3.Int("K_a")
        kb_v = _z3.Int("K_b")
        sa_v = _z3.Int("stride_a")
        sb_v = _z3.Int("stride_b")
        aa_v = _z3.Int("addr_a")
        ab_v = _z3.Int("addr_b")

        # Pin any concrete args; symbolic args remain free Ints. The caller
        # is responsible for any tighter range invariants -- we add only the
        # bare-minimum non-negativity / BV32 emulation below.
        for slot, py_val in (
            (ka_v, K_a), (kb_v, K_b), (sa_v, stride_a), (sb_v, stride_b),
            (aa_v, addr_a), (ab_v, addr_b),
        ):
            cv = _const_int_value(py_val)
            if cv is not None:
                s.add(slot == cv)

        # CPPMEGA Z3 idea #10: emulate Z3Prover::SetBitVectorMode(32) until
        # the C++-side branch (z3-bv-mode) lands. Without these range bounds
        # Z3 can satisfy ``addr % 4 != 0`` with a giant negative witness on
        # symbolic addresses -- sound over Int but spurious for any real
        # address. ``EnterConstraint(0 <= addr && addr < (1<<32))`` matches
        # the C++ Z3Prover semantics the BV-mode branch ships and is the
        # conservative thing to encode here.
        bound = 1 << _Z3_DOT4_ADDR_BITS
        s.add(aa_v >= 0, aa_v < bound)
        s.add(ab_v >= 0, ab_v < bound)
        # CPPMEGA Z3 idea #10 audit-fix: bound strides on both sides. Without
        # ``sa < _DOT4_MAX_STRIDE`` the solver explores arbitrary positive
        # values (Int sort), which is sound but inflates the search and
        # frequently times out (UNKNOWN -> conservative-False) when paired
        # with the modular addr obligations. Pinning a high but finite max
        # gives Z3 a closed search space; legality outside that range
        # remains conservative-False at the dispatcher (see
        # ``_buffer_innermost_stride`` callers).
        s.add(
            ka_v > 0,
            kb_v > 0,
            sa_v > 0, sa_v < _DOT4_MAX_STRIDE,
            sb_v > 0, sb_v < _DOT4_MAX_STRIDE,
        )

        legality = _z3.And(
            ka_v == kb_v,
            ka_v % 4 == 0,
            sa_v == 1,
            sb_v == 1,
            aa_v % 4 == 0,
            ab_v % 4 == 0,
        )

        # Ask Z3 for a counter-example. ``unsat`` => legality is implied;
        # anything else (sat, unknown, timeout) is conservative-False.
        s.push()
        s.add(_z3.Not(legality))
        res = s.check()
        s.pop()
        if res == _z3.unsat:
            # CPPMEGA Z3 idea #5: pair the alignment proof with the int24
            # non-overflow proof. If K is symbolic-and-unbounded the int24
            # prover returns False (see its docstring), keeping us on the
            # legacy scalar path. If K is constant or pinned the int24
            # prover discharges the overflow obligation in plain Python.
            from tilelang.analysis.int24_overflow_proof import prove_dot4_int24_safe
            int24_ok = prove_dot4_int24_safe(K_a)
            if not int24_ok:
                return False, (
                    "z3 proved dot4 alignment but int24 overflow proof failed; "
                    "falling back to scalar accumulator"
                )
            return True, (
                "z3 proved dot4 legal under symbolic constraints "
                f"(timeout={_Z3_DOT4_TIMEOUT_MS}ms, addr_bits={_Z3_DOT4_ADDR_BITS}, "
                "int24 safe)"
            )
        if res == _z3.unknown:
            return False, "z3 returned UNKNOWN (timeout / incomplete); falling back"
        return False, "z3 found a counter-example to dot4 legality; falling back"
    except Exception as exc:  # pragma: no cover - defensive
        # Solver-construction or query errors must NOT crash lowering.
        return False, f"z3 raised {type(exc).__name__}: {exc}; falling back"


def _z3_prove_dot4_legal_for_buffers(
    A_fp8,
    B_fp8,
    *,
    transpose_B: bool,
) -> tuple[bool, str]:
    """Adapter from buffer objects to the numeric proof obligation.

    Extracts ``(K, stride, addr)`` tuples plus dtype/transpose checks. The
    caller (``fp8_scaled_matmul``) uses this to decide whether to route to
    the packed-dot4 macro.
    """
    a_dtype = str(getattr(A_fp8, "dtype", ""))
    b_dtype = str(getattr(B_fp8, "dtype", ""))
    if not a_dtype.startswith("float8_e4m3"):
        return False, f"A dtype {a_dtype!r} is not float8_e4m3"
    if not b_dtype.startswith("float8_e4m3"):
        return False, f"B dtype {b_dtype!r} is not float8_e4m3"
    if not transpose_B:
        return False, "transpose_B=False; dot4 requires K-contiguous B"

    K_a = _shape_extent(A_fp8, 1)
    K_b = _shape_extent(B_fp8, 1) if transpose_B else _shape_extent(B_fp8, 0)
    if K_a <= 0 or K_b <= 0:
        return False, "K extent is symbolic / unknown"

    a_stride = _buffer_innermost_stride(A_fp8)
    b_stride = _buffer_innermost_stride(B_fp8)
    if a_stride is None or b_stride is None:
        return False, "buffer innermost stride is symbolic"

    a_off = _buffer_element_offset(A_fp8)
    b_off = _buffer_element_offset(B_fp8)
    if a_off is None or b_off is None:
        return False, "buffer elem_offset is symbolic"

    return _z3_prove_dot4_legal(
        K_a, K_b, a_stride, b_stride, a_off, b_off, A_fp8, B_fp8,
    )


def _normalize_block_scale_layout(
    block_scale_layout: BlockScaledLayout | None,
    *,
    scale_format: str | None,
    scale_block_size: int | None,
) -> BlockScaledLayout | None:
    if block_scale_layout is not None:
        if not isinstance(block_scale_layout, BlockScaledLayout):
            raise TypeError(
                "T.fp8_scaled_matmul block_scale_layout must be a "
                "T.BlockScaledLayout instance"
            )
        if scale_format is not None and scale_format != block_scale_layout.scale_format:
            raise ValueError(
                "T.fp8_scaled_matmul scale_format conflicts with block_scale_layout"
            )
        if scale_block_size is not None and int(scale_block_size) != block_scale_layout.block_size:
            raise ValueError(
                "T.fp8_scaled_matmul scale_block_size conflicts with block_scale_layout"
            )
        return block_scale_layout
    if scale_format is None and scale_block_size is None:
        return None
    if scale_format != E8M0_BLOCK_K32:
        raise ValueError(
            "T.fp8_scaled_matmul e8m0 block-scale metadata requires "
            "scale_format='e8m0_block_k32'"
        )
    if scale_block_size is None or int(scale_block_size) != E8M0_BLOCK_SIZE:
        raise ValueError(
            "T.fp8_scaled_matmul e8m0_block_k32 metadata requires scale_block_size=32"
        )
    return BlockScaledLayout.e8m0_k32()


def _block_scale_value(scale, *, axis: str, col, k):
    # Path C E8M0 is explicitly contracted-K-block indexed: kb = k // 32.
    kb = k // 32
    if axis == "B" and len(getattr(scale, "shape", ())) == 2:
        return e8m0_to_float(scale[col, kb])
    return e8m0_to_float(scale[kb])


def _validate_buffers(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_out,
    *,
    transpose_B: bool,
    accum_dtype: str,
    block_scale_layout: BlockScaledLayout | None = None,
    a_scale_offset=None,
    b_scale_offset=None,
) -> None:
    """Sanity-check operand dtypes and 2D shape compatibility.

    Raises ``TypeError`` / ``ValueError`` early so misuse surfaces at the
    macro call-site rather than deep inside the parser. The macro proper
    re-derives the same shape information at expansion time; this helper
    is the public-facing validator.
    """
    A_dtype = str(getattr(A_fp8, "dtype", "")) if hasattr(A_fp8, "dtype") else ""
    B_dtype = str(getattr(B_fp8, "dtype", "")) if hasattr(B_fp8, "dtype") else ""
    C_dtype = str(getattr(C_out, "dtype", "")) if hasattr(C_out, "dtype") else ""
    sa_dtype = str(getattr(A_scale, "dtype", "")) if hasattr(A_scale, "dtype") else ""
    sb_dtype = str(getattr(B_scale, "dtype", "")) if hasattr(B_scale, "dtype") else ""

    if not _is_fp8_dtype(A_dtype):
        raise TypeError(
            f"T.fp8_scaled_matmul: A_fp8 must be FP8 (e4m3 or e5m2), got dtype={A_dtype!r}"
        )
    if not _is_fp8_dtype(B_dtype):
        raise TypeError(
            f"T.fp8_scaled_matmul: B_fp8 must be FP8 (e4m3 or e5m2), got dtype={B_dtype!r}"
        )
    scale_prefixes = ("float32", "float16", "bfloat")
    if block_scale_layout is not None:
        scale_prefixes = ("uint8",)
    if sa_dtype and not sa_dtype.startswith(scale_prefixes):
        raise TypeError(
            f"T.fp8_scaled_matmul: A_scale must be a {'uint8 E8M0 block-scale' if block_scale_layout is not None else 'floating-point scalar'} buffer, got dtype={sa_dtype!r}"
        )
    if sb_dtype and not sb_dtype.startswith(scale_prefixes):
        raise TypeError(
            f"T.fp8_scaled_matmul: B_scale must be a {'uint8 E8M0 block-scale' if block_scale_layout is not None else 'floating-point scalar'} buffer, got dtype={sb_dtype!r}"
        )
    if C_dtype and not (C_dtype.startswith("float32") or C_dtype.startswith("float16") or C_dtype.startswith("bfloat")):
        raise TypeError(
            f"T.fp8_scaled_matmul: C output must be float32 / float16 / bfloat16 (got {C_dtype!r})"
        )

    A_shape = getattr(A_fp8, "shape", None)
    B_shape = getattr(B_fp8, "shape", None)
    C_shape = getattr(C_out, "shape", None)
    if A_shape is None or B_shape is None or C_shape is None:
        return  # opaque buffer types — defer to runtime
    if len(A_shape) < 2 or len(B_shape) < 2 or len(C_shape) < 2:
        raise ValueError(
            "T.fp8_scaled_matmul: operands must be at least 2D"
        )

    M = _shape_extent(A_fp8, 0)
    K = _shape_extent(A_fp8, 1)
    if transpose_B:
        N = _shape_extent(B_fp8, 0)
        K_b = _shape_extent(B_fp8, 1)
    else:
        K_b = _shape_extent(B_fp8, 0)
        N = _shape_extent(B_fp8, 1)
    M_c = _shape_extent(C_out, 0)
    N_c = _shape_extent(C_out, 1)

    if K > 0 and K_b > 0 and K != K_b:
        raise ValueError(
            f"T.fp8_scaled_matmul: K mismatch — A is {M}x{K}, "
            f"B is {'NxK' if transpose_B else 'KxN'} = {K_b}x{N}; "
            "the contracted dimension must agree"
        )
    if M > 0 and M_c > 0 and M != M_c:
        raise ValueError(
            f"T.fp8_scaled_matmul: M mismatch — A has {M} rows but C has {M_c} rows"
        )
    if N > 0 and N_c > 0 and N != N_c:
        raise ValueError(
            f"T.fp8_scaled_matmul: N mismatch — B has {N} columns but C has {N_c} columns"
        )

    sa_size = _shape_extent(A_scale, 0)
    sb_size = _shape_extent(B_scale, 0)
    if block_scale_layout is not None:
        block_scale_layout.validate_scale_shapes(
            k_extent=K,
            a_scale_shape=tuple(int(v) for v in A_scale.shape),
            b_scale_shape=tuple(int(v) for v in B_scale.shape),
            n_extent=N,
        )
        return
    if (
        M > 0
        and sa_size > 0
        and sa_size != 1
        and sa_size != M
        and sa_size < M
    ):
        raise ValueError(
            f"T.fp8_scaled_matmul: A_scale must be per-tensor (size 1) or "
            f"per-row for the local tile (size M={M}); got size {sa_size}. "
            "Pass a_scale_offset when using a larger global scale buffer."
        )
    if M > 0 and sa_size > M and a_scale_offset is None:
        raise ValueError(
            f"T.fp8_scaled_matmul: A_scale has global size {sa_size} for local tile M={M}; "
            "pass a_scale_offset explicitly."
        )
    if (
        N > 0
        and sb_size > 0
        and sb_size != 1
        and sb_size != N
        and sb_size < N
    ):
        raise ValueError(
            f"T.fp8_scaled_matmul: B_scale must be per-tensor (size 1) or "
            f"per-col for the local tile (size N={N}); got size {sb_size}. "
            "Pass b_scale_offset when using a larger global scale buffer."
        )
    if N > 0 and sb_size > N and b_scale_offset is None:
        raise ValueError(
            f"T.fp8_scaled_matmul: B_scale has global size {sb_size} for local tile N={N}; "
            "pass b_scale_offset explicitly."
        )

    # accum_dtype currently must be wider than FP8; we don't accept FP16
    # accumulators because the scaled-FMA reference always accumulates in
    # FP32 (the scales themselves are typically out-of-range for FP16).
    if accum_dtype not in ("float32", "float", "float64"):
        raise ValueError(
            f"T.fp8_scaled_matmul: accum_dtype must be float32 (or wider); got {accum_dtype!r}"
        )


@T.macro
def _fp8_scaled_matmul_macro(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_local,
    block_scale_layout=None,
    a_scale_offset=0,
    b_scale_offset=0,
):
    """Hygienic body of ``T.fp8_scaled_matmul``: dequant dot + post-scale.

    The body is parsed once at macro-decoration time and re-substituted at
    each call. Static integer extents — including ``A_scale.shape[0]`` and
    ``B_scale.shape[0]`` — drive the per-tensor-vs-per-row branch at
    expansion time, so the resulting MSL contains no runtime predicate.

    The outer ``(i, j)`` loop is ``T.Parallel`` so the layout-inference
    engine distributes the M*N output cells across ``threads`` cleanly:
    each thread owns a small slice of ``C_local`` and runs its private
    K-loop. Without ``T.Parallel`` the layout pass falls back to a
    replicated layout (every thread does the full work) which gives
    correct results but wastes work; ``T.Parallel`` matches the
    audiohacking kernel's threadgroup-tiling pattern. Ordinary FP8 scales
    stay outside the K-loop; E8M0 block scales remain inside because each
    K-block has a different scale byte.
    """
    M_dim, K_dim = A_fp8.shape
    K_dim_b, N_dim = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]

    for i, j in T.Parallel(M_dim, N_dim):
        base = C_local[i, j]
        for k in T.unroll(0, K_dim, explicit=False, unroll_factor=4):
            a_val = T.cast(A_fp8[i, k], "float32")
            b_val = T.cast(B_fp8[k, j], "float32")
            if block_scale_layout is not None:
                sa = _block_scale_value(A_scale, axis="A", col=j, k=k)
                sb = _block_scale_value(B_scale, axis="B", col=j, k=k)
                C_local[i, j] = C_local[i, j] + a_val * b_val * sa * sb
            else:
                C_local[i, j] = C_local[i, j] + a_val * b_val
        if block_scale_layout is None:
            sa = A_scale[0] if sa_size == 1 else A_scale[a_scale_offset + i]
            sb = B_scale[0] if sb_size == 1 else B_scale[b_scale_offset + j]
            C_local[i, j] = base + (C_local[i, j] - base) * sa * sb


@T.macro
def _fp8_scaled_matmul_macro_trans_b(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_local,
    block_scale_layout=None,
    a_scale_offset=0,
    b_scale_offset=0,
):
    """``transpose_B=True`` variant: B is (N, K) row-major, indexed B[j, k]."""
    M_dim, K_dim = A_fp8.shape
    N_dim, K_dim_b = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]

    for i, j in T.Parallel(M_dim, N_dim):
        base = C_local[i, j]
        for k in T.unroll(0, K_dim, explicit=False, unroll_factor=4):
            a_val = T.cast(A_fp8[i, k], "float32")
            b_val = T.cast(B_fp8[j, k], "float32")
            if block_scale_layout is not None:
                sa = _block_scale_value(A_scale, axis="A", col=j, k=k)
                sb = _block_scale_value(B_scale, axis="B", col=j, k=k)
                C_local[i, j] = C_local[i, j] + a_val * b_val * sa * sb
            else:
                C_local[i, j] = C_local[i, j] + a_val * b_val
        if block_scale_layout is None:
            sa = A_scale[0] if sa_size == 1 else A_scale[a_scale_offset + i]
            sb = B_scale[0] if sb_size == 1 else B_scale[b_scale_offset + j]
            C_local[i, j] = base + (C_local[i, j] - base) * sa * sb


@T.macro
def _fp8_scaled_matmul_m1_vecmat_metal_direct_macro(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_local,
    a_scale_offset,
    b_scale_offset,
    c_col_offset,
    simd_group_width,
    outputs_per_block,
):
    """Metal M=1 vecmat fast path: packed FP8 dot4 + SIMD-group reduction.

    The caller launches ``simd_group_width * outputs_per_block`` threads.
    Each SIMD group owns one output column, so a 128-threadgroup with 32-wide
    SIMD groups computes four columns in parallel instead of serializing all
    columns through one SIMD group.
    """
    M_dim, K_dim = A_fp8.shape
    N_dim, K_dim_b = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]
    grid_tid = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
    simd_lane = T.call_intrin("int32", "tir.metal.thread_index_in_simdgroup")
    col = T.floordiv(grid_tid, simd_group_width)
    k_words = K_dim // 4
    dot = T.alloc_var(T.float32)

    if col < N_dim:
        for kk in T.unroll(
            0,
            T.ceildiv(k_words, simd_group_width),
            explicit=False,
            unroll_factor=4,
        ):
            word_i = kk * simd_group_width + simd_lane
            # CPPMEGA pull hybrid: tl_pr_c structure + stack-c indexing
            if k_words % simd_group_width == 0:
                dot += T.metal_fp8_e4m3_dot4(
                    T.access_ptr(A_fp8[0, 0], "r", extent=K_dim),
                    T.access_ptr(B_fp8[col, 0], "r", extent=K_dim),
                    word_i,
                    word_i,
                )
            else:
                with T.If(word_i < k_words), T.Then():
                    dot += T.metal_fp8_e4m3_dot4(
                        T.access_ptr(A_fp8[0, 0], "r", extent=K_dim),
                        T.access_ptr(B_fp8[col, 0], "r", extent=K_dim),
                        word_i,
                        word_i,
                    )

        reduced = T.call_intrin("float32", "tir.metal.simd_sum", dot)
        if simd_lane == 0:
            sa = A_scale[0] if sa_size == 1 else A_scale[a_scale_offset]
            sb = B_scale[0] if sb_size == 1 else B_scale[col]
            C_local[0, col] = reduced * sa * sb


@T.macro
def _fp8_scaled_matmul_m1_vecmat_metal_macro(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_local,
    a_scale_offset,
    b_scale_offset,
    c_col_offset,
    simd_group_width,
    outputs_per_block,
):
    """Fallback M=1 vecmat macro for non-global accumulator writes."""
    M_dim, K_dim = A_fp8.shape
    N_dim, K_dim_b = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]
    grid_tid = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
    simd_lane = T.call_intrin("int32", "tir.metal.thread_index_in_simdgroup")
    local_simd_row = T.floordiv(T.cast(grid_tid, "int32"), simd_group_width) - c_col_offset
    j = local_simd_row
    col = j
    k_words = K_dim // 4
    dot = T.alloc_var(T.float32)

    if j < outputs_per_block:
        if col < N_dim:
            base = C_local[0, j]
            for kk in T.unroll(
                0,
                T.ceildiv(k_words, simd_group_width),
                explicit=False,
                unroll_factor=4,
            ):
                word_i = kk * simd_group_width + simd_lane
                # CPPMEGA pull hybrid: tl_pr_c structure + stack-c indexing
                if k_words % simd_group_width == 0:
                    dot += T.metal_fp8_e4m3_dot4(
                        T.access_ptr(A_fp8[0, 0], "r", extent=K_dim),
                        T.access_ptr(B_fp8[col, 0], "r", extent=K_dim),
                        word_i,
                        word_i,
                    )
                else:
                    with T.If(word_i < k_words), T.Then():
                        dot += T.metal_fp8_e4m3_dot4(
                            T.access_ptr(A_fp8[0, 0], "r", extent=K_dim),
                            T.access_ptr(B_fp8[col, 0], "r", extent=K_dim),
                            word_i,
                            word_i,
                        )

            reduced = T.call_intrin("float32", "tir.metal.simd_sum", dot)
            if simd_lane == 0:
                sa = A_scale[0] if sa_size == 1 else A_scale[a_scale_offset]
                sb = B_scale[0] if sb_size == 1 else B_scale[b_scale_offset + j]
                C_local[0, j] = base + reduced * sa * sb


@T.macro
def _fp8_scaled_matmul_trans_b_direct_metal_macro(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C_out,
    a_scale_offset,
    b_scale_offset,
    c_row_offset,
    c_col_offset,
    outputs_per_block,
):
    """Metal direct global matmul: transposed-B packed FP8 dot4 + post-scale.

    This mirrors the audiohacking/AppMana MSL matmul body for row-major
    ``A[M, K]`` and transposed row-major ``B[N, K]``. One Metal thread owns
    one output cell and walks K as packed uint32 words, so the scale multiply
    happens once after the dot loop instead of once per FP8 element.
    """
    M_dim, K_dim = A_fp8.shape
    N_dim, K_dim_b = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]
    col_lane = T.get_thread_binding(0)
    row_lane = T.get_thread_binding(1)
    row = c_row_offset + row_lane
    col = c_col_offset + col_lane
    k_words = K_dim // 4
    dot = T.alloc_var(T.float32)

    if row < M_dim:
        if col_lane < outputs_per_block:
            if col < N_dim:
                for word_i in T.unroll(0, k_words, explicit=False, unroll_factor=4):
                    dot += T.metal_fp8_e4m3_dot4(
                        T.access_ptr(A_fp8[row, 0], "r", extent=K_dim),
                        T.access_ptr(B_fp8[col, 0], "r", extent=K_dim),
                        word_i,
                        word_i,
                    )
                sa = A_scale[0] if sa_size == 1 else A_scale[a_scale_offset + row_lane]
                sb = B_scale[0] if sb_size == 1 else B_scale[b_scale_offset + col_lane]
                C_out[row, col] = dot * sa * sb


# CPPMEGA: legacy swap vecmat macro retained for local-fragment dispatch.
# The stack-c m1_vecmat macro requires Metal intrinsics
# `tir.metal.thread_index_in_simdgroup` and `tir.metal.fp8_e4m3_dot4`
# which are not registered in apache TVM; this legacy macro uses only
# `tirx.metal.simd_sum`, which IS registered, and is what the existing
# `test_m1_transposed_b_vecmat_lowers_to_simd_sum` regression-pins.
@T.macro
def _fp8_scaled_matmul_m1_vecmat_metal_macro_legacy(
    A_fp8, A_scale, B_fp8, B_scale, C_local
):
    """Metal M=1 vecmat path: 32 lanes reduce K with ``simd_sum``.

    Only dispatched for ``transpose_B=True`` and ``A.shape[0] == 1`` when no
    direct-global-store offsets were provided. Scales applied once after the
    SIMD-group reduction.
    """
    M_dim, K_dim = A_fp8.shape
    N_dim, K_dim_b = B_fp8.shape
    sa_size = A_scale.shape[0]
    sb_size = B_scale.shape[0]
    tx = T.get_thread_binding(0)

    for j in T.serial(N_dim):
        base = C_local[0, j]
        dot = T.alloc_var("float32", init=0.0)
        for kk in T.unroll(0, T.ceildiv(K_dim, 32), explicit=False, unroll_factor=4):
            k = kk * 32 + tx
            with T.If(k < K_dim), T.Then():
                a_val = T.cast(A_fp8[0, k], "float32")
                b_val = T.cast(B_fp8[j, k], "float32")
                dot += a_val * b_val

        reduced = T.call_intrin("float32", "tir.metal.simd_sum", dot)
        with T.If(tx == 0), T.Then():
            sa = A_scale[0] if sa_size == 1 else A_scale[0]
            sb = B_scale[0] if sb_size == 1 else B_scale[j]
            C_local[0, j] = base + reduced * sa * sb


def fp8_scaled_matmul(
    A_fp8: BufferLikeType,
    A_scale: BufferLikeType,
    B_fp8: BufferLikeType,
    B_scale: BufferLikeType,
    C_out: BufferLikeType,
    *,
    transpose_B: bool = False,
    accum_dtype: str = "float32",
    target: Optional[Target] = None,  # accepted for API compat, currently unused
    scale_format: str | None = None,
    scale_block_size: int | None = None,
    block_scale_layout: BlockScaledLayout | None = None,
    a_scale_offset=None,
    b_scale_offset=None,
    c_row_offset=None,
    c_col_offset=None,
    simd_group_width: Optional[int] = None,
    outputs_per_block: Optional[int] = None,
):
    """Scaled FP8 matmul intrinsic — accumulate scaled FP8 product into ``C``.

    Computes::

        C_out += (A_fp8 * A_scale) @ (B_fp8 * B_scale)

    where ``A_fp8`` and ``B_fp8`` are FP8 (``e4m3`` or ``e5m2``) storage
    buffers and the scales are floating-point scalars (per-tensor when
    shape is ``(1,)``, per-row / per-col otherwise). Mirrors the
    ``fp8_scaled_matmul_kernel`` algorithm from
    ``audiohacking/fp8-mps-metal`` (MIT).

    The accumulator ``C_out`` is read-modify-write — callers typically
    ``T.clear(C_local)`` once and then call this op inside the K-tile
    loop, exactly like ``T.gemm`` semantics.

    Behaviour by target
    ~~~~~~~~~~~~~~~~~~~

    The generic fallback emits scalar TIR on every target. Metal may select
    target-specific direct/vecmat fast paths for e4m3 transpose-B layouts.

    * **Metal** — ``T.cast(fp8 byte, fp32)`` lowers via
      ``__tvm_fp8_e4m3_to_half`` / ``__tvm_fp8_e5m2_to_half`` from Agent
      C's storage-only patch, then a half-to-float promotion. The
      resulting fallback MSL is functionally identical to the audiohacking
      ``fp8_scaled_matmul_kernel`` (one branch + a few shifts per byte
      per dequantization + fp32 fma). The direct and vecmat paths emit packed
      uint32 dot4 word loads plus post-dot scaling.
    * **CUDA / ROCm** — ``T.cast`` uses TVM's native FP8 path
      (``__nv_fp8_e4m3_to_half`` etc.). For Hopper / Blackwell, callers
      who want the tensor-core FP8 FMA path should use
      ``T.tcgen05_gemm_blockscaled(...)`` directly (PRs #202 / #1600);
      those gemms ingest the ``e8m0fnu`` block-scale operand explicitly.
      This op supports scalar E8M0 K/32 block scales as a portable fallback.
    * **CPU / fallback** — same scalar TIR; ``T.cast(fp8, fp32)`` lowers
      via TVM's CPU FP8 helpers.

    Args:
        A_fp8: Input A in FP8 storage. Shape ``(M, K)`` row-major.
        A_scale: Per-tensor (shape ``(1,)``) or per-row (shape ``(M,)``)
            fp32 scale for A.
        B_fp8: Input B in FP8 storage. Shape ``(K, N)`` row-major when
            ``transpose_B`` is False, otherwise ``(N, K)`` row-major.
        B_scale: Per-tensor (shape ``(1,)``) or per-col (shape ``(N,)``)
            fp32 scale for B.
        C_out: Accumulator output. Shape ``(M, N)``, fp32.
        transpose_B: Mirror ``T.gemm`` semantics. Defaults to ``False``.
        accum_dtype: Accumulator dtype for the inner GEMM (and the cast
            target for FP8 dequant). Defaults to ``"float32"``.
        target: Optional lowering target. Metal enables packed-dot4 and
            SIMD-group vecmat specializations when the layout permits.
        scale_format: Optional block-scale metadata. Only
            ``"e8m0_block_k32"`` is accepted.
        scale_block_size: Block size for ``scale_format``. Must be 32 for
            ``"e8m0_block_k32"``.
        block_scale_layout: Explicit E8M0 block-scale layout descriptor.
        a_scale_offset: Global row offset used for per-row ``A_scale``.
        b_scale_offset: Global column offset used for per-col ``B_scale``.
        c_row_offset: Global row offset for direct Metal stores.
        c_col_offset: Global column offset for direct Metal stores.
        simd_group_width: Metal vecmat SIMD-group width. Defaults to 32.
        outputs_per_block: Number of output columns owned by one block.

    Returns:
        The handle returned by the underlying ``@T.macro`` invocation,
        which the TileLang parser inlines as a ``tir.SeqStmt`` at the
        call site.

    Raises:
        TypeError: If ``A_fp8`` / ``B_fp8`` are not FP8 dtypes, or any
            scale / accumulator dtype is not a real-valued type.
        ValueError: If shapes don't agree (``K`` mismatch, ``M`` /
            ``N`` mismatch, or scale shapes that are neither 1 nor
            matching).
    """
    layout = _normalize_block_scale_layout(
        block_scale_layout,
        scale_format=scale_format,
        scale_block_size=scale_block_size,
    )
    inferred_a_scale_offset = a_scale_offset
    if inferred_a_scale_offset is None and c_row_offset is not None:
        inferred_a_scale_offset = c_row_offset
    inferred_b_scale_offset = b_scale_offset
    if inferred_b_scale_offset is None and c_col_offset is not None:
        inferred_b_scale_offset = c_col_offset
    _validate_buffers(
        A_fp8, A_scale, B_fp8, B_scale, C_out,
        transpose_B=transpose_B,
        accum_dtype=accum_dtype,
        block_scale_layout=layout,
        a_scale_offset=inferred_a_scale_offset,
        b_scale_offset=inferred_b_scale_offset,
    )

    if inferred_a_scale_offset is None:
        inferred_a_scale_offset = 0
    if inferred_b_scale_offset is None:
        inferred_b_scale_offset = 0
    direct_global_store = c_col_offset is not None
    direct_2d_global_store = c_row_offset is not None and c_col_offset is not None
    if c_row_offset is None:
        c_row_offset = 0
    if c_col_offset is None:
        c_col_offset = 0

    # CPPMEGA Z3 idea #10: before falling back to the scalar legacy M=1 vecmat
    # macro, ask the dot4-legality prover whether the packed fast path is
    # provably legal for this call site. Auto-promote ONLY when:
    #   * Metal target, transpose_B=True, M=1 (vecmat shape).
    #   * Caller has not opted out via TILELANG_DISABLE_FP8_DOT4_AUTO.
    #   * Both dot4 intrinsics are registered in this TVM build.
    #   * The prover returns True (static fast path or Z3-discharged).
    # When proven, route to the fragment-store dot4 macro
    # (``_fp8_scaled_matmul_m1_vecmat_metal_macro``); otherwise fall through
    # to the legacy scalar simd_sum path. This preserves semantics for any
    # case the prover cannot fully discharge and avoids regressing existing
    # kernels that depend on the legacy emission.
    if (
        layout is None
        and transpose_B
        and not direct_global_store
        and _is_metal_target(target)
        and _shape_extent(A_fp8, 0) == 1
        and not _fp8_dot4_auto_disabled()
        and _dot4_intrinsics_registered()
    ):
        proved, _reason = _z3_prove_dot4_legal_for_buffers(
            A_fp8, B_fp8, transpose_B=transpose_B,
        )
        if proved:
            sgw = simd_group_width
            if sgw is None:
                sgw = _target_thread_warp_size(target)
            if int(sgw) <= 0:
                raise ValueError(
                    f"T.fp8_scaled_matmul: simd_group_width must be positive, got {sgw!r}"
                )
            opb = outputs_per_block
            if opb is None:
                opb = _shape_extent(B_fp8, 0)
            if int(opb) <= 0:
                raise ValueError(
                    f"T.fp8_scaled_matmul: outputs_per_block must be positive, got {opb!r}"
                )
            return _fp8_scaled_matmul_m1_vecmat_metal_macro(
                A_fp8,
                A_scale,
                B_fp8,
                B_scale,
                C_out,
                inferred_a_scale_offset,
                inferred_b_scale_offset,
                c_col_offset,
                int(sgw),
                int(opb),
            )

    # CPPMEGA: keep swap's broader M=1 vecmat dispatch via the legacy macro
    # (uses `tirx.metal.simd_sum` only) when no direct-global-store offsets
    # are passed. The stack-c packed-dot4 vecmat path uses
    # `tir.metal.thread_index_in_simdgroup` and `tir.metal.fp8_e4m3_dot4`
    # which are not registered in apache TVM, so it only fires when the
    # direct-global-store offsets opt the caller into that fast path, OR
    # when the Z3 idea #10 prover above proved legality.
    if (
        layout is None
        and transpose_B
        and not direct_global_store
        and _is_metal_target(target)
        and _shape_extent(A_fp8, 0) == 1
    ):
        return _fp8_scaled_matmul_m1_vecmat_metal_macro_legacy(
            A_fp8, A_scale, B_fp8, B_scale, C_out
        )

    if (
        layout is None
        and transpose_B
        and direct_global_store
        and _is_metal_target(target)
        and _shape_extent(A_fp8, 0) == 1
        and _shape_extent(A_fp8, 1) > 0
        and _shape_extent(A_fp8, 1) % 4 == 0
        and str(getattr(A_fp8, "dtype", "")).startswith("float8_e4m3")
        and str(getattr(B_fp8, "dtype", "")).startswith("float8_e4m3")
    ):
        if simd_group_width is None:
            simd_group_width = _target_thread_warp_size(target)
        if int(simd_group_width) <= 0:
            raise ValueError(
                f"T.fp8_scaled_matmul: simd_group_width must be positive, got {simd_group_width!r}"
            )
        if outputs_per_block is None:
            outputs_per_block = _shape_extent(B_fp8, 0)
        if int(outputs_per_block) <= 0:
            raise ValueError(
                f"T.fp8_scaled_matmul: outputs_per_block must be positive, got {outputs_per_block!r}"
            )
        if direct_global_store:
            return _fp8_scaled_matmul_m1_vecmat_metal_direct_macro(
                A_fp8,
                A_scale,
                B_fp8,
                B_scale,
                C_out,
                inferred_a_scale_offset,
                inferred_b_scale_offset,
                c_col_offset,
                int(simd_group_width),
                int(outputs_per_block),
            )
        return _fp8_scaled_matmul_m1_vecmat_metal_macro(
            A_fp8,
            A_scale,
            B_fp8,
            B_scale,
            C_out,
            inferred_a_scale_offset,
            inferred_b_scale_offset,
            c_col_offset,
            int(simd_group_width),
            int(outputs_per_block),
        )

    if (
        layout is None
        and transpose_B
        and direct_2d_global_store
        and _is_metal_target(target)
        and _shape_extent(A_fp8, 0) > 0
        and _shape_extent(A_fp8, 1) > 0
        and _shape_extent(A_fp8, 1) % 4 == 0
        and str(getattr(A_fp8, "dtype", "")).startswith("float8_e4m3")
        and str(getattr(B_fp8, "dtype", "")).startswith("float8_e4m3")
    ):
        if outputs_per_block is None:
            outputs_per_block = _shape_extent(B_fp8, 0)
        if int(outputs_per_block) <= 0:
            raise ValueError(
                f"T.fp8_scaled_matmul: outputs_per_block must be positive, got {outputs_per_block!r}"
            )
        return _fp8_scaled_matmul_trans_b_direct_metal_macro(
            A_fp8,
            A_scale,
            B_fp8,
            B_scale,
            C_out,
            inferred_a_scale_offset,
            inferred_b_scale_offset,
            c_row_offset,
            c_col_offset,
            int(outputs_per_block),
        )

    if transpose_B:
        return _fp8_scaled_matmul_macro_trans_b(
            A_fp8,
            A_scale,
            B_fp8,
            B_scale,
            C_out,
            layout,
            inferred_a_scale_offset,
            inferred_b_scale_offset,
        )
    return _fp8_scaled_matmul_macro(
        A_fp8,
        A_scale,
        B_fp8,
        B_scale,
        C_out,
        layout,
        inferred_a_scale_offset,
        inferred_b_scale_offset,
    )
