"""Triton-native TTIR parser provider for hosts with no mlir.ir build.

Why this module exists
----------------------
``mlir_walker.parse_ttir`` needs an ``mlir.ir.Module`` whose ops expose
the property API (``op.operands``, ``op.results``, ``op.regions``,
``value.type.shape``, ``str(op)``) that the OP_TABLE emitters consume.

On a generic Linux/aarch64 host (e.g. gb10) there is no upstream
``mlir_core`` Python build, no IREE, jaxlib's ``mlir.ir`` only parses
*generic*-form MLIR (Triton prints *custom* form), and the
``_triton_frontend_cxx`` shim (which would convert custom -> generic)
needs Triton's static archives that the pip wheel does not ship.

BUT the installed Triton itself parses its own custom-form TTIR
perfectly via ``triton._C.libtriton.ir.parse_mlir_module``. We use it to
(a) VALIDATE the TTIR is well-formed, then (b) re-print canonical text
via ``module.str_nodebug()`` and parse THAT text into mlir.ir-shaped
adapter objects the existing walker/emitters consume unchanged.

Driving structure from the canonical printed text (not Triton's thin
pybind op tree) is deliberate: Triton's pybind exposes only a flat
``module.walk`` with no structured block->ops iteration (``op.get_block``
returns None for many ops, ``region`` has no ops accessor), so nested
``scf.for`` regions cannot be reconstructed from it. The printed text,
by contrast, carries the full nesting via braces/indentation and every
operand as a ``%name`` token -- a complete, unambiguous source.

RULE #1: this is fail-fast. Any op-line we cannot parse, any brace
imbalance, or any structural surprise RAISES with the offending line --
we never emit a partially-parsed (mistyped / mis-wired) module that
would silently lower to a wrong kernel.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["triton_native_available", "parse_ttir_via_triton", "TtirParseError"]


class TtirParseError(RuntimeError):
    """Raised when the canonical TTIR text cannot be faithfully parsed."""


# ---------------------------------------------------------------------------
# Provider availability + canonical re-print via Triton's own parser
# ---------------------------------------------------------------------------


def _libtriton_ir() -> Optional[Any]:
    try:
        import triton._C.libtriton as _lt  # noqa: WPS433
    except Exception:
        return None
    return getattr(_lt, "ir", None)


def triton_native_available() -> bool:
    """Whether Triton's own MLIR parser is importable on this host."""
    ir = _libtriton_ir()
    return ir is not None and hasattr(ir, "parse_mlir_module")


def _canonicalize_via_triton(ttir_text: str) -> str:
    """Round-trip TTIR through Triton's parser; return ``str_nodebug()``.

    Validates the TTIR (raises if Triton rejects it) and strips the
    ``#loc`` debug locations so the line parser sees clean op lines.
    """
    ir = _libtriton_ir()
    if ir is None or not hasattr(ir, "parse_mlir_module"):
        raise TtirParseError("triton._C.libtriton.ir.parse_mlir_module unavailable")
    ctx = ir.context()
    ir.load_dialects(ctx)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".ttir", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(ttir_text)
        tmp.flush()
        tmp.close()
        tmod = ir.parse_mlir_module(tmp.name, ctx)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return tmod.str_nodebug()


# ---------------------------------------------------------------------------
# mlir.ir-shaped adapters
# ---------------------------------------------------------------------------


_TENSOR_RE = re.compile(r"^tensor<([0-9x?]+)x(.+)>$")


class _Type:
    """mlir.ir-style type wrapper backed by the printed MLIR type string."""

    __slots__ = ("_s", "_shape", "_elt")

    def __init__(self, type_str: str) -> None:
        self._s = (type_str or "").strip()
        self._shape: Optional[Tuple[int, ...]] = None
        self._elt: Optional[str] = None
        m = _TENSOR_RE.match(self._s)
        if m is not None:
            try:
                self._shape = tuple(
                    int(d) for d in m.group(1).split("x") if d not in ("", "?")
                )
            except ValueError:
                self._shape = None
            self._elt = m.group(2)

    def __str__(self) -> str:
        return self._s

    def __repr__(self) -> str:
        return f"_Type({self._s!r})"

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @property
    def element_type(self) -> Optional["_Type"]:
        return _Type(self._elt) if self._elt is not None else None


class _Value:
    """mlir.ir-style SSA Value: ``.get_name()`` (``%12``) + ``.type``."""

    __slots__ = ("_name", "_type", "uses")

    def __init__(self, name: str, type_str: str) -> None:
        self._name = name
        self._type = _Type(type_str)
        # WalkerCtx / op_mapping check ``getattr(result, 'uses', None)``;
        # None -> conservative "consumed" (never spuriously drops results).
        self.uses = None

    def get_name(self) -> str:
        return self._name

    @property
    def type(self) -> _Type:
        return self._type

    def set_type(self, type_str: str) -> None:
        """Re-assign the SSA value's type.

        Used for generic-form region-bearing ops (tt.reduce / tt.scan)
        whose result type is printed on the region CLOSING line
        (``}) : (...) -> R``), not on the header where the result Value is
        first created.
        """
        self._type = _Type(type_str)

    def __str__(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Value) and other._name == self._name


class _NamedAttr:
    __slots__ = ("name", "attr")

    def __init__(self, name: str, attr: Any) -> None:
        self.name = name
        self.attr = attr


class _ScalarAttr:
    """Mimics MLIR Integer/Float attr: ``.value`` + ``.type`` (str)."""

    __slots__ = ("value", "type")

    def __init__(self, value: Any, type_str: str) -> None:
        self.value = value
        self.type = type_str


class _DenseSplatAttr:
    """Mimics MLIR DenseElementsAttr splat: ``is_splat`` + ``get_splat_value``.

    ``control._is_dense_attr`` keys off ``is_splat`` + ``type.shape``;
    ``_extract_dense_attr`` reads ``type.shape`` / ``type.element_type``
    and ``get_splat_value().value``.
    """

    __slots__ = ("is_splat", "type", "_scalar")

    def __init__(self, scalar: Any, tensor_type: "_Type") -> None:
        self.is_splat = True
        self.type = tensor_type
        self._scalar = scalar

    def get_splat_value(self) -> "_ScalarAttr":
        elt = self.type.element_type
        return _ScalarAttr(self._scalar, str(elt) if elt is not None else "f32")


def _build_constant_value_attr(line: str, result_type: str) -> Optional[Any]:
    """Build the ``value`` attribute for an ``arith.constant`` line.

    Custom form is ``arith.constant <literal> : <type>`` where ``<literal>``
    is either a scalar (``42`` / ``0.000000e+00``) or a splat
    ``dense<V>``. Returns a ``_ScalarAttr``-shaped string ("V : T") for
    scalars or a ``_DenseSplatAttr`` for splat-dense; ``None`` if the line
    is not a recognizable constant (caller raises).
    """
    m = re.match(
        r"^\s*(?:%[\w$.\-]+\s*=\s*)?arith\.constant\s+(.*?)\s*:\s*(.+)$",
        line.strip(),
    )
    if m is None:
        # Bool constant has no explicit type: ``arith.constant true`` /
        # ``arith.constant false`` -> i1.
        mb = re.match(
            r"^\s*(?:%[\w$.\-]+\s*=\s*)?arith\.constant\s+(true|false)\s*$",
            line.strip(),
        )
        if mb is not None:
            return f"{1 if mb.group(1) == 'true' else 0} : i1"
        return None
    literal = m.group(1).strip()
    type_str = m.group(2).strip()
    dm = re.match(r"^dense<(.+)>$", literal)
    if dm is not None:
        inner = dm.group(1).strip()
        try:
            scalar: Any = int(inner)
        except ValueError:
            try:
                scalar = float(inner)
            except ValueError:
                if inner in ("true", "false"):
                    scalar = inner == "true"
                else:
                    raise TtirParseError(
                        f"triton_native_parse: unsupported dense literal "
                        f"{inner!r} in {line!r}"
                    )
        return _DenseSplatAttr(scalar, _Type(type_str))
    # Scalar: hand back the generic-form string the emitter already parses.
    return f"{literal} : {type_str}"


class _Block:
    __slots__ = ("arguments", "operations")

    def __init__(self) -> None:
        self.arguments: List[_Value] = []
        self.operations: List["_Op"] = []


class _Region:
    __slots__ = ("blocks",)

    def __init__(self) -> None:
        self.blocks: List[_Block] = []


class _Op:
    """mlir.ir-style Operation adapter."""

    __slots__ = ("name", "operands", "results", "regions", "_text", "_attrs")

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.operands: List[_Value] = []
        self.results: List[_Value] = []
        self.regions: List[_Region] = []
        self._text = text
        self._attrs: List[_NamedAttr] = []

    @property
    def attributes(self) -> List[_NamedAttr]:
        return self._attrs

    @property
    def operation(self) -> "_Op":
        return self

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"_Op({self.name!r})"


class _ModuleAdapter:
    __slots__ = ("operation", "_text")

    def __init__(self, func_ops: List["_Op"], text: str) -> None:
        module_op = _Op("builtin.module", "module { ... }")
        region = _Region()
        block = _Block()
        for func_op in func_ops:
            block.operations.append(func_op)
        region.blocks.append(block)
        module_op.regions.append(region)
        self.operation = module_op
        self._text = text

    def __str__(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Text grammar parsing
# ---------------------------------------------------------------------------


# LHS of a producing op: ``%a, %b`` or ``%266:4`` (N-result shorthand).
_LHS_RE = re.compile(
    r"^\s*((?:%[\w$.\-]+(?::\d+)?)(?:\s*,\s*%[\w$.\-]+(?::\d+)?)*)\s*=\s*(.*)$"
)
# Op name: either custom form ``dialect.op`` or generic form ``"dialect.op"``
# (quoted). Generic form appears for ops Triton prints generically -- e.g.
# ``tt.reduce`` with a combiner region: ``%0 = "tt.reduce"(%a) <{...}> ({...})``.
_OPNAME_RE = re.compile(r'^"?([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)"?')
# SSA value token. Includes the ``#N`` result-index suffix so a reference
# to the Nth result of a multi-result op (``%266#3``) is captured whole --
# without ``#`` the operand would resolve to the unbound base ``%266``.
_SSA_TOK_RE = re.compile(r"%[\w$.\-]+(?:#\d+)?")
# Attribute block ``{key = val, ...}`` (NOT the ``<{...}>`` properties).
_ATTR_BLOCK_RE = re.compile(r"\{([^{}]*)\}")
_ATTR_PAIR_RE = re.compile(
    r"([A-Za-z_][\w]*)\s*=\s*(-?\d+|true|false|\"[^\"]*\")"
)
_FUNC_RE = re.compile(
    r"tt\.func\s+[\w ]*@([\w$.]+)\((.*?)\)\s*(?:->\s*[^{]+?)?\s*(?:attributes|\{)",
    re.S,
)

# Ops whose custom form carries a leading positional keyword attribute
# (the token immediately after the op name, before the first operand).
# Maps op name -> the attribute name the emitter looks up.
_POSITIONAL_KEYWORD_ATTRS: Dict[str, str] = {
    "arith.cmpi": "predicate",
    "arith.cmpf": "predicate",
    # tt.atomic_rmw custom form: ``tt.atomic_rmw fadd, acq_rel, gpu, %ptr,
    # %val, %mask : ...``. The FIRST bare keyword (``fadd`` / ``max`` /
    # ``min`` / ``xchg`` / ...) is the RMW op; map_tt_atomic_rmw reads it as
    # the ``rmw_op`` attribute. The memory-ordering (``acq_rel``) and scope
    # (``gpu``) tokens that follow are not needed by the emitter.
    "tt.atomic_rmw": "rmw_op",
}

_LEADING_KW_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\b")

# Generic-form properties block ``<{key = val, ...}>`` (e.g. tt.reduce's
# ``<{axis = 1 : i32}>``). Distinct from the regular ``{...}`` attr block.
_PROPS_BLOCK_RE = re.compile(r"<\{([^{}]*)\}>")
# Generic-form region block label ``^bb0(%arg0: f32, %arg1: f32):``.
_BB_LABEL_RE = re.compile(r"^\s*\^[\w$.\-]+\s*(?:\((.*)\))?\s*:\s*$")
# Closing line of a generic inline region: ``}) : (T1, ...) -> R`` (the
# operand-type tuple and the result type). May also be just ``})``.
_GENERIC_REGION_CLOSE_RE = re.compile(r"^\s*\}\)\s*(?::\s*(.*))?$")


def _parse_props_attrs(header: str) -> List[_NamedAttr]:
    """Parse the generic-form ``<{key = val, ...}>`` properties block.

    Triton prints inherent attributes (e.g. ``tt.reduce``'s ``axis``) in a
    ``<{...}>`` properties block rather than the trailing ``{...}`` attr
    block. We parse the same ``key = literal`` pairs ``_parse_attrs`` does.
    """
    attrs: List[_NamedAttr] = []
    for block in _PROPS_BLOCK_RE.findall(header):
        for k, v in _ATTR_PAIR_RE.findall(block):
            if v == "true":
                val: Any = True
            elif v == "false":
                val = False
            elif v.startswith('"'):
                val = v[1:-1]
            else:
                try:
                    val = int(v)
                except ValueError:
                    val = v
            attrs.append(_NamedAttr(k, val))
    return attrs


def _bb_block_args(label_line: str) -> List[Tuple[str, str]]:
    """Parse ``^bb0(%arg0: f32, %arg1: f32):`` -> [(name, type), ...]."""
    m = _BB_LABEL_RE.match(label_line)
    if m is None or m.group(1) is None:
        return []
    out: List[Tuple[str, str]] = []
    for part in _split_top_level(m.group(1)):
        part = re.sub(r"loc\(.*?\)", "", part).strip()
        am = re.match(r"(%[\w$.\-]+)\s*:\s*(.+)$", part)
        if am is not None:
            out.append((am.group(1), am.group(2).strip()))
    return out


def _leading_keyword(rhs_after_op: str) -> Optional[str]:
    """Return the bare keyword token right after the op name, if any.

    ``arith.cmpi sle, %a, %b : i64`` -> ``"sle"``. Returns None if the
    next token is an operand (``%``) or a literal.
    """
    m = _LEADING_KW_RE.match(rhs_after_op)
    return m.group(1) if m is not None else None


def _result_names(lhs: str) -> List[str]:
    """Expand ``%266:4`` to ``['%266#0', '%266#1', '%266#2', '%266#3']``."""
    names: List[str] = []
    for tok in (t.strip() for t in lhs.split(",")):
        m = re.match(r"(%[\w$.\-]+):(\d+)$", tok)
        if m is not None:
            base, n = m.group(1), int(m.group(2))
            names.extend(f"{base}#{i}" for i in range(n))
        else:
            names.append(tok)
    return names


def _split_result_types(type_sig: str, n_results: int) -> List[str]:
    sig = type_sig.strip()
    if not sig:
        return ["" for _ in range(max(n_results, 1))]
    if "->" in sig:
        sig = sig.rsplit("->", 1)[1].strip()
    elif " to " in f" {sig} ":
        sig = sig.split(" to ")[-1].strip()
    if n_results <= 1:
        # Some single-result ops print MULTIPLE types in the trailing
        # signature -- e.g. ``arith.select %c, %a, %b : tensor<...xi1>,
        # tensor<...xf32>`` lists the condition type then the result type.
        # The RESULT type is the LAST top-level segment. (For a genuine
        # single-type signature this is a no-op since there's one segment.)
        segs = _split_top_level(sig)
        return [segs[-1].strip() if segs else sig]
    # Split top-level commas only (tensor<...> uses 'x', tuples use ',').
    parts = _split_top_level(sig)
    if len(parts) == n_results:
        return [p.strip() for p in parts]
    return [sig] * n_results


def _split_top_level(s: str) -> List[str]:
    """Split ``s`` on commas not nested inside <> () [] {}."""
    out: List[str] = []
    depth = 0
    cur = []
    for ch in s:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _operand_tokens(rhs_after_op: str) -> List[str]:
    """Extract operand ``%name`` tokens from the operand portion of a line.

    The operand portion is everything before the trailing `` : <type>``
    signature and before any ``{...}`` attribute block. iter_args operands
    (inside ``iter_args(...)``) are intentionally NOT treated as plain
    operands -- scf.for's emitter consumes them via the region block args.
    """
    # Cut the trailing type signature (last `` : ``) and attribute blocks.
    body = rhs_after_op
    if " : " in body:
        body = body.rsplit(" : ", 1)[0]
    body = _ATTR_BLOCK_RE.sub(" ", body)
    return _SSA_TOK_RE.findall(body)


def _parse_attrs(line: str) -> List[_NamedAttr]:
    attrs: List[_NamedAttr] = []
    for block in _ATTR_BLOCK_RE.findall(line):
        for k, v in _ATTR_PAIR_RE.findall(block):
            if v == "true":
                val: Any = True
            elif v == "false":
                val = False
            elif v.startswith('"'):
                val = v[1:-1]
            else:
                try:
                    val = int(v)
                except ValueError:
                    val = v
            attrs.append(_NamedAttr(k, val))
    return attrs


def parse_func_signature(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    m = _FUNC_RE.search(text)
    if m is None:
        raise TtirParseError("triton_native_parse: no tt.func declaration found")
    sym = m.group(1)
    body = m.group(2)
    args: List[Tuple[str, str]] = []
    for part in _split_top_level(body):
        part = part.strip()
        if not part:
            continue
        am = re.match(r"(%[\w$.\-]+)\s*:\s*(.+)$", part)
        if am is None:
            raise TtirParseError(
                f"triton_native_parse: cannot parse func arg {part!r}"
            )
        args.append((am.group(1), am.group(2).strip()))
    return sym, args


# ---------------------------------------------------------------------------
# Recursive body builder (brace-nesting aware)
# ---------------------------------------------------------------------------


def _iter_arg_bindings(for_line: str) -> List[str]:
    """Return the iter_args block-arg names from an scf.for header line."""
    m = re.search(r"iter_args\((.*?)\)\s*->", for_line)
    if m is None:
        return []
    out: List[str] = []
    for part in _split_top_level(m.group(1)):
        am = re.match(r"\s*(%[\w$.\-]+)\s*=", part)
        if am is not None:
            out.append(am.group(1))
    return out


def _iter_arg_inits(for_line: str) -> List[str]:
    """Return the iter_args INIT operand names (RHS of ``%blk = %init``)."""
    m = re.search(r"iter_args\((.*?)\)\s*->", for_line)
    if m is None:
        return []
    out: List[str] = []
    for part in _split_top_level(m.group(1)):
        am = re.search(r"=\s*(%[\w$.\-]+)", part)
        if am is not None:
            out.append(am.group(1))
    return out


def _for_induction_var(for_line: str) -> Optional[str]:
    m = re.match(r"\s*(?:%[\w$.\-:]+\s*=\s*)?scf\.for\s+(%[\w$.\-]+)\s*=",
                 for_line)
    return m.group(1) if m is not None else None


def _for_result_types(for_line: str) -> List[str]:
    """Parse the ``-> (T1, ..., Tn)`` result/iter-arg types of an scf op."""
    m = re.search(r"->\s*\((.*)\)\s*(?::|{)", for_line)
    if m is None:
        # Single-result form ``-> T`` (no parens).
        m2 = re.search(r"->\s*([^({:][^:{]*?)\s*(?::|{)", for_line)
        if m2 is not None:
            return [m2.group(1).strip()]
        return []
    return [p.strip() for p in _split_top_level(m.group(1))]


class _Cursor:
    __slots__ = ("lines", "i")

    def __init__(self, lines: List[str]) -> None:
        self.lines = lines
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.lines[self.i] if self.i < len(self.lines) else None

    def next(self) -> Optional[str]:
        line = self.peek()
        if line is not None:
            self.i += 1
        return line


def _build_block(cur: _Cursor, value_by_name: Dict[str, _Value],
                 block: _Block, stop_at_brace: bool,
                 region: Optional["_Region"] = None) -> None:
    """Consume op lines into ``block`` until a closing ``}`` (if nested).

    A ``^bbN:`` block label starts a NEW block in the same region. This
    matters for helper funcs whose entry block is followed by an
    unreachable ``^bb1`` block (``ub.poison`` + a sentinel ``tt.return``):
    ``emit_tt_call`` only walks block 0, so the dead block's ops must land
    in a separate block, not the entry block.
    """
    while True:
        raw = cur.peek()
        if raw is None:
            if stop_at_brace:
                raise TtirParseError(
                    "triton_native_parse: unexpected EOF inside a region "
                    "(missing closing brace)"
                )
            return
        line = raw.strip()
        if line == "}" or line.endswith("}") and line.replace("}", "").strip() == "":
            cur.next()
            if stop_at_brace:
                return
            # Stray close at top level: ignore the module's closing brace.
            continue
        if line.startswith("^"):
            # New basic block in the same region (e.g. unreachable ^bb1).
            cur.next()
            if region is not None:
                block = _Block()
                region.blocks.append(block)
            continue
        if not line or line.startswith("module") or line.startswith("tt.func"):
            cur.next()
            continue
        cur.next()
        _emit_line(cur, line, value_by_name, block)


def _build_generic_region(
    cur: _Cursor,
    value_by_name: Dict[str, _Value],
    adapted: _Op,
) -> None:
    """Build the INLINE region of a generic-form region-bearing op.

    Handles the combiner region Triton prints for ``tt.reduce`` / ``tt.scan``::

        %0 = "tt.reduce"(%input) <{axis = 1 : i32}> ({
        ^bb0(%arg0: f32, %arg1: f32):
          %2 = arith.addf %arg0, %arg1 : f32
          tt.reduce.return %2 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>

    The closing ``}) : (...) -> R`` line carries the op's result type, which
    is NOT on the header (where the result Value was created). We parse it
    here and rewrite the result Value's type so ``map_tt_reduce`` reads the
    right output dtype.

    The combiner detector (``op_emitters.reduction._detect_via_mlir``) walks
    ``op.regions[].blocks[].operations[]`` and reads the inner ``arith.*``
    op name, so we materialise that exact structure.
    """
    region = _Region()
    child = _Block()
    region.blocks.append(child)
    adapted.regions.append(region)
    while True:
        raw = cur.peek()
        if raw is None:
            raise TtirParseError(
                "triton_native_parse: unexpected EOF inside a generic inline "
                "region (missing ``})`` close)"
            )
        line = raw.strip()
        close = _GENERIC_REGION_CLOSE_RE.match(line)
        if close is not None:
            cur.next()
            # The closing line carries ``: (operandTypes) -> resultType``.
            sig = close.group(1)
            if sig and "->" in sig and adapted.results:
                result_sig = sig.rsplit("->", 1)[1].strip()
                # A multi-result reducer (e.g. welford) prints
                # ``-> (R0, R1, R2)``; split and assign per result.
                rs = result_sig.strip()
                if rs.startswith("(") and rs.endswith(")"):
                    parts = _split_top_level(rs[1:-1])
                else:
                    parts = [rs]
                for k, rv in enumerate(adapted.results):
                    if k < len(parts):
                        rv.set_type(parts[k].strip())
            return
        if line.startswith("^"):
            # Block label with combiner block args (``^bb0(%a: f32, %b: f32):``).
            cur.next()
            for argname, type_str in _bb_block_args(line):
                bv = _Value(argname, type_str)
                child.arguments.append(bv)
                value_by_name[argname] = bv
            continue
        if not line:
            cur.next()
            continue
        cur.next()
        _emit_line(cur, line, value_by_name, child)


def _emit_line(cur: _Cursor, line: str, value_by_name: Dict[str, _Value],
               block: _Block) -> None:
    result_names: List[str] = []
    rhs = line
    m = _LHS_RE.match(line)
    if m is not None:
        result_names = _result_names(m.group(1))
        rhs = m.group(2).strip()
    opm = _OPNAME_RE.match(rhs)
    if opm is None:
        # A stray ``}) : (...) -> ...`` line is the CLOSE of a generic-form
        # inline region (``"tt.reduce"(...) ({ ^bb0: ... })``) that we do
        # not yet parse. Name the construct precisely so the boundary is
        # unambiguous (RULE #1: fail loud, never a partial/wrong kernel).
        if rhs.lstrip().startswith("})"):
            raise TtirParseError(
                "triton_native_parse: hit the close of a GENERIC-FORM inline "
                "region (``}) : (...) -> ...``). This means a region-bearing "
                "op (tt.reduce / tt.scan combiner) was printed in generic "
                "form, which the text parser does not yet reconstruct. "
                "Routing this kernel needs generic-form region parsing for "
                f"tt.reduce/tt.scan. Offending line: {line!r}"
            )
        raise TtirParseError(
            f"triton_native_parse: cannot extract op name from line: {line!r}"
        )
    op_name = opm.group(1)
    rhs_after_op = rhs[opm.end():]

    # Trailing type signature (last `` : ``) -> per-result types.
    type_sig = ""
    # A region-opening op (scf.for/scf.if) ends its header with `` {``;
    # strip that before reading the type signature.
    header = rhs_after_op
    # A generic-form region-bearing op (tt.reduce / tt.scan combiner) ends
    # its header with ``({`` -- the open of an INLINE region. Its result
    # type signature lives on the CLOSING ``}) : (...) -> R`` line, not the
    # header. Detect this BEFORE the scf ``{`` test so we don't confuse the
    # two: scf.for opens a region with a bare `` {`` and carries result
    # types in a ``-> (...)`` clause on the header itself.
    opens_generic_region = header.rstrip().endswith("({")
    opens_region = (not opens_generic_region) and header.rstrip().endswith("{")
    if opens_generic_region:
        header = header.rstrip()[:-2].rstrip()
    elif opens_region:
        header = header.rstrip()[:-1].rstrip()
    if " : " in header:
        type_sig = header.rsplit(" : ", 1)[1].strip()
    result_types = _split_result_types(type_sig, max(len(result_names), 1))

    adapted = _Op(op_name, line)
    adapted._attrs.extend(_parse_attrs(line))
    # Generic-form ops carry inherent attrs (e.g. tt.reduce's ``axis``) in a
    # ``<{...}>`` properties block; surface them as named attrs so the
    # emitter's ``_attrs(op)`` finds ``axis``.
    if opens_generic_region:
        adapted._attrs.extend(_parse_props_attrs(line))

    # Positional keyword attributes (custom form puts these as bare tokens
    # right after the op name, e.g. ``arith.cmpi sle, %a, %b``). The
    # emitters read them via ``_attrs_with_properties_shared``; synthesize
    # the named attr here so they resolve.
    _positional = _POSITIONAL_KEYWORD_ATTRS.get(op_name)
    if _positional is not None:
        kw = _leading_keyword(rhs_after_op)
        if kw is not None:
            adapted._attrs.append(_NamedAttr(_positional, kw))

    # tt.call custom form: ``tt.call @symbol(operands) : (...) -> ...``.
    # The emitter resolves the callee via ``_parse_callee_attr`` which
    # accepts a ``callee`` attr; synthesize it so the inline-expansion
    # finds the right helper tt.func.
    if op_name == "tt.call":
        cm = re.search(r"@([\w$.]+)", rhs_after_op)
        if cm is not None:
            adapted._attrs.append(_NamedAttr("callee", "@" + cm.group(1)))

    # arith.constant carries its value as a positional literal (custom
    # form), not a ``{key = val}`` attr; synthesize the ``value`` attr the
    # emitter expects (scalar string or dense-splat object).
    if op_name == "arith.constant":
        cval = _build_constant_value_attr(
            line, result_types[0] if result_types else ""
        )
        if cval is None:
            raise TtirParseError(
                f"triton_native_parse: cannot parse arith.constant: {line!r}"
            )
        adapted._attrs.append(_NamedAttr("value", cval))

    # Operands (excluding iter_args init values for scf.for, which are
    # surfaced as region block args).
    if op_name == "scf.for":
        # MLIR scf.for operands are ``[lb, ub, step, init0, init1, ...]``.
        # The lb/ub/step appear before ``iter_args``; the per-iter init
        # values are the RHS of each ``%blkarg = %init`` inside iter_args.
        # map_scf_for reads ``operands[3:]`` as the inits, so we must keep
        # this exact order.
        # ``scf.for %iv = %lb to %ub step %step`` -> operands [lb, ub, step]
        # (the induction var %iv is a block arg, NOT an operand).
        lb_ub_step = re.search(
            r"scf\.for\s+%[\w$.\-]+\s*=\s*(%[\w$.\-]+)\s+to\s+"
            r"(%[\w$.\-]+)\s+step\s+(%[\w$.\-]+)",
            line,
        )
        if lb_ub_step is None:
            raise TtirParseError(
                f"triton_native_parse: cannot parse scf.for bounds: {line!r}"
            )
        operand_toks = [lb_ub_step.group(1), lb_ub_step.group(2),
                        lb_ub_step.group(3)]
        operand_toks.extend(_iter_arg_inits(line))
    else:
        operand_toks = _operand_tokens(header)
    for tok in operand_toks:
        adapted.operands.append(
            value_by_name.get(tok) or _Value(tok, "")
        )

    # scf.for/scf.if carry their result types in the ``-> (T1, ..., Tn)``
    # clause, NOT the trailing ``: <inductiontype>``. Override result_types
    # from that clause so the carry buffers get the right dtype/shape.
    carry_types: List[str] = []
    if op_name in ("scf.for", "scf.if") and opens_region:
        carry_types = _for_result_types(line)
        if carry_types and len(carry_types) == len(result_names):
            result_types = carry_types

    for k, rname in enumerate(result_names):
        rtype = result_types[k] if k < len(result_types) else ""
        rv = _Value(rname, rtype)
        adapted.results.append(rv)
        value_by_name[rname] = rv

    block.operations.append(adapted)

    if opens_region:
        # scf.for / scf.if body. Build one region; bind induction var +
        # iter_args as block arguments.
        region = _Region()
        child = _Block()
        if op_name == "scf.for":
            iv = _for_induction_var(line)
            if iv is not None:
                bv = _Value(iv, "i32")
                child.arguments.append(bv)
                value_by_name[iv] = bv
            # iter_arg block-args carry the SAME types as the for results
            # (the ``-> (...)`` clause). Assigning them lets the carry-buffer
            # allocation in map_scf_for pick the right dtype/shape.
            ia_names = _iter_arg_bindings(line)
            for ia_idx, ia in enumerate(ia_names):
                ia_type = carry_types[ia_idx] if ia_idx < len(carry_types) else ""
                bv = _Value(ia, ia_type)
                child.arguments.append(bv)
                value_by_name[ia] = bv
        region.blocks.append(child)
        adapted.regions.append(region)
        _build_block(cur, value_by_name, child, stop_at_brace=True)

    if opens_generic_region:
        # Inline combiner region of a generic-form op (tt.reduce / tt.scan).
        _build_generic_region(cur, value_by_name, adapted)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def parse_ttir_via_triton(ttir_text: str) -> Optional[Any]:
    """Parse custom-form TTIR into an mlir.ir-shaped module via Triton.

    Returns a :class:`_ModuleAdapter` consumable by
    ``mlir_walker.walk_module`` / ``TTIRWalker``, or ``None`` if Triton's
    native parser is unavailable (caller falls through to other providers).

    Raises :class:`TtirParseError` on any structural surprise -- never a
    partial module (RULE #1).
    """
    if not triton_native_available():
        return None

    nodebug = _canonicalize_via_triton(ttir_text)

    # A TTIR module may contain multiple tt.func ops: the entry kernel plus
    # helper funcs (e.g. ``@triton.language.standard.cdiv...``) that
    # ``tt.call`` inline-expands. Parse them ALL so the module pre-pass can
    # seed ctx.callees.
    func_ops: List[_Op] = []
    for fidx, (header_line, sig_line, body_lines) in enumerate(_iter_funcs(nodebug)):
        sym, func_args = _parse_func_header(sig_line)
        # SSA names are function-local in MLIR, so ``%0``/``%1`` recur in
        # every func. The emitters bind results into ONE shared
        # ``ctx.value_map`` keyed (also) by the printed ``%name`` string,
        # so a helper func's ``%1`` would clobber the entry kernel's ``%1``
        # when ``tt.call`` inline-expands the helper -- producing a
        # wrong-dtype/garbage operand later (a RULE #1 silent-corruption
        # bug). We make names globally unique by prefixing every SSA token
        # in NON-entry funcs with the func index (``%1`` -> ``%f1$1``). The
        # entry kernel (fidx 0) keeps raw names so caller-seeded
        # ``arg_buffer_shapes`` keyed by ``%argN`` still match.
        prefix = "" if fidx == 0 else f"%f{fidx}$"
        if prefix:
            # Rename the header ONCE; re-parsing the renamed header already
            # yields prefixed arg names (do NOT prefix them a second time).
            header_line = _rename_ssa(header_line, prefix)
            sym, func_args = _parse_func_header(header_line)
            body_lines = [_rename_ssa(b, prefix) for b in body_lines]
        value_by_name: Dict[str, _Value] = {}
        func_op = _Op("tt.func", header_line)
        func_op._attrs.append(_NamedAttr("sym_name", sym))
        func_region = _Region()
        entry_block = _Block()
        func_region.blocks.append(entry_block)
        func_op.regions.append(func_region)
        for argname, type_str in func_args:
            v = _Value(argname, type_str)
            entry_block.arguments.append(v)
            value_by_name[argname] = v
        cur = _Cursor(body_lines)
        _build_block(cur, value_by_name, entry_block, stop_at_brace=False,
                     region=func_region)
        func_ops.append(func_op)

    if not func_ops:
        raise TtirParseError(
            "triton_native_parse: no tt.func ops found in module"
        )

    return _ModuleAdapter(func_ops, nodebug)


def _rename_one(name: str, prefix: str) -> str:
    """Prefix a single ``%name`` token (keeps any ``:N`` result count)."""
    if not name.startswith("%"):
        return name
    base = name[1:]
    return f"{prefix}{base}"


def _rename_ssa(line: str, prefix: str) -> str:
    """Prefix every ``%ssa`` token in ``line`` with ``prefix``.

    ``%12`` -> ``%f2$12``. Leaves ``@symbol`` refs, types, and attrs
    untouched (only ``%`` tokens are SSA values in MLIR text).
    """
    return _SSA_TOK_RE.sub(lambda m: f"{prefix}{m.group(0)[1:]}", line)


def _iter_funcs(nodebug: str):
    """Yield ``(header_line, full_signature_text, body_lines)`` per tt.func.

    A tt.func header may span multiple physical lines (long arg lists),
    but Triton's ``str_nodebug`` prints the whole signature on ONE line
    ending in `` {``. We slice each func body by brace-depth from its
    opening ``{`` to the matching ``}``.
    """
    lines = nodebug.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("tt.func") and stripped.endswith("{"):
            header = stripped
            # Body is from i+1 to the matching close brace (depth tracking).
            depth = 1
            body: List[str] = []
            j = i + 1
            while j < n and depth > 0:
                ls = lines[j].strip()
                # Count net brace delta on this line (region opens/closes).
                opens = ls.count("{")
                closes = ls.count("}")
                if depth + opens - closes <= 0 and closes > 0 and opens == 0 \
                        and ls == "}":
                    # The func's own closing brace.
                    depth = 0
                    break
                depth += opens - closes
                body.append(lines[j])
                j += 1
            yield header, header, body
            i = j + 1
            continue
        i += 1


def _parse_func_header(header_line: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Parse ``tt.func ... @sym(args) ... {`` -> (sym, [(arg, type), ...])."""
    return parse_func_signature(header_line)


def _func_decl_line(nodebug: str) -> str:
    for line in nodebug.splitlines():
        if line.strip().startswith("tt.func"):
            return line.strip()
    return "tt.func"


def _func_body_lines_UNUSED(nodebug: str) -> List[str]:
    """Return the lines inside the entry func body (between its braces)."""
    lines = nodebug.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("tt.func") and line.rstrip().endswith("{"):
            start = idx + 1
            break
    if start is None:
        raise TtirParseError(
            "triton_native_parse: could not locate tt.func opening brace"
        )
    # The remaining lines include the func body and a final module ``}``;
    # _build_block stops on the func's own closing brace via depth tracking.
    return lines[start:]
