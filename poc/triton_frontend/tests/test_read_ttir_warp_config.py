"""Local unit verification of _read_ttir_warp_config (no GPU, pure logic).

Verifies the GENERIC autotune-honoring warp-config reader added in the TileFix:
text-TTIR -> (None, None) (defaults preserved); MLIR module attrs -> honored;
malformed/non-positive warp attr -> RAISES (RULE #1 fail-loud).
"""
import os
# Resolve poc/triton_frontend/__init__.py relative to THIS test file so the
# unit test is path-independent (works on any checkout / on gb10).
_INIT = os.path.join(os.path.dirname(__file__), os.pardir, "__init__.py")
# We can't fully import the package (pulls mlir_walker + heavy deps); instead
# exec just the function source extracted from __init__.py.
src = open(_INIT).read()
import re
m = re.search(r"def _read_ttir_warp_config\(.*?\n    return num_warps, num_stages\n", src, re.S)
assert m, "could not extract function"
from typing import Any, Optional, Tuple
ns = {"Any": Any, "Optional": Optional, "Tuple": Tuple}
exec(m.group(0), ns)
fn = ns["_read_ttir_warp_config"]

# 1. text-TTIR stand-in (a plain string) -> (None, None)
assert fn("module { tt.func ... }") == (None, None), "text-TTIR should give defaults"
print("OK text-TTIR -> (None, None)")

# 2. mock MLIR module with attributes -> (8, 3)
class IntAttr:
    def __init__(self, v): self.value = v
class Attrs:
    def __init__(self, d): self._d = d
    def __getitem__(self, k):
        if k in self._d: return self._d[k]
        raise KeyError(k)
class Op:
    def __init__(self, d): self.attributes = Attrs(d)
class Mod:
    def __init__(self, d): self.operation = Op(d)
mod = Mod({"ttg.num-warps": IntAttr(8), "ttg.num-stages": IntAttr(3)})
assert fn(mod) == (8, 3), fn(mod)
print("OK MLIR attrs ttg.num-warps=8/ttg.num-stages=3 -> (8, 3)")

# 2b. alternate spelling num_warps
mod2 = Mod({"num_warps": IntAttr(4), "num_stages": IntAttr(2)})
assert fn(mod2) == (4, 2), fn(mod2)
print("OK MLIR attrs num_warps=4/num_stages=2 -> (4, 2)")

# 3. malformed (non-positive) -> RAISE (RULE #1 fail loud)
bad = Mod({"ttg.num-warps": IntAttr(0)})
try:
    fn(bad); raise SystemExit("FAIL: non-positive did not raise")
except ValueError as e:
    print("OK non-positive warp attr RAISES:", str(e)[:60])

# 3b. malformed (unparseable) -> RAISE
class Junk:
    def __str__(self): return "not-a-number"
bad2 = Mod({"ttg.num-warps": Junk()})
try:
    fn(bad2); raise SystemExit("FAIL: unparseable did not raise")
except ValueError as e:
    print("OK unparseable warp attr RAISES:", str(e)[:60])

print("ALL_WARP_READER_TESTS_PASS")
