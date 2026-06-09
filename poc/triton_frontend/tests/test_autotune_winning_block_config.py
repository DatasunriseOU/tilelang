"""Local unit verification of autotune_winning_block_config (no GPU, pure logic).

Verifies the GENERIC capture-side autotune-BLOCK reader added in the BlockFix:
a @triton.autotune-wrapped kernel's winning Config -> {BLOCK_SIZE_*, num_warps,
num_stages}; missing/empty configs -> RAISES (RULE #1 fail-loud, never silently
default to the smallest tile).
"""
import os
import re

_INIT = os.path.join(os.path.dirname(__file__), os.pardir, "__init__.py")
src = open(_INIT).read()
m = re.search(
    r"def autotune_winning_block_config\(.*?\n    return out\n", src, re.S
)
assert m, "could not extract autotune_winning_block_config"
from typing import Any, Dict  # noqa: E402

ns = {"Any": Any, "Dict": Dict}
exec(m.group(0), ns)
fn = ns["autotune_winning_block_config"]


class _Config:
    def __init__(self, kwargs, num_warps, num_stages):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages


class _Autotuner:
    def __init__(self, configs):
        self.configs = configs


# 1. winning config (first entry) is honoured, with warps/stages.
k = _Autotuner([
    _Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, 8, 3),
    _Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 4, 2),
])
got = fn(k)
assert got == {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "num_warps": 8,
    "num_stages": 3,
}, got
print("OK winning config -> 128x256x64 nw=8 ns=3")

# 2. no configs -> RAISE (RULE #1, never silently default to smallest tile).
try:
    fn(_Autotuner([]))
    raise SystemExit("FAIL: empty configs did not raise")
except ValueError as e:
    print("OK empty configs RAISES:", str(e)[:60])

# 2b. bare object with no .configs -> RAISE.
class _Bare:
    __name__ = "bare_jit"


try:
    fn(_Bare())
    raise SystemExit("FAIL: missing configs did not raise")
except ValueError as e:
    print("OK missing configs RAISES:", str(e)[:60])

# 3. winning config with empty kwargs -> RAISE (no BLOCK_SIZE to honour).
try:
    fn(_Autotuner([_Config({}, 4, 2)]))
    raise SystemExit("FAIL: empty kwargs did not raise")
except ValueError as e:
    print("OK empty-kwargs config RAISES:", str(e)[:60])

print("ALL_BLOCK_READER_TESTS_PASS")
