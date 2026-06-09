"""Local unit verification of autotune_winning_block_config (no GPU, pure logic).

Verifies the SURVIVING-config selection (RULE #1, MEASURED): configs[0] can be a
PHANTOM whose dynamic shared exceeds the device cap (Triton PRUNES it at runtime).
The selector estimates each config's GEMM shared footprint, keeps only configs
that FIT the cap, and -- given the tile actually compiled (``target_block``) --
returns the matching surviving config's nw/ns (for the dstates 64x64x32 capture
that is the native winner nw2/ns4), NOT the pruned 128x256x64 nw8 configs[0].
Missing/empty configs -> RAISES (fail-loud, never silently default).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
from poc.triton_frontend import autotune_winning_block_config as fn  # noqa: E402


class _Config:
    def __init__(self, kwargs, num_warps, num_stages):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages


class _Autotuner:
    def __init__(self, configs):
        self.configs = configs


# The REAL §P1 dstates autotune list (fp32). configs[0] 128x256x64 nw8 needs
# ~294 KB shared >> the 101376 B cap -> PRUNED. The surviving native winner for
# the captured 64x64x32 tile is nw2/ns4.
_DSTATES = _Autotuner([
    _Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, 8, 3),
    _Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, 4, 4),
    _Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, 4, 4),
    _Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 4, 4),
    _Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, 4, 4),
    _Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, 4, 4),
    _Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, 2, 5),
    _Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 2, 5),
    _Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 2, 4),
])

# 1. target the captured 64x64x32 tile -> surviving native winner nw2/ns4.
got = fn(_DSTATES, target_block={"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32})
assert got == {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "num_warps": 2, "num_stages": 4,
}, got
print("OK target 64x64x32 -> surviving 64x64x32 nw2 ns4 (NOT pruned configs[0])")

# 1b. the pruned phantom 128x256x64 is NEVER returned (no target -> largest
# fitting, but the OOM tile is excluded).
got2 = fn(_DSTATES)
assert not (got2["BLOCK_SIZE_M"] == 128 and got2["BLOCK_SIZE_N"] == 256), got2
print("OK phantom 128x256x64 (>cap) pruned; selected", got2["BLOCK_SIZE_M"], "x", got2["BLOCK_SIZE_N"])

# 1c. a tiny cap prunes EVERYTHING -> RAISE (never launch an OOM tile).
try:
    fn(_DSTATES, device_shared_cap=1024)
    raise SystemExit("FAIL: no-fit did not raise")
except ValueError as e:
    print("OK no-fit RAISES:", str(e)[:50])

# 1d. target_block with no matching surviving config -> RAISE.
try:
    fn(_DSTATES, target_block={"BLOCK_SIZE_M": 999, "BLOCK_SIZE_N": 999, "BLOCK_SIZE_K": 32})
    raise SystemExit("FAIL: bad target did not raise")
except ValueError as e:
    print("OK bad target RAISES:", str(e)[:50])

# 2. no configs -> RAISE (RULE #1, never silently default to smallest tile).
try:
    fn(_Autotuner([]))
    raise SystemExit("FAIL: empty configs did not raise")
except ValueError as e:
    print("OK empty configs RAISES:", str(e)[:60])


class _Bare:
    __name__ = "bare_jit"


try:
    fn(_Bare())
    raise SystemExit("FAIL: missing configs did not raise")
except ValueError as e:
    print("OK missing configs RAISES:", str(e)[:60])

# 3. surviving config with empty kwargs -> RAISE (no BLOCK_SIZE to honour).
try:
    fn(_Autotuner([_Config({}, 4, 2)]))
    raise SystemExit("FAIL: empty kwargs did not raise")
except ValueError as e:
    print("OK empty-kwargs config RAISES:", str(e)[:60])

print("ALL_BLOCK_READER_TESTS_PASS")
