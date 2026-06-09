import sys, subprocess, json as _json
sys.path.insert(0, "/home/dave/source/tilelang")

name = sys.argv[1]

# CONFIG SELECTION (RULE #1): pick the autotune config that SURVIVES Triton's
# runtime shared-cap pruning for the tile we actually compile, NOT configs[0]
# (the pruned 128x256x64 nw8 phantom). For the dstates capture the TTIR bakes a
# 64x64x32 GEMM tile -> the surviving native winner is nw2/ns4.
#
# IMPORTANT (MEASURED): importing Triton/mamba in THIS process loads libtriton,
# which DISABLES the PtrAnalysis C++ shim (both static-link LLVM and cl::opts
# double-registration aborts). With the shim disabled the walker takes a
# degraded path that loses the routed cp.async metadata (cp_async drops 6->0).
# So we read the autotune config in a SEPARATE subprocess and keep THIS process
# shim-clean: from_ttir runs with NO triton loaded. RULE #1: no silent loss of
# the cp.async path -- the capture process stays clean by construction.
_cfg_probe = (
    "import sys; sys.path.insert(0, \"/home/dave/source/tilelang\")\n"
    "from poc.triton_frontend import autotune_winning_block_config\n"
    "from mamba_ssm.ops.triton import ssd_chunk_scan as s\n"
    "import json\n"
    "ak = getattr(s, %r, None)\n"
    "if ak is not None and getattr(ak, 'configs', None):\n"
    "    c = autotune_winning_block_config(ak, target_block={'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32})\n"
    "    print(json.dumps({'num_warps': int(c['num_warps']), 'num_stages': int(c['num_stages'])}))\n"
    "else:\n"
    "    print(json.dumps({}))\n"
) % (name + "_kernel")
nw = ns = None
try:
    out = subprocess.check_output([sys.executable, "-c", _cfg_probe], text=True,
                                  stderr=subprocess.DEVNULL)
    cfg = _json.loads(out.strip().splitlines()[-1])
    if cfg:
        nw, ns = int(cfg["num_warps"]), int(cfg["num_stages"])
        print("CONFIG selected (surviving subprocess): nw=%d ns=%d block=64x64x32" % (nw, ns))
except Exception as _e:
    print("CONFIG selection skipped: %r" % (_e,))

# Now emit with NO triton loaded in this process (shim stays available).
import tilelang, tvm
from poc.triton_frontend import from_ttir
ttir = open("/tmp/ttir7/%s.ttir" % name).read()
kw = {}
if nw is not None:
    kw["num_warps"] = nw
if ns is not None:
    kw["num_stages"] = ns
pf = from_ttir(ttir, name=name+"_kernel", target="cuda", _allow_text_ttir=True, **kw)
js = tvm.ir.save_json(pf)
open("/tmp/pf_%s.json" % name, "w").write(js)
print("WROTE json len=%d" % len(js))
pf2 = tvm.ir.load_json(js)
print("ROUNDTRIP_OK type=%s nparams=%d" % (type(pf2).__name__, len(pf2.params)))
