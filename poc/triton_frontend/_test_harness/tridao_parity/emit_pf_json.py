import sys, subprocess, json as _json
sys.path.insert(0, "/home/dave/source/tilelang")

name = sys.argv[1]

# CONFIG SELECTION (RULE #1): pick the autotune config that SURVIVES Triton's
# runtime shared-cap pruning for the tile we actually compile, NOT configs[0]
# (the pruned 128x256x64 nw8 phantom). For the dstates capture the TTIR bakes a
# 64x64x32 GEMM tile.
#
# WARP COUNT (GENERIC, honor NATIVE): the autotune *config object* declares its
# own num_warps (for the dstates 64x64x32 config that is nw2), but the warp
# count the NATIVE kernel actually COMPILES AND LAUNCHES for that pinned tile is
# the value Triton bakes into the SASS `.reqntid` -- which is what the parity
# reference runs against. The parity harness pins BLOCK_SIZE_* on a DIRECT JIT
# (no num_warps kwarg) so Triton uses its compile default; for the dstates
# 64x64x32 tile this compiles to `.reqntid 128` = 4 warps. We therefore HONOR
# the native kernel's real `.reqntid // 32` (measured = 4), not the autotune
# config's declared nw2. This is generic (any captured kernel -> read the warp
# count its own native compile emits) and grounded in the kernel the reference
# actually executes, not a per-kernel hardcode. num_stages still comes from the
# surviving autotune config. RULE #1: if the native `.reqntid` cannot be read we
# RAISE rather than silently falling back to a default warp count.
#
# IMPORTANT (MEASURED): importing Triton/mamba in THIS process loads libtriton,
# which DISABLES the PtrAnalysis C++ shim (both static-link LLVM and cl::opts
# double-registration aborts). With the shim disabled the walker takes a
# degraded path that loses the routed cp.async metadata (cp_async drops 6->0).
# So we read the config + native reqntid in a SEPARATE subprocess and keep THIS
# process shim-clean: from_ttir runs with NO triton loaded. RULE #1: no silent
# loss of the cp.async path -- the capture process stays clean by construction.
_cfg_probe = (
    "import sys; sys.path.insert(0, \"/home/dave/source/tilelang\")\n"
    "from poc.triton_frontend import autotune_winning_block_config\n"
    "from mamba_ssm.ops.triton import ssd_chunk_scan as s\n"
    "import json, re, torch, triton\n"
    "ak = getattr(s, %r, None)\n"
    "if ak is None or not getattr(ak, 'configs', None):\n"
    "    print(json.dumps({})); sys.exit(0)\n"
    "BM,BN,BK = 64,64,32\n"
    "c = autotune_winning_block_config(ak, target_block={'BLOCK_SIZE_M':BM,'BLOCK_SIZE_N':BN,'BLOCK_SIZE_K':BK})\n"
    "ns = int(c['num_stages'])\n"
    "# Honor the warp count the NATIVE kernel actually compiles to for this tile.\n"
    "torch.manual_seed(0); dev='cuda'\n"
    "b,nh,hd,ds,cs,ng,seq = 1,112,64,64,64,8,4096; nc = seq//cs\n"
    "dout=torch.randn(b,seq,nh,hd,device=dev,dtype=torch.float32)\n"
    "C=torch.randn(b,seq,ng,ds,device=dev,dtype=torch.float32)\n"
    "dA=torch.randn(b,nh,nc,cs,device=dev,dtype=torch.float32)\n"
    "out=torch.empty(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)\n"
    "_jit=getattr(ak,'fn',ak)\n"
    "kc=_jit.warmup(dout,C,out,dA,None,hd,ds,cs,b,seq,nc,nh//ng,\n"
    "    dout.stride(0),dout.stride(1),dout.stride(2),dout.stride(3),\n"
    "    C.stride(0),C.stride(1),C.stride(2),C.stride(3),\n"
    "    out.stride(0),out.stride(1),out.stride(2),out.stride(3),out.stride(4),\n"
    "    dA.stride(0),dA.stride(2),dA.stride(1),dA.stride(3),0,0,\n"
    "    HAS_SEQ_IDX=False,BLOCK_SIZE_M=BM,BLOCK_SIZE_N=BN,BLOCK_SIZE_K=BK,grid=(1,))\n"
    "ptx=kc.asm['ptx']\n"
    "m=re.search(r'\\.reqntid (\\d+)',ptx)\n"
    "if not m:\n"
    "    raise RuntimeError('native kernel PTX has no .reqntid -- cannot honor native warp count (RULE #1)')\n"
    "nthreads=int(m.group(1))\n"
    "if nthreads %% 32 != 0:\n"
    "    raise RuntimeError('native .reqntid %%d not a warp multiple' %% nthreads)\n"
    "nw=nthreads//32\n"
    "print(json.dumps({'num_warps': nw, 'num_stages': ns, 'reqntid': nthreads, 'config_nw': int(c['num_warps'])}))\n"
) % (name + "_kernel")
nw = ns = None
# RULE #1 (fail loud): the warp/stage config probe must NOT silently degrade.
# A subprocess crash, non-JSON output, or an unparseable warp count RAISES so we
# never emit the PrimFunc with a SILENTLY WRONG warp count (e.g. the from_ttir
# default) masquerading as the honored-native config. The ONLY non-error path
# that legitimately yields no override is an EMPTY dict ({}), which the probe
# emits when the kernel carries no @triton.autotune configs at all -- a genuine
# "this kernel is not autotuned, keep walker defaults" signal, not a failure.
try:
    proc = subprocess.run([sys.executable, "-c", _cfg_probe], text=True,
                          capture_output=True)
except Exception as _e:
    raise RuntimeError(
        "warp-config probe subprocess failed to launch (%r); refusing to emit "
        "with a silently-defaulted warp count (RULE #1)." % (_e,))
if proc.returncode != 0:
    raise RuntimeError(
        "warp-config probe subprocess exited %d; refusing to emit with a "
        "silently-defaulted warp count (RULE #1).\nSTDERR:\n%s"
        % (proc.returncode, proc.stderr))
last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
try:
    cfg = _json.loads(last)
except Exception as _e:
    raise RuntimeError(
        "warp-config probe produced non-JSON output %r (%r); refusing to emit "
        "with a silently-defaulted warp count (RULE #1).\nSTDERR:\n%s"
        % (last, _e, proc.stderr))
if cfg:
    nw, ns = int(cfg["num_warps"]), int(cfg["num_stages"])
    print("CONFIG selected (surviving subprocess): nw=%d ns=%d block=64x64x32 "
          "[native .reqntid=%s => %d warps; autotune-config declared nw=%s]"
          % (nw, ns, cfg.get("reqntid"), nw, cfg.get("config_nw")))
else:
    print("CONFIG: kernel carries no @triton.autotune configs; keeping walker "
          "default num_warps/num_stages (no override).")

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
