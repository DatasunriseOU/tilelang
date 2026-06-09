"""2nd-KERNEL GENERIC: build dc at num_warps=4 vs 8 -> show the SAME warp-partition
mechanism (compute_warp_partition) responds to num_warps for a DIFFERENT kernel,
dropping spills. Proves the fix is generic codegen, not a dstates hack."""
import os, sys, re
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton, tilelang, tvm
from poc.triton_frontend import from_ttir
ttir = open("/tmp/ttir7/_chunk_scan_bwd_dc.ttir").read()
def build_census(nw, ns, tag):
    os.environ["TL_FORCE_CP_ASYNC"] = "1"
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dc_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=True, num_warps=nw, num_stages=ns)
    k = tilelang.compile(pf, target="cuda")
    try:
        p="/tmp/dcn_%s.sass"%tag; k.export_sass(p); sass=open(p).read()
    except Exception:
        sass=k._get_sass()
    c=lambda pat: len(re.findall(pat, sass))
    d=dict(HMMA=c(r"\bHMMA"),LDGSTS=c(r"\bLDGSTS"),IMAD=c(r"\bIMAD"),ISETP=c(r"\bISETP"),spill=c(r"\bSTL\b")+c(r"\bLDL\b"))
    print("DC_%s nw=%d HMMA=%d LDGSTS=%d IMAD=%d ISETP=%d spill=%d"%(tag,nw,d["HMMA"],d["LDGSTS"],d["IMAD"],d["ISETP"],d["spill"]),flush=True)
    return d
print("=== dc: warp-partition mechanism responds to num_warps (2nd kernel) ===",flush=True)
c4=build_census(4,2,"NW4"); c8=build_census(8,3,"NW8")
print("DCGEN dc_NW4_spill=%d dc_NW8_spill=%d dc_NW4_HMMA=%d dc_NW8_HMMA=%d SPILL_DELTA=%.2fx"%(
    c4["spill"],c8["spill"],c4["HMMA"],c8["HMMA"],(c4["spill"]/c8["spill"] if c8["spill"] else 0)),flush=True)
print("DONE",flush=True)
