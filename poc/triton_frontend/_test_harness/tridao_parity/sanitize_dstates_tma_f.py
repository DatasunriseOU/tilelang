"""compute-sanitizer EXEC probe: run ONLY the grounded C-tile TMA (OPT) kernel
once under sm_121f + dyn-shared so the sanitizer pinpoints the faulting PC.
Minimal: build OPT, single launch, sync. No OFF/timing (keep sanitizer fast).
"""
import os, sys
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton, tilelang
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()
print("ARCH=%s DYN=%s TMA=%s" % (os.environ.get("TL_ARCH_SUFFIX"),
      os.environ.get("TL_FORCE_DYN_SHARED"), os.environ.get("TL_TMA_ROUTE")), flush=True)

pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
               _allow_text_ttir=True, prologue_opt=True,
               contiguous_innermost_sources={"%arg1"}, async_loads=True)
k = tilelang.compile(pf, target="cuda")
src = k.get_kernel_source()
print("tl_tma_load=%d prefetch=%s" % (src.count("tl::tma_load"), "prefetch_tma" in src), flush=True)

torch.manual_seed(0); dev = "cuda"
b, nh, hd, ds = 1, 112, 64, 64
cs, ng = 64, 8
s = 4096; nc = s // cs
dout = torch.randn(b, s, nh, hd, device=dev, dtype=torch.float32)
C = torch.randn(b, s, ng, ds, device=dev, dtype=torch.float32)
dA = torch.randn(b, nh, nc, cs, device=dev, dtype=torch.float32)
sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
sa = [int(x) for x in dA.stride()]
gd0 = triton.cdiv(hd, 64) * triton.cdiv(ds, 64); gd1 = b * nc; gd2 = nh
dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
sp = [int(x) for x in dprev.stride()]
args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
        torch.zeros(1, device=dev, dtype=torch.float32),
        hd, ds, cs, b, s, nc, nh // ng,
        sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
        sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
        gd2, gd1, gd0]
print("LAUNCHING grounded TMA OPT kernel under f...", flush=True)
k(*args)
torch.cuda.synchronize()
print("EXEC_COMPLETED_NO_FAULT", flush=True)
