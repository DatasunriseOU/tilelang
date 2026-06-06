import sys, os
sys.path.insert(0, "/home/dave/source/tilelang")
import tilelang, tvm
from poc.triton_frontend import from_ttir
os.makedirs("/tmp/route_cuda7", exist_ok=True)
names = ["_chunk_scan_bwd_dstates","_chunk_scan_bwd_dc","_chunk_scan_bwd_dcb",
         "_chunk_scan_bwd_dx","_chunk_state_bwd_db","_chunk_state_bwd_dx",
         "_chunk_state_bwd_ddAcs_stable"]
ok=0
tgt=tvm.target.Target("cuda")
for name in names:
    ttir = open("/tmp/ttir7/%s.ttir" % name).read()
    try:
        pf = from_ttir(ttir, name=name + "_kernel", target="cuda", _allow_text_ttir=True)
        from tilelang.engine import lower as L
        with tgt:
            rt = L(pf, target=tgt)
        src = rt.kernel_source
        open("/tmp/route_cuda7/%s.cu" % name, "w").write(src)
        has_global = "__global__" in src
        mma = src.count("mma_sync") + src.count("mma.sync")
        atom = src.count("atomicAdd") + src.count("AtomicAdd")
        print("ROUTE %-34s len=%6d global=%s mma=%d atomicAdd=%d" % (name, len(src), has_global, mma, atom))
        if has_global and mma>0: ok+=1
    except Exception as e:
        print("ROUTEFAIL %-34s %s: %s" % (name, type(e).__name__, str(e)[:160]))
print("TOTAL_OK_WITH_MMA=%d/7" % ok)
