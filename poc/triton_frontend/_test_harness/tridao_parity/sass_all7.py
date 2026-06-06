import sys, os
sys.path.insert(0,"/home/dave/source/tilelang")
import tilelang, tvm
from poc.triton_frontend import from_ttir
os.makedirs("/tmp/sass7",exist_ok=True)
names=["_chunk_scan_bwd_dstates","_chunk_scan_bwd_dc","_chunk_scan_bwd_dcb",
       "_chunk_scan_bwd_dx","_chunk_state_bwd_db","_chunk_state_bwd_dx",
       "_chunk_state_bwd_ddAcs_stable"]
tgt=tvm.target.Target("cuda")
for n in names:
    ttir=open("/tmp/ttir7/%s.ttir"%n).read()
    pf=from_ttir(ttir,name=n+"_kernel",target="cuda",_allow_text_ttir=True)
    k=tilelang.compile(pf,target="cuda")
    try:
        k.export_sass("/tmp/sass7/%s.sass"%n)
        print("SASS_OK",n)
    except Exception as e:
        print("SASS_FAIL",n,str(e)[:80])
