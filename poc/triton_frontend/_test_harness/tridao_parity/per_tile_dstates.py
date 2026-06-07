import sys
sys.path.insert(0,"/home/dave/source/tilelang")
import tilelang, tvm, torch, triton
from poc.triton_frontend import from_ttir
ttir=open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()

# OPT with dout reorder hint (default name-gated) vs OPT with hint explicitly disabled.
def build(prologue_opt, dout_hint):
    kw=dict(prologue_opt=prologue_opt, _allow_text_ttir=True, target="cuda")
    if dout_hint is False:
        kw["contiguous_tile_axis"]={}   # explicit empty -> no reorder
    pf=from_ttir(ttir,name="_chunk_scan_bwd_dstates_kernel",**kw)
    return tilelang.compile(pf,target="cuda")

k_opt_dout = build(True, None)    # default: dout reorder active
k_opt_nodout = build(True, False) # OPT but dout reorder OFF

torch.manual_seed(0); dev="cuda"
b,nh,hd,ds=1,112,64,64; cs,ng=64,8; s=4096; nc=s//cs
dout=torch.randn(b,s,nh,hd,device=dev,dtype=torch.float32)
C=torch.randn(b,s,ng,ds,device=dev,dtype=torch.float32)
dA=torch.randn(b,nh,nc,cs,device=dev,dtype=torch.float32)
gd0=1; gd1=b*nc; gd2=nh
sd=[int(x) for x in dout.stride()];sc=[int(x) for x in C.stride()];sa=[int(x) for x in dA.stride()]
dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
sp=[int(x) for x in dprev.stride()]
args=[dout.reshape(-1),C.reshape(-1),dprev.reshape(-1),dA.reshape(-1),torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng, sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0, gd2,gd1,gd0]
def timed(kernel,n):
    for _ in range(10): kernel(*args)
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
        st.record();kernel(*args);en.record();torch.cuda.synchronize();ts.append(st.elapsed_time(en))
    ts.sort();return ts[len(ts)//2]
N=50; a=[];d=[]
for r in range(3):
    a.append(timed(k_opt_dout,N)); d.append(timed(k_opt_nodout,N))
    print("rep%d OPT+dout-reorder=%.4f OPT-no-dout-reorder=%.4f"%(r,a[-1],d[-1]),flush=True)
md=sorted(a)[1]; nd=sorted(d)[1]
print("=== PER-TILE (dout reorder isolation) ===")
print("OPT_WITH_DOUT_REORDER_MS=%.4f"%md)
print("OPT_NO_DOUT_REORDER_MS=%.4f"%nd)
print("DOUT_REORDER_CONTRIB_MS=%.4f"%(nd-md))
