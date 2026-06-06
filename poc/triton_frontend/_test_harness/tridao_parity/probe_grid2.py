import sys, os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton
import tilelang, tvm
js=open("/tmp/pf__chunk_scan_bwd_dstates.json").read()
pf=tvm.ir.load_json(js); kernel=tilelang.compile(pf,target="cuda")
dev="cuda"
b,nh,hd,ds,nc,cs=1,2,64,64,2,64; s=nc*cs; ng=1
sp=[nc*nh*hd*ds, nh*hd*ds, hd*ds, ds, 1]
sd=[s*nh*hd, nh*hd, hd, 1]; sc=[s*ng*ds, ng*ds, ds, 1]; sa=[nh*nc*cs, cs, nc*cs, 1]
gd0=1; gd1=b*nc; gd2=nh; gd01=1
G=gd0*gd1*gd2*gd01
# declared extents
def buf(mult, real):
    t=torch.zeros(G*mult,device=dev,dtype=torch.float32)
    t[:real.numel()]=real.reshape(-1)
    return t
dout=torch.randn(b,s,nh,hd,device=dev,dtype=torch.float32)
C=torch.randn(b,s,ng,ds,device=dev,dtype=torch.float32)
dA=torch.randn(b,nh,nc,cs,device=dev,dtype=torch.float32)
dprev=torch.zeros(G*4096,device=dev,dtype=torch.float32)
a_dout=buf(2048,dout); a_C=buf(2048,C); a_dA=buf(32,dA)
args=[a_dout,a_C,dprev,a_dA,torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng,
      sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
      gd1,gd2,gd0,gd01]
print("G=%d declared: dout=%d C=%d dA=%d dprev=%d"%(G,G*2048,G*2048,G*32,G*4096),flush=True)
try:
    kernel(*args); torch.cuda.synchronize(); print("OK nz_dprev=%d/%d"%(int((dprev.abs()>0).sum()),G*4096))
except Exception as e:
    print("ERR",type(e).__name__,str(e)[:200])
