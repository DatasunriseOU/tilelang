import sys
sys.path.insert(0,"/home/dave/source/tilelang")
import tilelang, tvm, torch, triton
js=open("/tmp/pf__chunk_scan_bwd_dstates.json").read()
pf=tvm.ir.load_json(js); kernel=tilelang.compile(pf,target="cuda")
torch.manual_seed(0); dev="cuda"
# SMALL real-strided multi-K-trip config: cs=64 -> 2 K-trips (BK=32), small nh
b,nh,hd,ds=1,8,64,64
cs,ng=64,8
s=512; nc=s//cs  # 8 chunks, multi-K-trip (cs/BK = 2)
BM,BN,BK=64,64,32
dout=torch.randn(b,s,nh,hd,device=dev,dtype=torch.float32)
C=torch.randn(b,s,ng,ds,device=dev,dtype=torch.float32)
dA=torch.randn(b,nh,nc,cs,device=dev,dtype=torch.float32)
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit=getattr(_autok,"fn",_autok)
native=torch.empty(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
grid=(triton.cdiv(hd,BM)*triton.cdiv(ds,BN),b*nc,nh)
_jit[grid](dout,C,native,dA,None,hd,ds,cs,b,s,nc,nh//ng,
    dout.stride(0),dout.stride(1),dout.stride(2),dout.stride(3),
    C.stride(0),C.stride(1),C.stride(2),C.stride(3),
    native.stride(0),native.stride(1),native.stride(2),native.stride(3),native.stride(4),
    dA.stride(0),dA.stride(2),dA.stride(1),dA.stride(3),0,0,
    HAS_SEQ_IDX=False,BLOCK_SIZE_M=BM,BLOCK_SIZE_N=BN,BLOCK_SIZE_K=BK)
torch.cuda.synchronize()
gd0=triton.cdiv(hd,BM)*triton.cdiv(ds,BN); gd1=b*nc; gd2=nh
dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
sd=[int(x) for x in dout.stride()];sc=[int(x) for x in C.stride()]
sp=[int(x) for x in dprev.stride()];sa=[int(x) for x in dA.stride()]
args=[dout.reshape(-1),C.reshape(-1),dprev.reshape(-1),dA.reshape(-1),torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng,
      sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
      gd2,gd1,gd0]
kernel(*args); torch.cuda.synchronize()
routed=dprev
md=float((routed-native).abs().max()); ac=torch.allclose(routed,native,atol=1e-3,rtol=1e-3)
print("SMALL real-strided multi-K-trip (b%d nh%d s%d nc%d cs%d ktrips=%d) vs NATIVE:"%(b,nh,s,nc,cs,cs//BK))
print("MAXDIFF=%.6e ALLCLOSE_1e-3=%s %s"%(md,ac,"PASS" if ac else "FAIL"))
