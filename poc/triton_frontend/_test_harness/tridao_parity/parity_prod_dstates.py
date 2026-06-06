import sys, time
sys.path.insert(0,"/home/dave/source/tilelang")
import torch, triton
import tilelang, tvm
js=open("/tmp/pf__chunk_scan_bwd_dstates.json").read()
pf=tvm.ir.load_json(js); kernel=tilelang.compile(pf,target="cuda")
print("PROD compiled dstates nparams=%d"%len(kernel.params),flush=True)
torch.manual_seed(0); dev="cuda"
# REAL production §P1 config
b,nh,hd,ds=1,112,64,64
cs,ng=64,8
s=4096; nc=s//cs  # 64
BM,BN,BK=64,64,32
dout=torch.randn(b,s,nh,hd,device=dev,dtype=torch.float32)
C=torch.randn(b,s,ng,ds,device=dev,dtype=torch.float32)
dA=torch.randn(b,nh,nc,cs,device=dev,dtype=torch.float32)
# native reference
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
# routed
gd0=triton.cdiv(hd,BM)*triton.cdiv(ds,BN); gd1=b*nc; gd2=nh; gd01=1
dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
sd=[int(x) for x in dout.stride()];sc=[int(x) for x in C.stride()]
sp=[int(x) for x in dprev.stride()];sa=[int(x) for x in dA.stride()]
args=[dout.reshape(-1),C.reshape(-1),dprev.reshape(-1),dA.reshape(-1),torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng,
      sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
      gd1,gd2,gd0,gd01]
kernel(*args); torch.cuda.synchronize()
routed=dprev
nz_r=int((routed.abs()>0).sum()); nz_n=int((native.abs()>0).sum())
md=float((routed-native).abs().max()); ac=torch.allclose(routed,native,atol=1e-3,rtol=1e-3)
print("=== PARITY _chunk_scan_bwd_dstates PRODUCTION §P1 ===")
print("CONFIG: b%d nh%d hd%d ds%d nc%d cs%d numel=%d grid=%s"%(b,nh,hd,ds,nc,cs,native.numel(),str(grid)))
print("NATIVE nonzero=%d/%d sum=%.4f"%(nz_n,native.numel(),float(native.sum())))
print("ROUTED nonzero=%d/%d sum=%.4f"%(nz_r,routed.numel(),float(routed.sum())))
print("MAXDIFF=%.6e"%md)
print("ALLCLOSE_1e-3=%s"%ac)
print("PASS" if ac else "FAIL")
# timing routed
for _ in range(10): kernel(*args)
torch.cuda.synchronize()
N=50;t0=time.time()
for _ in range(N): kernel(*args)
torch.cuda.synchronize()
print("ROUTED_PROD_MS_PER_KERNEL=%.5f"%((time.time()-t0)/N*1000))
