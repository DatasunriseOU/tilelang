import sys, torch, triton, time
sys.path.insert(0,"/home/dave/state-spaces-mamba")
torch.manual_seed(0); dev="cuda"
# REAL production config: S=4096 c=64 g=8 H=112 P=64 N=64 bs1
batch,nheads,headdim,dstate=1,112,64,64
chunk_size,ngroups=64,8
seqlen=4096; nchunks=seqlen//chunk_size  # 64
BLOCK_M,BLOCK_N,BLOCK_K=64,64,32
dout=torch.randn(batch,seqlen,nheads,headdim,device=dev,dtype=torch.float32)
C=torch.randn(batch,seqlen,ngroups,dstate,device=dev,dtype=torch.float32)
dA=torch.randn(batch,nheads,nchunks,chunk_size,device=dev,dtype=torch.float32)
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit=getattr(_autok,"fn",_autok)
dprev=torch.empty(batch,nchunks,nheads,headdim,dstate,device=dev,dtype=torch.float32)
grid=(triton.cdiv(headdim,BLOCK_M)*triton.cdiv(dstate,BLOCK_N),batch*nchunks,nheads)
def call():
    _jit[grid](dout,C,dprev,dA,None,headdim,dstate,chunk_size,batch,seqlen,nchunks,nheads//ngroups,
        dout.stride(0),dout.stride(1),dout.stride(2),dout.stride(3),
        C.stride(0),C.stride(1),C.stride(2),C.stride(3),
        dprev.stride(0),dprev.stride(1),dprev.stride(2),dprev.stride(3),dprev.stride(4),
        dA.stride(0),dA.stride(2),dA.stride(1),dA.stride(3),0,0,
        HAS_SEQ_IDX=False,BLOCK_SIZE_M=BLOCK_M,BLOCK_SIZE_N=BLOCK_N,BLOCK_SIZE_K=BLOCK_K)
for _ in range(20): call()
torch.cuda.synchronize()
N=200; t0=time.time()
for _ in range(N): call()
torch.cuda.synchronize()
ms=(time.time()-t0)/N*1000
print("PROD grid=%s dprev_numel=%d"%(str(grid),dprev.numel()))
print("NATIVE_PROD_MS_PER_KERNEL=%.5f"%ms)
print("dprev sum=%.3f nz=%d/%d"%(float(dprev.sum()),int((dprev.abs()>0).sum()),dprev.numel()))
