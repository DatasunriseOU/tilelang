import sys, time
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton
import tilelang, tvm
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()

def build(prologue_opt):
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt)
    return tilelang.compile(pf, target="cuda")

print("building OFF (prologue_opt=False)...", flush=True)
k_off = build(False)
print("building OPT (prologue_opt=True)...", flush=True)
k_opt = build(True)
print("OFF nparams=%d OPT nparams=%d" % (len(k_off.params), len(k_opt.params)), flush=True)

torch.manual_seed(0); dev="cuda"
b,nh,hd,ds=1,112,64,64
cs,ng=64,8
s=4096; nc=s//cs
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

gd0=triton.cdiv(hd,BM)*triton.cdiv(ds,BN); gd1=b*nc; gd2=nh
sd=[int(x) for x in dout.stride()];sc=[int(x) for x in C.stride()]
sa=[int(x) for x in dA.stride()]
def make_args(dprev):
    sp=[int(x) for x in dprev.stride()]
    return [dout.reshape(-1),C.reshape(-1),dprev.reshape(-1),dA.reshape(-1),
            torch.zeros(1,device=dev,dtype=torch.float32),
            hd,ds,cs,b,s,nc,nh//ng,
            sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
            sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
            gd2,gd1,gd0]

# parity for both
def run(kernel):
    dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
    kernel(*make_args(dprev)); torch.cuda.synchronize()
    return dprev

for label,kernel in [("OFF",k_off),("OPT",k_opt)]:
    r=run(kernel)
    md=float((r-native).abs().max()); ac=torch.allclose(r,native,atol=1e-3,rtol=1e-3)
    print("PARITY %s MAXDIFF=%.6e ALLCLOSE_1e-3=%s %s"%(label,md,ac,"PASS" if ac else "FAIL"),flush=True)

# interleaved OFF/OPT/OFF/OPT CUDA-event timing
dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
args=make_args(dprev)
def timed(kernel, n):
    # warmup
    for _ in range(10): kernel(*args)
    torch.cuda.synchronize()
    times=[]
    for _ in range(n):
        st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
        st.record(); kernel(*args); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    times.sort()
    return times[len(times)//2], sum(times)/len(times), times[0]

N=60
off_med=[]; opt_med=[]
for rep in range(4):
    m1,a1,mn1=timed(k_off, N); off_med.append(m1)
    m2,a2,mn2=timed(k_opt, N); opt_med.append(m2)
    print("rep%d OFF med=%.4f OPT med=%.4f"%(rep,m1,m2),flush=True)
off=sorted(off_med)[len(off_med)//2]; opt=sorted(opt_med)[len(opt_med)//2]
print("=== INTERLEAVED CUDA-EVENT (N=%d/rep x4 reps, median-of-medians) ==="%N)
print("OFF_MS=%.4f"%off)
print("OPT_MS=%.4f"%opt)
print("DELTA_MS=%.4f (OFF-OPT)"%(off-opt))
print("SPEEDUP=%.4fx"%(off/opt))
