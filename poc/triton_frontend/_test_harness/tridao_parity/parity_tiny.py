import sys, os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
sys.path.insert(0,"/home/dave/source/tilelang")
import tilelang, tvm
js=open("/tmp/pf__chunk_scan_bwd_dstates.json").read()
pf=tvm.ir.load_json(js); kernel=tilelang.compile(pf,target="cuda")
import torch, triton
io=torch.load("/tmp/dstates_io_tiny.pt"); cfg=io["cfg"]
b,nh,hd,ds=cfg["batch"],cfg["nheads"],cfg["headdim"],cfg["dstate"]
nc,cs,ng=cfg["nchunks"],cfg["chunk_size"],cfg["ngroups"]; s=cfg["seqlen"]
BM,BN,BK=cfg["BLOCK_M"],cfg["BLOCK_N"],cfg["BLOCK_K"]
dev="cuda"
dout=io["dout"].to(dev).contiguous();C=io["C"].to(dev).contiguous();dA=io["dA_cumsum"].to(dev).contiguous()
native=io["dprev_native"].to(dev)
gd0=triton.cdiv(hd,BM)*triton.cdiv(ds,BN); gd1=b*nc; gd2=nh; gd01=1
G=gd0*gd1*gd2*gd01
# size buffers EXACTLY to declared extents (pack real data); dout/C true=grid*2048 here (cs=BK)
def pack(mult, real):
    t=torch.zeros(G*mult,device=dev,dtype=torch.float32); t[:real.numel()]=real.reshape(-1); return t
a_dout=pack(2048,dout); a_C=pack(2048,C); a_dA=pack(32,dA)
dprev=torch.zeros(G*4096,device=dev,dtype=torch.float32)
sd=[int(x) for x in dout.stride()];sc=[int(x) for x in C.stride()];sa=[int(x) for x in dA.stride()]
sp=[nc*nh*hd*ds,nh*hd*ds,hd*ds,ds,1]
print("G=%d declared dout=%d real_dout=%d declared_dA=%d real_dA=%d"%(G,G*2048,dout.numel(),G*32,dA.numel()),flush=True)
args=[a_dout,a_C,dprev,a_dA,torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng,
      sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
      gd1,gd2,gd0,gd01]
kernel(*args); torch.cuda.synchronize()
routed=dprev[:b*nc*nh*hd*ds].view(b,nc,nh,hd,ds)
nz_r=int((routed.abs()>0).sum());nz_n=int((native.abs()>0).sum())
md=float((routed-native).abs().max()); ac=torch.allclose(routed,native,atol=1e-3,rtol=1e-3)
print("=== PARITY DEGENERATE (cs=BK=32, single K-trip) ===")
print("NATIVE nz=%d/%d sum=%.4f"%(nz_n,native.numel(),float(native.sum())))
print("ROUTED nz=%d/%d sum=%.4f"%(nz_r,routed.numel(),float(routed.sum())))
print("MAXDIFF=%.6e ALLCLOSE_1e-3=%s"%(md,ac))
print("PASS" if ac else "FAIL")
