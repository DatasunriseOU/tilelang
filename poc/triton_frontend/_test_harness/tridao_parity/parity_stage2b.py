import sys
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton           # torch FIRST
import tilelang, tvm           # then tilelang (no from_ttir/MLIR anywhere)
name = sys.argv[1]
js = open("/tmp/pf_%s.json" % name).read()
pf = tvm.ir.load_json(js)
kernel = tilelang.compile(pf, target="cuda")
print("STAGE2b compiled %s nparams=%d" % (name, len(kernel.params)), flush=True)
io = torch.load("/tmp/dstates_io.pt"); cfg = io["cfg"]
b,nh,hd,ds=cfg["batch"],cfg["nheads"],cfg["headdim"],cfg["dstate"]
nc,cs,ng=cfg["nchunks"],cfg["chunk_size"],cfg["ngroups"]
s=cfg["seqlen"]; BM,BN,BK=cfg["BLOCK_M"],cfg["BLOCK_N"],cfg["BLOCK_K"]
dev="cuda"
dout=io["dout"].to(dev).contiguous(); C=io["C"].to(dev).contiguous(); dA=io["dA_cumsum"].to(dev).contiguous()
native=io["dprev_native"].to(dev)
gridDim_0=triton.cdiv(hd,BM)*triton.cdiv(ds,BN); gridDim_1=b*nc; gridDim_2=nh; gridDim_0_1=1
dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
sd=[int(x) for x in dout.stride()]; sc=[int(x) for x in C.stride()]
sp=[int(x) for x in dprev.stride()]; sa=[int(x) for x in dA.stride()]
args=[dout.view(-1),C.view(-1),dprev.view(-1),dA.view(-1),torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng,
      sd[0],sd[1],sd[2],sd[3], sc[0],sc[1],sc[2],sc[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
      gridDim_1,gridDim_2,gridDim_0,gridDim_0_1]
kernel(*args); torch.cuda.synchronize()
routed=dprev
nz_r=int((routed.abs()>0).sum()); nz_n=int((native.abs()>0).sum())
maxdiff=float((routed-native).abs().max())
ac=torch.allclose(routed,native,atol=1e-3,rtol=1e-3)
print("=== PARITY %s SMALL ===" % name)
print("CONFIG: b%d nh%d hd%d ds%d nc%d cs%d numel=%d" % (b,nh,hd,ds,nc,cs,native.numel()))
print("NATIVE nonzero=%d/%d sum=%.4f" % (nz_n,native.numel(),float(native.sum())))
print("ROUTED nonzero=%d/%d sum=%.4f" % (nz_r,routed.numel(),float(routed.sum())))
print("MAXDIFF=%.6e" % maxdiff)
print("ALLCLOSE_1e-3=%s" % ac)
print("PASS" if ac else "FAIL")
