import sys, os
sys.path.insert(0,"/home/dave/source/tilelang")
import tilelang, tvm
import torch, triton
js=open("/tmp/pf__chunk_scan_bwd_dstates.json").read()
pf=tvm.ir.load_json(js); kernel=tilelang.compile(pf,target="cuda")
io=torch.load("/tmp/dstates_io.pt", weights_only=False); cfg=io["cfg"]
b,nh,hd,ds=cfg["batch"],cfg["nheads"],cfg["headdim"],cfg["dstate"]
nc,cs,ng=cfg["nchunks"],cfg["chunk_size"],cfg["ngroups"]; s=cfg["seqlen"]
BM,BN,BK=cfg["BLOCK_M"],cfg["BLOCK_N"],cfg["BLOCK_K"]
dev="cuda"; ratio=nh//ng
dout=io["dout"].to(dev).contiguous();C=io["C"].to(dev).contiguous();dA=io["dA_cumsum"].to(dev).contiguous()
native=io["dprev_native"].to(dev)

def ref_ktrips(ntrip):
    out=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
    for bb in range(b):
        for c in range(nc):
            for h in range(nh):
                g=h//ratio
                acc=torch.zeros(hd,ds,device=dev)
                for kt in range(ntrip):
                    k0=kt*BK
                    do=dout[bb, c*cs+k0:c*cs+k0+BK, h, :]   # [BK,hd]
                    cc=C[bb, c*cs+k0:c*cs+k0+BK, g, :]      # [BK,ds]
                    sc=torch.exp(dA[bb,h,c,k0:k0+BK])       # [BK]
                    dosc=do*sc[:,None]
                    acc=acc+dosc.t()@cc
                out[bb,c,h]=acc
    return out
ref1=ref_ktrips(1)
ref2=ref_ktrips(2)

gd0=triton.cdiv(hd,BM)*triton.cdiv(ds,BN); gd1=b*nc; gd2=nh; gd01=1
dprev=torch.zeros(b,nc,nh,hd,ds,device=dev,dtype=torch.float32)
sd=[int(x) for x in dout.stride()];sc_=[int(x) for x in C.stride()]
sp=[int(x) for x in dprev.stride()];sa=[int(x) for x in dA.stride()]
args=[dout.view(-1),C.view(-1),dprev.view(-1),dA.view(-1),torch.zeros(1,device=dev,dtype=torch.float32),
      hd,ds,cs,b,s,nc,nh//ng,
      sd[0],sd[1],sd[2],sd[3], sc_[0],sc_[1],sc_[2],sc_[3],
      sp[0],sp[1],sp[2],sp[3],sp[4], sa[0],sa[2],sa[1],sa[3], 0,0,
      gd1,gd2,gd0,gd01]
kernel(*args); torch.cuda.synchronize()
routed=dprev
print("ROUTED vs ref(1trip) maxdiff=%.4e"%float((routed-ref1).abs().max()))
print("ROUTED vs ref(2trip) maxdiff=%.4e"%float((routed-ref2).abs().max()))
# look at tile [0,0,0] block 0: routed vs ref2 for first 8x8
print("routed tile[0,0,0,:4,:4]=\n",routed[0,0,0,:4,:4].cpu().numpy())
print("ref2   tile[0,0,0,:4,:4]=\n",ref2[0,0,0,:4,:4].cpu().numpy())
# Is routed only first row correct? count per-row error
err=(routed[0,0,0]-ref2[0,0,0]).abs()
print("per-row(hd) max err first 8 rows:",[round(float(err[i].max()),3) for i in range(8)])
print("per-col(ds) max err first 8 cols:",[round(float(err[:,j].max()),3) for j in range(8)])
# how many elements of tile [0,0,0] match within 1e-2?
m=(err<1e-2).sum()
print("tile[0,0,0] matched(<1e-2)=%d/4096"%int(m))
# maybe routed == ref2 but scaled? ratio
nz=ref2[0,0,0].abs()>1.0
r=routed[0,0,0][nz]/ref2[0,0,0][nz]
print("routed/ref2 ratio mean=%.4f std=%.4f"%(float(r.mean()),float(r.std())))
