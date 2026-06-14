"""Single-launch driver for ncu: compiles ONE json (argv) and launches the
dstates kernel once at full prod grid. ncu wraps this to read regs/occupancy.
"""
import sys, os, re
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ["TILELANG_DISABLE_CACHE"] = "1"
import torch, triton
import tilelang, tvm

jp = sys.argv[1]
pf = tvm.ir.load_json(open(jp).read())
k = tilelang.compile(pf, target="cuda")
torch.manual_seed(0); dev = "cuda"
b, nh, hd, ds = 1, 112, 64, 64
cs, ng = 64, 8
s = 4096; nc = s // cs
dout = torch.randn(b, s, nh, hd, device=dev, dtype=torch.float32)
C = torch.randn(b, s, ng, ds, device=dev, dtype=torch.float32)
dA = torch.randn(b, nh, nc, cs, device=dev, dtype=torch.float32)
gd0 = 1; gd1 = b * nc; gd2 = nh
dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
sp = [int(x) for x in dprev.stride()]; sa = [int(x) for x in dA.stride()]
args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
        torch.zeros(1, device=dev, dtype=torch.float32),
        hd, ds, cs, b, s, nc, nh // ng,
        sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
        sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
        gd2, gd1, gd0]
def call(a):
    try:
        k(*a); torch.cuda.synchronize()
    except ValueError as e:
        if "expected 37" in str(e):
            a2 = a[:-3] + [gd2, gd2, gd1, gd1, gd0, gd0]
            k(*a2); torch.cuda.synchronize(); return
        raise
call(args)
call(args)
print("LAUNCH_DONE")
