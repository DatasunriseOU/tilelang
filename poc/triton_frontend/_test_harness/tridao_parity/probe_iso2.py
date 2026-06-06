import sys, os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
sys.path.insert(0, "/home/dave/source/tilelang")
import torch
import tilelang, tvm
js=open("/tmp/pf__chunk_scan_bwd_dstates.json").read()
pf=tvm.ir.load_json(js); kernel=tilelang.compile(pf,target="cuda")
dev="cuda"
def z(n): return torch.zeros(n,device=dev,dtype=torch.float32)
# EXACT replica of probe_launch (known-good) but vary buffer sizes
def trial(tag, sizes):
    a=[z(sizes[0]),z(sizes[1]),z(sizes[2]),z(sizes[3]),z(sizes[4]),
       64,64,64,1,64,1,8]+[1]*19+[1,1,1,1]
    try:
        kernel(*a); torch.cuda.synchronize(); print(tag,"OK")
    except Exception as e:
        print(tag,"ERR",type(e).__name__,str(e)[:120])
trial("small (probe_launch sizes)", [2048,2048,4096,32,1])
trial("big dout only", [200000,2048,4096,32,1])
trial("all big", [200000,200000,200000,200000,200000])
