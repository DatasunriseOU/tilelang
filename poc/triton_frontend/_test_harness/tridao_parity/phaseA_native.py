import sys, torch, triton
torch.manual_seed(0)
dev = "cuda"
batch, nheads, headdim, dstate = 1, 8, 64, 64
nchunks, chunk_size, ngroups = 8, 64, 1
seqlen = nchunks * chunk_size
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

dout = torch.randn(batch, seqlen, nheads, headdim, device=dev, dtype=torch.float32)
C    = torch.randn(batch, seqlen, ngroups, dstate, device=dev, dtype=torch.float32)
dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, device=dev, dtype=torch.float32)

from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit = getattr(_autok, "fn", _autok)

dprev = torch.empty(batch, nchunks, nheads, headdim, dstate, device=dev, dtype=torch.float32)
grid = (triton.cdiv(headdim, BLOCK_M) * triton.cdiv(dstate, BLOCK_N), batch * nchunks, nheads)
_jit[grid](
    dout, C, dprev, dA_cumsum, None,
    headdim, dstate, chunk_size,
    batch, seqlen, nchunks, nheads // ngroups,
    dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
    C.stride(0), C.stride(1), C.stride(2), C.stride(3),
    dprev.stride(0), dprev.stride(1), dprev.stride(2), dprev.stride(3), dprev.stride(4),
    dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
    0, 0,
    HAS_SEQ_IDX=False,
    BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
)
torch.cuda.synchronize()

# measure native (per-kernel)
import time
for _ in range(10):
    _jit[grid](dout, C, dprev, dA_cumsum, None, headdim, dstate, chunk_size, batch, seqlen, nchunks, nheads//ngroups,
        dout.stride(0),dout.stride(1),dout.stride(2),dout.stride(3),C.stride(0),C.stride(1),C.stride(2),C.stride(3),
        dprev.stride(0),dprev.stride(1),dprev.stride(2),dprev.stride(3),dprev.stride(4),
        dA_cumsum.stride(0),dA_cumsum.stride(2),dA_cumsum.stride(1),dA_cumsum.stride(3),0,0,
        HAS_SEQ_IDX=False,BLOCK_SIZE_M=BLOCK_M,BLOCK_SIZE_N=BLOCK_N,BLOCK_SIZE_K=BLOCK_K)
torch.cuda.synchronize()
N=100; t0=time.time()
for _ in range(N):
    _jit[grid](dout, C, dprev, dA_cumsum, None, headdim, dstate, chunk_size, batch, seqlen, nchunks, nheads//ngroups,
        dout.stride(0),dout.stride(1),dout.stride(2),dout.stride(3),C.stride(0),C.stride(1),C.stride(2),C.stride(3),
        dprev.stride(0),dprev.stride(1),dprev.stride(2),dprev.stride(3),dprev.stride(4),
        dA_cumsum.stride(0),dA_cumsum.stride(2),dA_cumsum.stride(1),dA_cumsum.stride(3),0,0,
        HAS_SEQ_IDX=False,BLOCK_SIZE_M=BLOCK_M,BLOCK_SIZE_N=BLOCK_N,BLOCK_SIZE_K=BLOCK_K)
torch.cuda.synchronize()
native_ms = (time.time()-t0)/N*1000
print("NATIVE_MS_PER_KERNEL", native_ms)

torch.save({"dout":dout.cpu(),"C":C.cpu(),"dA_cumsum":dA_cumsum.cpu(),
            "dprev_native":dprev.cpu(),
            "strides":{"dout":dout.stride(),"C":C.stride(),"dprev":dprev.stride(),"dA":dA_cumsum.stride()},
            "cfg":dict(batch=batch,nheads=nheads,headdim=headdim,dstate=dstate,nchunks=nchunks,
                       chunk_size=chunk_size,ngroups=ngroups,seqlen=seqlen,
                       BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,BLOCK_K=BLOCK_K),
            "native_ms":native_ms},
           "/tmp/dstates_io.pt")
print("NATIVE dprev sum", float(dprev.sum()), "abs-mean", float(dprev.abs().mean()))
print("SAVED /tmp/dstates_io.pt")
