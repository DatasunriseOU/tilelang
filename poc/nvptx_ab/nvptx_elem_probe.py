import tilelang, tilelang.language as T, tvm, traceback
# Plain elementwise: no tensor cores, no ldmatrix, no barriers -> simplest possible NVPTX candidate
@T.prim_func
def add(A: T.Tensor((1024,1024),"float32"), B: T.Tensor((1024,1024),"float32"), C: T.Tensor((1024,1024),"float32")):
    with T.Kernel(T.ceildiv(1024,128), threads=128) as bx:
        for i in T.serial(128):
            for j in T.serial(1024):
                with T.block():
                    pass
# simpler: use parallel copy add
@T.prim_func
def add2(A: T.Tensor((1048576,),"float32"), B: T.Tensor((1048576,),"float32"), C: T.Tensor((1048576,),"float32")):
    with T.Kernel(8192, threads=128) as bx:
        tx = T.get_thread_binding()
        idx = bx*128 + tx
        C[idx] = A[idx] + B[idx]

from tilelang.engine.lower import lower_to_host_device_ir
tgt=tvm.target.Target({"kind":"cuda","arch":"sm_121"})
with tgt:
    _,device_mod,_,target,_=lower_to_host_device_ir(add2,target=tgt)
import re
src=str(device_mod)
ops=sorted(set(re.findall(r'\btl\.[a-zA-Z0-9_]+|\bptx_[a-zA-Z0-9_]+', src)))
print("ops in elementwise device TIR:", ops)
print("\n=== NVPTX on plain elementwise ===")
try:
    nt=tvm.target.Target({"kind":"nvptx","mcpu":"sm_121"}, host=tvm.target.Target({"kind":"llvm"}))
    f=tvm.ffi.get_global_func("target.build.nvptx")
    m=f(device_mod, nt)
    ptx=m.inspect_source()
    print("NVPTX OK! PTX len", len(ptx), "| PROVENANCE markers:", ".visible .entry" in ptx, "| .target sm" in ptx if False else "")
    print(ptx[:300])
except Exception as e:
    print("NVPTX FAILED:", traceback.format_exc()[-800:])
