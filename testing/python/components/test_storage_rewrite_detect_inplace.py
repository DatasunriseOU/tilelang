import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang import language as T
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget, get_device_call


@tilelang.jit
def _compile_kernel_without_inplace():
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float]):
        with T.Kernel(num_tokens, threads=32) as pid:
            read = T.alloc_var(T.int)
            read = x[pid]

            write = T.alloc_var(T.int)
            write = read * 2
            x[pid] = write

    return buggy_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_STORAGE_REWRITE_DETECT_INPLACE: True,
    },
)
def _compile_kernel_with_inplace():
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float]):
        with T.Kernel(num_tokens, threads=32) as pid:
            read = T.alloc_var(T.int)
            read = x[pid]

            write = T.alloc_var(T.int)
            write = read * 2
            x[pid] = write

    return buggy_kernel


def _get_device_kernel_script(detect_inplace: bool) -> str:
    jit_func = _compile_kernel_with_inplace if detect_inplace else _compile_kernel_without_inplace
    prim_func = jit_func.get_tir()
    target = tvm.target.Target("cuda", host="c")
    mod = tvm.IRModule({prim_func.attrs["global_symbol"]: prim_func})
    with tvm.transform.PassContext(opt_level=3, config=jit_func.pass_configs or {}):
        with target:
            mod = LowerAndLegalize(mod, target)
            mod = OptimizeForTarget(mod, target)
        device_mod = tvm.tir.transform.Filter(get_device_call())(mod)
    return device_mod.script()


def _has_scaled_assignment(script: str, lhs_prefix: str, rhs_prefix: str) -> bool:
    return any(
        line.strip().startswith(lhs_prefix) and f"= {rhs_prefix}" in line and "* 2" in line
        for line in script.splitlines()
    )


def test_storage_rewrite_detect_inplace_toggle():
    script_off = _get_device_kernel_script(detect_inplace=False)
    script_on = _get_device_kernel_script(detect_inplace=True)

    assert not _has_scaled_assignment(script_off, "read_", "read_"), f"inplace pattern found when disabled:\n{script_off}"
    assert _has_scaled_assignment(script_on, "read_", "read_"), f"inplace pattern not found when enabled:\n{script_on}"
    assert _has_scaled_assignment(script_off, "write_", "read_"), f"separate-write pattern not found when disabled:\n{script_off}"


if __name__ == "__main__":
    tilelang.testing.main()
