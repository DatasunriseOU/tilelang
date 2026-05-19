import tilelang
import tilelang.language as T
from tilelang.tileop.metal_quant import float_to_fp8_e4m3fn_bits


@T.prim_func
def _fp8_e4m3fn_encode_probe(
    x: T.Tensor((4,), "float32"),
    y: T.Tensor((4,), "uint8"),
):
    with T.Kernel(1, threads=1):
        y[0] = float_to_fp8_e4m3fn_bits(x[0])


def test_float_to_fp8_e4m3fn_bits_lowers_to_uint8_storage():
    artifact = tilelang.lower(_fp8_e4m3fn_encode_probe, target="metal")
    source = artifact.kernel_source or ""

    assert "device uchar* y" in source
    assert "log2" in source
    assert "exp2" in source
    assert "as_type<uchar>(((uchar)" not in source
