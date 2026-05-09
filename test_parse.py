import sys
sys.path.append(".")
from poc.triton_frontend.op_mapping import _parse_generic_properties_shared

op_str = '"tt.func"() <{function_type = (f32, f32) -> f32, sym_name = "triton_language_standard_max__fp32S128S_c0_cFalse_cTrue_cFalse", visibility = 1 : i64}> ({'
class DummyOp:
    def __str__(self): return op_str

print(_parse_generic_properties_shared(DummyOp()))
