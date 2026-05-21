from tilelang.jit.adapter.utils import extract_python_func_declaration


def test_extract_python_func_declaration_handles_nested_defaults():
    source = """
import cutlass.cute as cute

@cute.kernel
def external_tile(
    a: cute.Tensor,
    b: cute.Tensor,
    scale: int = (1 + 2),
    shape: tuple[int, int] = (16, 16),
):
    pass
"""

    declaration = extract_python_func_declaration(source, "external_tile")

    assert declaration == """def external_tile(
    a: cute.Tensor,
    b: cute.Tensor,
    scale: int = (1 + 2),
    shape: tuple[int, int] = (16, 16),
)"""
