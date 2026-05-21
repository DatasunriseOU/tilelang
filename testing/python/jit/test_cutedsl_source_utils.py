import pytest

from tilelang.jit.adapter.utils import extract_python_func_declaration, match_declare_kernel_cutedsl


def test_match_declare_kernel_cutedsl_accepts_imported_cute_default_alias():
    source = """
import cutlass.cute as cute

@cute.kernel
def external_tile(a: cute.Tensor):
    pass
"""

    index = match_declare_kernel_cutedsl(source)

    assert source[index:].startswith("(a: cute.Tensor)")


def test_match_declare_kernel_cutedsl_accepts_imported_fully_qualified_cute():
    source = """
import cutlass.cute

@cutlass.cute.kernel
def external_tile(a: cutlass.cute.Tensor):
    pass
"""

    index = match_declare_kernel_cutedsl(source)

    assert source[index:].startswith("(a: cutlass.cute.Tensor)")


def test_match_declare_kernel_cutedsl_accepts_imported_cute_alias():
    source = """
import cutlass.cute as ct

@ct.kernel(preprocessor=False)
def external_tile(a: ct.Tensor):
    pass
"""

    index = match_declare_kernel_cutedsl(source)

    assert source[index:].startswith("(a: ct.Tensor)")


def test_match_declare_kernel_cutedsl_accepts_imported_kernel_alias():
    source = """
from cutlass.cute import Tensor, kernel as cute_kernel

@cute_kernel
def external_tile(a: Tensor):
    pass
"""

    index = match_declare_kernel_cutedsl(source)

    assert source[index:].startswith("(a: Tensor)")


def test_match_declare_kernel_cutedsl_rejects_unimported_bare_kernel():
    source = """
@kernel
def external_tile(a):
    pass
"""

    with pytest.raises(ValueError, match="No global kernel found"):
        match_declare_kernel_cutedsl(source)


def test_match_declare_kernel_cutedsl_rejects_unimported_fully_qualified_kernel():
    source = """
@cutlass.cute.kernel
def external_tile(a):
    pass
"""

    with pytest.raises(ValueError, match="No global kernel found"):
        match_declare_kernel_cutedsl(source)


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
