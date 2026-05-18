from __future__ import annotations

import pytest


def test_parse_ttir_custom_form_uses_cxx_generic_fallback() -> None:
    from poc.triton_frontend._mlir_path_setup import bootstrap_jaxlib_alias
    from poc.triton_frontend.mlir_walker import parse_ttir
    from poc.triton_frontend.ptr_analysis import shim_available

    if not shim_available() or not bootstrap_jaxlib_alias():
        pytest.skip("requires C++ shim plus jaxlib mlir bindings")

    text = """\
module {
  tt.func public @tiny(%arg0: !tt.ptr<f32>) {
    tt.return
  }
}
"""
    module = parse_ttir(text)

    assert module is not None
    assert '"tt.func"' in str(module)

