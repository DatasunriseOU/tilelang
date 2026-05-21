import pytest
from poc.torch_dynamo.aot_autograd_glue import _import_make_boxed_func, make_aot_backend


def _missing_module(_name: str):
    raise ImportError("blocked by test resolver")


def test_import_make_boxed_func_raises_sensible_error():
    with pytest.raises(ImportError, match="integration #10 requires PyTorch >= 2.10"):
        _import_make_boxed_func(import_module=_missing_module)


def test_make_aot_backend_raises_sensible_error():
    with pytest.raises(ImportError, match="Could not import aot_autograd"):
        make_aot_backend(lambda x: x, lambda x: x, import_module=_missing_module)
