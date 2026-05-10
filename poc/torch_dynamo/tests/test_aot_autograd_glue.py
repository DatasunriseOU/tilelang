import pytest
import sys
from unittest.mock import patch
from poc.torch_dynamo.aot_autograd_glue import _import_make_boxed_func, make_aot_backend

def test_import_make_boxed_func_raises_sensible_error():
    with patch.dict('sys.modules', {'functorch.compile': None, 'torch._functorch.aot_autograd': None}):
        with pytest.raises(ImportError, match="integration #10 requires PyTorch >= 2.10"):
            _import_make_boxed_func()

def test_make_aot_backend_raises_sensible_error():
    with patch.dict('sys.modules', {
        'torch._dynamo.backends.common': None,
        'torch._functorch.aot_autograd': None,
        'functorch.compile': None
    }):
        with pytest.raises(ImportError, match="Could not import aot_autograd"):
            make_aot_backend(lambda x: x, lambda x: x)

if __name__ == "__main__":
    pytest.main([__file__])
