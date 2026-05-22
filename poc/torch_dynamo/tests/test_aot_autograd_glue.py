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


def test_torch_compile_linear_bias_relu_backward_saved_params_passthrough():
    torch = pytest.importorskip("torch")

    from poc.torch_dynamo import register

    register()

    def fn(x, w, b):
        return torch.relu(torch.nn.functional.linear(x, w, b)).sum()

    torch.manual_seed(0)
    x_ref = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    w_ref = torch.randn(4, 3, dtype=torch.float32, requires_grad=True)
    b_ref = torch.randn(4, dtype=torch.float32, requires_grad=True)
    expected = fn(x_ref, w_ref, b_ref)
    expected.backward()
    expected_grads = (
        x_ref.grad.detach().clone(),
        w_ref.grad.detach().clone(),
        b_ref.grad.detach().clone(),
    )

    x = x_ref.detach().clone().requires_grad_(True)
    w = w_ref.detach().clone().requires_grad_(True)
    b = b_ref.detach().clone().requires_grad_(True)

    compiled = torch.compile(fn, backend="tilelang", fullgraph=True)
    actual = compiled(x, w, b)
    actual.backward()

    torch.testing.assert_close(actual.detach(), expected.detach())
    for actual_grad, expected_grad in zip(
        (x.grad, w.grad, b.grad),
        expected_grads,
    ):
        assert actual_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad)
