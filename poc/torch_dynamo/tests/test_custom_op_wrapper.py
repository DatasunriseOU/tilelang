from dataclasses import dataclass

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from poc.torch_dynamo.aot_autograd_glue import register_double_backward
from poc.torch_dynamo.custom_op_wrapper import (
    FusedKernelArtifact,
    _ensure_non_aliased_outputs,
    wrap_as_custom_op,
)


def test_output_aliasing_input_is_rejected_without_hidden_clone():
    x = torch.randn(2, 3)

    with pytest.raises(RuntimeError, match=r"output #0 .*aliases input"):
        _ensure_non_aliased_outputs((x,), x)


def test_duplicate_output_alias_is_rejected_without_hidden_clone():
    x = torch.randn(2, 3)
    y = x * 2

    with pytest.raises(RuntimeError, match=r"output #1 .*aliases output #0"):
        _ensure_non_aliased_outputs((), (y, y))


def test_passthrough_saved_input_stays_outside_custom_op_without_clone():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    spec = Spec((2, 3), "float32")

    def my_launcher(x):
        return x * 2.0, x

    artifact = FusedKernelArtifact(
        name="test_passthrough_saved_input_op",
        launcher=my_launcher,
        input_specs=[spec],
        output_specs=[spec, spec],
        output_passthrough_sources=(None, ("input", 0)),
    )

    runner = wrap_as_custom_op(artifact, fx_signature={})
    x = torch.randn(2, 3, dtype=torch.float32)

    y, saved_x = runner(x)

    torch.testing.assert_close(y, x * 2.0)
    assert saved_x is x


def test_passthrough_saved_input_t_reconstructs_transpose_outside_custom_op():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    input_spec = Spec((4, 3), "float32")
    output_spec = Spec((3, 4), "float32")

    def my_launcher(w):
        return w.t()

    artifact = FusedKernelArtifact(
        name="test_passthrough_saved_input_t_op",
        launcher=my_launcher,
        input_specs=[input_spec],
        output_specs=[output_spec],
        output_passthrough_sources=(("input_t", 0),),
    )

    runner = wrap_as_custom_op(artifact, fx_signature={})
    w = torch.randn(4, 3, dtype=torch.float32)

    saved_w_t = runner(w)

    torch.testing.assert_close(saved_w_t, w.t())
    assert saved_w_t.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()


def test_duplicate_saved_output_stays_outside_custom_op_without_clone():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    spec = Spec((2, 3), "float32")

    def my_launcher(x):
        y = x * 2.0
        return y, y

    artifact = FusedKernelArtifact(
        name="test_duplicate_saved_output_op",
        launcher=my_launcher,
        input_specs=[spec],
        output_specs=[spec, spec],
        output_passthrough_sources=(None, ("output", 0)),
    )

    runner = wrap_as_custom_op(artifact, fx_signature={})
    x = torch.randn(2, 3, dtype=torch.float32)

    y, saved_y = runner(x)

    torch.testing.assert_close(y, x * 2.0)
    assert saved_y is y


def test_shape_dtype_inference():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    input_specs = [Spec((2, 3), "float32")]
    output_specs = [Spec((2, 3), "float32")]

    def my_launcher(x):
        return x * 2.0

    artifact = FusedKernelArtifact(
        name="test_shape_inference_op",
        launcher=my_launcher,
        input_specs=input_specs,
        output_specs=output_specs
    )

    runner = wrap_as_custom_op(artifact, fx_signature={})

    x = torch.randn(2, 3, dtype=torch.float32)
    traced = make_fx(runner)(x)
    op_found = False
    for node in traced.graph.nodes:
        if node.op == "call_function" and "tilelang.test_shape_inference_op_fwd" in str(node.target):
            op_found = True
            meta = node.meta.get('tensor_meta') or node.meta.get('val')
            assert meta is not None
            assert meta.shape == torch.Size((2, 3))
            assert meta.dtype == torch.float32

    assert op_found


def test_single_output_tuple_container_is_preserved():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    spec = Spec((2, 3), "float32")

    def my_launcher(x):
        return x * 2.0

    artifact = FusedKernelArtifact(
        name="test_single_output_tuple_container_op",
        launcher=my_launcher,
        input_specs=[spec],
        output_specs=[spec],
    )

    runner = wrap_as_custom_op(
        artifact,
        fx_signature={"output_container": "tuple"},
    )

    x = torch.randn(2, 3, dtype=torch.float32)
    out = runner(x)

    assert isinstance(out, tuple)
    assert len(out) == 1
    torch.testing.assert_close(out[0], x * 2.0)


def test_multi_output_shape_dtype_inference():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    input_specs = [Spec((2, 3), "float32")]
    output_specs = [Spec((2, 3), "float32"), Spec((2, 3), "float16")]

    def my_launcher(x):
        return x * 2.0, x.to(torch.float16)

    artifact = FusedKernelArtifact(
        name="test_multi_shape_inference_op",
        launcher=my_launcher,
        input_specs=input_specs,
        output_specs=output_specs
    )

    runner = wrap_as_custom_op(artifact, fx_signature={})

    x = torch.randn(2, 3, dtype=torch.float32)
    traced = make_fx(runner)(x)

    op_found = False
    for node in traced.graph.nodes:
        if node.op == "call_function" and "tilelang.test_multi_shape_inference_op_fwd" in str(node.target):
            op_found = True
            meta = node.meta.get('tensor_meta') or node.meta.get('val')
            assert meta is not None
            assert len(meta) == 2

    assert op_found

def test_double_backward_zero_grad():
    @dataclass
    class Spec:
        shape: tuple
        dtype: str

    input_specs = [Spec((2, 3), "float32")]
    output_specs = [Spec((2, 3), "float32")]

    def my_fwd_launcher(x):
        return x * 2.0

    def my_bwd_launcher(x, grad_out):
        return grad_out * 2.0

    artifact_fwd = FusedKernelArtifact(
        name="test_dbw_op",
        launcher=my_fwd_launcher,
        input_specs=input_specs,
        output_specs=output_specs
    )

    artifact_bwd = FusedKernelArtifact(
        name="test_dbw_op",
        launcher=my_bwd_launcher,
        input_specs=input_specs + output_specs, # args to bwd
        output_specs=input_specs # bwd returns grad_in
    )

    wrap_as_custom_op(artifact_fwd, fx_signature={}, is_backward=False)
    runner_bwd = wrap_as_custom_op(artifact_bwd, fx_signature={}, is_backward=True)

    # Wire them up
    register_double_backward(
        "tilelang::test_dbw_op_fwd",
        "tilelang::test_dbw_op_bwd",
        has_atomic_accumulator=False
    )

    x = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    grad_y = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)

    # We call runner_bwd because aot_autograd essentially builds the backward graph using the bwd op.
    # We want to make sure we can compute the gradient of bwd_op's output.
    # runner_bwd allows requires_grad=True because is_backward=True.

    # Call the PyTorch op directly to ensure autograd engine is invoked instead of just runner
    # Wait, the custom op is registered. But runner_bwd wraps it.
    # Does runner_bwd invoke autograd correctly? Yes, because it calls the custom op.
    grad_x = runner_bwd(x, grad_y)

    # Now we try to compute grad of grad_x w.r.t. grad_y
    # Since we need to implement double_backward for the bwd_op, let's see if it works.

    grad_grad_y, = torch.autograd.grad(grad_x, grad_y, grad_outputs=torch.ones_like(grad_x))

    assert grad_grad_y is not None
    assert torch.all(grad_grad_y == 0.0)
