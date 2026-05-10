import torch
from torch.fx.experimental.proxy_tensor import make_fx
from poc.torch_dynamo.custom_op_wrapper import wrap_as_custom_op, FusedKernelArtifact
from poc.torch_dynamo.aot_autograd_glue import register_double_backward
from dataclasses import dataclass

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
    
    runner_fwd = wrap_as_custom_op(artifact_fwd, fx_signature={}, is_backward=False)
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

