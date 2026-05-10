import torch
import torch.fx
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
from poc.torch_dynamo.fx_to_tilelang import _node_op_key

def test_keys():
    def simple_softmax(x):
        return F.softmax(x, dim=-1)

    def simple_gelu(x):
        return F.gelu(x)

    def my_compiler(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function":
                print("TARGET:", node.target, "KEY:", _node_op_key(node.target))
        return gm.forward

    backend = aot_autograd(fw_compiler=my_compiler)
    compiled = torch.compile(simple_softmax, backend=backend)
    x = torch.randn(10, 10)
    compiled(x)

    compiled = torch.compile(simple_gelu, backend=backend)
    compiled(x)

if __name__ == "__main__":
    test_keys()
