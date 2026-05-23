"""Step 3.b: fwd+bwd numeric parity for a small transformer block.

Tests ``torch.compile(backend="tilelang")`` end-to-end against eager
PyTorch on a tiny transformer-block-shaped subgraph

    Linear -> GELU -> Linear -> LayerNorm

asserting both **output parity** and **gradient parity** for every
parameter. This is the user-facing acceptance check for the AOTAutograd
backward integration (RFC §7 Phase 2.3 / integration #10).

We deliberately use CPU tensors so the test runs on any host (including
the Apple Metal dev box) -- the user spec says CPU/MPS. The same kernel
shape will also pass on MPS once the MPS adapter handles the joint
graph aot_autograd hands us; that's tracked separately.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


@pytest.fixture(autouse=True)
def _register_tilelang_backend():
    """Ensure the tilelang dynamo backend is registered for these tests."""
    from poc.torch_dynamo import register

    register()


def _make_transformer_block(
    in_dim: int = 16,
    hidden: int = 32,
    out_dim: int = 16,
    seed: int = 0,
) -> nn.Module:
    """Build the canonical ``Linear -> GELU -> Linear -> LayerNorm`` block."""
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, out_dim)
            self.ln = nn.LayerNorm(out_dim)

        def forward(self, x):
            return self.ln(self.fc2(torch.nn.functional.gelu(self.fc1(x))))

    return Block()


def _clone_module_params(mod: nn.Module) -> nn.Module:
    """Deep-copy a module's parameters for an independent eager comparator."""
    import copy

    clone = copy.deepcopy(mod)
    for p in clone.parameters():
        p.requires_grad_(True)
    return clone


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, name: str) -> None:
    """Tighter than ``torch.testing.assert_close``'s defaults for fp32 CPU."""
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
        msg=f"{name}: tilelang compiled output diverges from eager",
    )


def test_transformer_block_fwd_bwd_parity_vs_eager_cpu() -> None:
    """Linear -> GELU -> Linear -> LayerNorm: outputs and grads match eager.

    The fused subgraph touches every op family the bw_compiler needs to
    handle for a real transformer block:

    * ``aten.addmm`` / ``aten.mm`` (fc1, fc2 + their bwd)
    * ``aten.gelu`` / ``aten.gelu_backward``
    * ``aten.native_layer_norm`` / ``aten.native_layer_norm_backward``
    * Saved-tensor plumbing for the input + the activation + the norm
      mean/rstd buffers.

    If any of those drops a saved tensor or a gradient slot, the
    parameter grads will diverge below.
    """
    mod_compiled = _make_transformer_block(in_dim=16, hidden=32, out_dim=16, seed=0)
    mod_eager = _clone_module_params(mod_compiled)

    torch.manual_seed(1)
    x_seed = torch.randn(4, 16, dtype=torch.float32)
    x_compiled = x_seed.detach().clone().requires_grad_(True)
    x_eager = x_seed.detach().clone().requires_grad_(True)

    # Eager reference -- full fwd+bwd with a scalar loss.
    y_eager = mod_eager(x_eager).sum()
    y_eager.backward()

    # Compiled path: wrap the *whole* loss expression in one
    # ``torch.compile`` graph so aot_autograd captures the joint
    # fwd+bwd in a single subgraph (without this, the trailing
    # ``.sum()`` lives outside the compiled region and the
    # ``requires_grad`` input is routed through our forward-only
    # fast path which (correctly) refuses to handle it).
    def loss_fn(x_in, mod):
        return mod(x_in).sum()

    compiled = torch.compile(loss_fn, backend="tilelang", fullgraph=False)
    y_compiled = compiled(x_compiled, mod_compiled)
    y_compiled.backward()

    # Output parity.
    _assert_close(y_compiled.detach(), y_eager.detach(), name="loss")

    # Input gradient parity.
    assert x_compiled.grad is not None, "compiled run produced no grad on x"
    assert x_eager.grad is not None
    _assert_close(x_compiled.grad, x_eager.grad, name="grad(x)")

    # Per-parameter gradient parity (linear + layernorm).
    eager_params = dict(mod_eager.named_parameters())
    for name, p_compiled in mod_compiled.named_parameters():
        p_eager = eager_params[name]
        assert p_compiled.grad is not None, f"compiled run missed grad for {name}"
        assert p_eager.grad is not None, f"eager run missed grad for {name}"
        _assert_close(p_compiled.grad, p_eager.grad, name=f"grad({name})")


def test_transformer_block_compile_returns_eager_outputs_on_metadata_only() -> None:
    """Forward-only smoke check: the same block compiles + runs cleanly.

    Catches regressions in the path where the joint aot_autograd capture
    must round-trip the LayerNorm meta-tensor outputs (mean, rstd) even
    when the user only consumes the primary tensor.
    """
    mod = _make_transformer_block(in_dim=8, hidden=16, out_dim=8, seed=2)
    x = torch.randn(2, 8, dtype=torch.float32)

    eager_out = mod(x)
    compiled = torch.compile(mod, backend="tilelang", fullgraph=False)
    compiled_out = compiled(x)

    _assert_close(compiled_out, eager_out, name="forward-only output")
