"""Flash-attention forward kernel factory used by the FX -> TileLang lowerer.

RFC reference: ``RFC_unified_fused_kernel.md`` §7 Phase 2.2 (FX op map row
``aten._scaled_dot_product_flash_attention`` -> TileLang). The
implementation is a thin JIT factory around
``examples/flash_attention/example_mha_fwd_bhsd.py`` (FA-v2 forward,
BHSD layout) that exposes a single entry point
:func:`make_flash_attention_kernel` returning a compiled callable plus
the symbolic ``PrimFunc`` so the FX walker can content-hash it.

PyTorch's ``aten._scaled_dot_product_flash_attention`` returns a
``(out, lse, philox_seed, philox_offset, debug_attn_mask)`` 5-tuple.
We materialise only ``out`` here; the remaining 4 slots are filled by
the eager fallback wrapper in ``custom_op_wrapper.py`` with sentinel
zero tensors of the contractually-correct shapes.

# Verified: philox_seed / philox_offset (and newer rng_state tensors)
# are properly ignored and zero-filled by the _emit_getitem placeholder
# interceptor in fx_to_tilelang.py, preventing backwards-pass crashes.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


def _import_tilelang() -> Tuple[Any, Any]:
    """Lazy import of tilelang + tilelang.language."""
    import tilelang  # type: ignore
    import tilelang.language as T  # type: ignore
    return tilelang, T


def make_flash_attention_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    dtype: str = "float16",
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    target: Optional[str] = None,
) -> Tuple[Any, Callable[..., Any]]:
    """Build and JIT-compile an FA-v2 forward kernel.

    Returns ``(prim_func, launcher)``. ``launcher`` takes ``(Q, K, V)``
    PyTorch tensors of shape ``(B, H, S, D)`` and returns ``Out`` of the
    same Q-shape. Cribs from
    ``examples/flash_attention/example_mha_fwd_bhsd.py``.

    Top design choice: we use ``log2(e)`` pre-multiplied into ``scale``
    so the inner loop can use the cheaper ``T.exp2`` instead of
    ``T.exp`` — this matches the example.
    """
    tilelang, T = _import_tilelang()

    # Pre-multiply log2(e) into scale so we can use exp2 in the hot loop.
    scale_log2e = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    accum_dtype = "float32"
    past_len = seq_kv - seq_q

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_q, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N),
                    T.ceildiv((bx + 1) * block_M + past_len, block_N),
                )
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            q_idx >= k_idx, 0, -T.infinity(acc_s.dtype)
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_kv,
                            -T.infinity(acc_s.dtype),
                            0,
                        )
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale_log2e - scores_max[i] * scale_log2e
                    )
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale_log2e - scores_max[i] * scale_log2e
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(
                    acc_s_cast,
                    V_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] = acc_o[i, j] / logsum[i]
            T.copy(
                acc_o,
                Output[bz, by, bx * block_M:(bx + 1) * block_M, :],
            )

    # JIT-compile. If the active target lacks a JIT backend the caller is
    # expected to fall back to FX-eager replay (see fx_to_tilelang.py).
    kernel = tilelang.compile(main, target=target) if target else tilelang.compile(main)

    def _launcher(q: Any, k: Any, v: Any) -> Any:
        # Correctness fix (grok review #2): the prim_func declares four
        # buffers ``(Q, K, V, Output)`` so the compiled kernel uses the
        # explicit-output calling convention. Allocate ``Output`` with
        # the contractual Q-shape and pass it as the fourth argument.
        # If the underlying tilelang.compile output binding ever changes
        # to the implicit-output convention (output auto-returned), the
        # ``TypeError`` branch transparently falls back to ``kernel(q,k,v)``.
        import torch  # type: ignore[import-not-found]
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
        try:
            out = torch.empty(
                tuple(q_shape), dtype=torch_dtype, device=q.device,
            )
            res = kernel(q, k, v, out)
            return res if res is not None else out
        except TypeError:
            return kernel(q, k, v)

    return main, _launcher


def make_sdpa_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    dtype: str = "float16",
    is_causal: bool = False,
    target: Optional[str] = None,
) -> Tuple[Any, Callable[..., Any]]:
    """Build a non-flash SDPA kernel as fused QK / softmax / V via FA-v2.

    The PyTorch ``aten.scaled_dot_product_attention`` op has the same
    forward math as flash-attention; only the *return contract* differs
    (default returns out only, ``return_debug_mask=True`` adds lse).
    We reuse :func:`make_flash_attention_kernel` and let the wrapper
    drop the lse if not requested.
    """
    return make_flash_attention_kernel(
        batch=batch,
        heads=heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        dim=dim,
        dtype=dtype,
        is_causal=is_causal,
        target=target,
    )
