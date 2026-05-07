# Wave-7/8 Bugfix Dual-LLM Review (grok-4 vs gpt-5-5-pro)

First dual-provider review since the chatgpt cookies refresh (Wave-8 #7,
commit 6dcca098). Same 6-file bundle submitted to both providers:

- `cppmega.mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py`
- `cppmega.mlx/cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py`
- `tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py`
- `tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py`
- `tl_poc_review/src/transform/lower_tma_to_ptr_arith.cc`
- `tl_poc_review/tilelang/language/reduce_op.py`

| Provider | review_id | text_len | Reports dir |
|---|---|---:|---|
| grok-4 | `rev_38ff59759f` | 37,020 | `/private/tmp/tl_apache_tvm_swap/reports/grok/20260507T130255/` |
| gpt-5-5-pro | `rev_b8987e8e52` | 24,514 | (chatgpt path stored in `/tmp/code_review_async/rev_b8987e8e52.json`) |

## Side-by-side findings

| # | Finding | grok-4 | gpt-5-5-pro | Severity |
|---:|---|:---:|:---:|---|
| 1 | `custom_op_wrapper.py:152-160` `_impl`/`_fake` return-type mismatch (multi-output FA 9-tuple) | ✅ HIGH (correctness #1, design §3) | ✅ HIGH (security #3, "input validation") | **HIGH — both flagged** |
| 2 | `dsa_splitk_indexer_loss.py` `topk_indices` OOB → `scatter_` unchecked GPU write (debug-gated, prod foot-gun) | ✅ HIGH (security #1, concrete attacker scenario) | ❌ missed (mentioned only generic "uncontrolled env var") | grok-only |
| 3 | `dsa_splitk_indexer_loss.py` tiled BLOCK_SQ ASq-non-multiple edge case | ✅ HIGH (correctness #2) | ❌ missed | grok-only |
| 4 | `fp8_amax.py` device-padding `zeros + copy` per-call hot-path regression for non-pow2 N (~50% extra HBM) | ✅ HIGH (perf §2) | ❌ generic "memory access optimization" only | grok-only |
| 5 | `dsa_splitk_indexer_loss.py` `sparse_loss=True` `O(AB×ASq×Sk)` scatter alloc per forward | ✅ HIGH (perf §2) | ❌ missed | grok-only |
| 6 | `reduce_op.py` `mul` AllReduce blocked by C++ `vectorize_loop.cc:67` invariant violation | ✅ confirmed dead-code path | ⚠️ vague "memory integrity" — didn't trace to C++ | grok-only |
| 7 | `lower_tma_to_ptr_arith.cc` `TryVisitAllocateMutator` dispatcher pattern | ✅ correct, no regression | ⚠️ flagged as "complex memory operation" but no specific concern | grok-only verified |
| 8 | `fp8_amax.py` `_expose_to_globals` thread-safety (lru_cache + sequential JIT contract) | ✅ noted untested under concurrency | ❌ flagged generically as "closure injection" via env-vars | grok-only |
| 9 | Environment-variable input validation (`CPPMEGA_MLX_TILELANG_ENGINE`, `CPPMEGA_DSA_KL_MODE`, `CPPMEGA_MLX_DSA_DEBUG`) | ❌ not flagged (treats env-vars as trusted local config) | ✅ HIGH security #1, #2 | chatgpt-only |
| 10 | Generic "magic numbers", "redundant type checks", "inefficient intermediate variables" | ❌ not flagged | ✅ MEDIUM (perf §6, §9, §11) | chatgpt-only (low-signal noise) |

## Items both LLMs flagged (high-confidence ship-blockers)

1. **`custom_op_wrapper.py:152-160` return-type mismatch** — both providers
   independently identified the multi-output (FA 9-tuple) contract as
   under-specified. `_impl` returns whatever `artifact.launcher` produces
   (tuple), `_fake` does `if n_outputs == 1: return outs[0]; return tuple(outs)`,
   schema annotation is `-> List[torch.Tensor]`. Three different shapes for the
   same logical contract.
   **Action**: standardise launcher + `_fake` + schema annotation to **always**
   return `list(outs)` (or use `Union[Tensor, List[Tensor]]` + `@overload`).

## Items only grok-4 caught (8/10)

grok-4 produced **substantially deeper, code-grounded findings**. It traced
specific file:line sites, named attacker scenarios with concrete consequences,
and quantified perf hits (e.g. "~50% extra HBM traffic for N=4097 → 8192").

Notable grok-only HIGH-severity catches that should drive wave-9 work:

- **DSA splitK security foot-gun** (security #1): `topk_indices` OOB →
  unchecked `scatter_` GPU write → silent memory corruption. Wave-7/8
  *deliberately* gated the bounds check behind `CPPMEGA_MLX_DSA_DEBUG`
  for perf, turning a correctness foot-gun into a production security
  hazard. Fix is one line: `topk_idx64.clamp_(0, Sk-1)` (zero perf cost).
- **fp8_amax pow2 padding hot path** (perf §2): every non-pow2 activation
  shape pays full-device `zeros + copy_` per forward/backward. In LLM
  training (hundreds of activations per step) this is measurable
  regression vs the original Triton path. Fix: persistent buffer pool or
  exact-N kernel (mask already in inner loop).
- **DSA `sparse_loss=True` scatter alloc per call** (perf §2): O(N) full
  alloc + scatter per forward. Cache the mask when topk_indices stable.
- **Tiled BLOCK_SQ ASq-non-multiple** (correctness #2): wave-8 366b5be
  added `_can_use_q_cache_v5_tiled` but no test exercises non-multiple
  `ASq` with the reduced BLOCK_SQ. Latent off-by-one risk.

## Items only chatgpt caught

- **Environment variable input validation** (security #1, #2): treats
  `CPPMEGA_MLX_TILELANG_ENGINE` and `CPPMEGA_DSA_KL_MODE` as untrusted
  attack surface. In our threat model these are local developer
  configuration (not attacker-controlled) — **low signal**.
- Generic "magic numbers / redundant intermediate variables" — not
  actionable for a fused-kernel project where those constants come from
  hardware register-tile geometry.

ChatGPT's review was significantly less code-grounded: many findings
restated the directive ("the issue identified by grok") without
independent code analysis. This may be a model-side rate-limit on tool
calls; gpt-5-5-pro's review reads like the model didn't fully ingest the
attached files.

## Ship-blocker consensus list

**Wave-9 priorities, ranked by combined-LLM signal**:

1. **(both)** Standardise `_impl`/`_fake`/schema return contract for
   multi-output ops — finding #1.
2. **(grok)** `topk_indices.clamp_(0, Sk-1)` in `dsa_splitk_indexer_loss_tilelang`
   — one-line security-+correctness fix that wave-7/8 left as a debug-gated
   foot-gun.
3. **(grok)** Add ASq-non-multiple unit test for tiled BLOCK_SQ on Metal.
4. **(grok)** Persistent padded-buffer pool in `fp8_amax_tilelang` to kill
   the per-call `zeros + copy_` regression.
5. **(grok)** Cache the `index_mask` scatter in DSA `sparse_loss=True` when
   `topk_indices` are stable across steps.
6. **(grok)** Land the C++ `kMul` enum + `vectorize_loop.cc` fix (wave-8 #5
   commit 9a7d1d3e is in flight) so `reduce_prod` exits dead-code state.

## Provider verdict

For deep code review of the unified-pipeline kernels, **grok-4 was
substantially more useful** than gpt-5-5-pro on this bundle. ChatGPT
reaffirmed the most obvious finding (return-type mismatch) but missed
the security foot-gun, the perf regressions, and the BLOCK_SQ edge case
— all of which grok identified with specific file:line + attacker
scenario.

Recommend: keep dual-provider as the default for high-stakes review
waves (catches finding #1 with two-shot confidence, reveals
provider-specific blind spots like grok's env-var-trust assumption), but
budget time around grok's depth for actionable triage.

The chatgpt cookies refresh has unblocked the dual-review workflow — first
since wave-1 in this session.
