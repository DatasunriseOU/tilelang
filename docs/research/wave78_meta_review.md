# Wave-9 Third-Provider Review — Meta llama-4

**Headline**: Meta `rev_1469790ac9` (40,017 chars). 7 distinct findings; **3-4 NEW that grok-4 + chatgpt-pro both missed**.

Cross-reference:
- Grok rev_38ff59759f — 8 HIGH severity
- ChatGPT rev_b8987e8e52 — 1 HIGH severity (consensus on multi-output)
- **Meta rev_1469790ac9 — verified 70c3bc3b fix + 6 NEW HIGHs**

## Items Meta confirms (already fixed)

| # | Finding | Status |
|---|---------|--------|
| 1 | `custom_op_wrapper.py:260-331` multi-output return contract | **FIXED in 70c3bc3b** (verified by Meta lines 281-290 / 311-320) |

## Items Meta NEW (not in grok or chatgpt)

| # | File:line | Severity | Bug |
|---|-----------|----------|-----|
| 2 | `custom_op_wrapper.py:38,238` | HIGH | `_REGISTRY_LOCK` + cache check NOT atomic — race condition: two threads compile same op, version skew where `_impl` closure captures stale `artifact`. Real risk in TorchDynamo concurrent inference. |
| 3 | `fx_to_tilelang.py:1715` + `dsa_splitk_indexer_loss.py:340` | HIGH | Integer overflow in block-size math → `gridDim.x` int32 wraparound. Attacker can submit `dim=1<<30` → wrapped grid → atomic_max wrong → silent NaN propagation. **No upper-bound validation on user-controlled shapes.** |
| 4 | `fx_to_tilelang.py:1422` | MEDIUM | Tainted shape data flows to kernel launch w/o validation. Supply-chain HF model with `torch.empty(N * 1e9)` → bypasses 32KB Metal check via dim truncation → DoS. |
| 5 | `reduce_op.py:189-199, 508-510` | HIGH | Warp-reduction comment admits "value accumulated multiple times for inactive threads". `BLOCK_SIZE % 32 != 0` → last partial warp double-counts → `softmax` denominators wrong → policy bypass in LLM safety filters using compiler-fused kernels. |
| 6 | `dsa_splitk_indexer_loss.py:892` | HIGH | Stage-2 lacks the `T.max(1, ...)` ASq=0 clamp that stage-1 has. Silent OOB read of `K[0,0]` when ASq=0. |
| 7 | `dsa_splitk_indexer_loss.py:840-843` | HIGH | `AH * BLOCK_SQ` fragment alloc bypassed by env override. `TILELANG_DSA_BLOCK_SQ=64 + AH=128` → 128KB threadgroup register → spills, 20x slowdown, no error. |
| 8 | `fp8_amax.py:90-92` `_expose_to_globals` | HIGH | Mutates shared `fn.__globals__` dict. Concurrent compile of different `(N, BLOCK)` kernels races → 1% compile failure → eager fallback → 100x slower. Wave-8 #1 fix is thread-unsafe. |
| 9 | `lower_tma_to_ptr_arith.cc:518-560` | HIGH | **Meta-only "4th category"**: `tma_load_im2col` returns `std::nullopt`, logs warning. If a downstream pass deletes failed nodes, graph runs with missing conv data → silent NaN. Affects 100% of CNNs on ROCm/Metal. |
| 10 | `reduce_op.py:236-251` | HIGH | `reduce_prod` emits RuntimeWarning but still emits TIR with `mul` → C++ pass crashes mid-training, losing optimizer state. Should raise NotImplementedError or auto-rewrite to log-sum-exp. (Wave-8 #5 9a7d1d3e added kMul C++ — verifies if Meta's claim still applies.) |

## Items Meta MISSED that grok caught

| Grok HIGH | Meta verdict | Reality |
|-----------|--------------|---------|
| Tiled BLOCK_SQ ASq-non-multiple edge case | Not flagged | grok caught; Wave-9 #2 verdict GUARDED — no source bug |
| fp8_amax non-pow2 pad+copy hot path | **Caught** (item #2 Meta) | Both flagged — Wave-9 #3 fixed in d764f88 |
| sparse_loss scatter alloc per forward | **Caught** (item #3 Meta) | Both flagged |

## Items Meta MISSED that chatgpt caught

ChatGPT only had 1 HIGH (multi-output return contract), which Meta also caught + verified fix. Zero misses.

## Grok HIGH validity audit (per Meta)

| Grok HIGH # | Meta verdict | Note |
|-------------|--------------|------|
| #1 Command injection | False positive (no `subprocess`/`os.system`) | Grok pattern-scan w/o context |
| #2 Unsafe deserialization | False positive | No `pickle`/`yaml.load` |
| #3 Multi-output contract | Valid, fixed in 70c3bc3b | Consensus all 3 LLMs |
| #4 SSRF | False positive | No HTTP code in GPU kernels |
| #5 Path traversal | False positive | No user-path `open()` |
| #6 Secrets in code | False positive | None present |
| #7 Weak crypto | False positive | No crypto |
| #8 XSS | False positive | GPU kernel code |

**Meta's verdict**: 7 of grok's 8 HIGHs are pattern-match false positives; only #3 was real (and it's fixed).

## Triple-LLM ship-blocker consensus

After triple cross-check, the REAL HIGH-severity backlog is:

| # | Issue | Source | Fix status |
|---|-------|--------|------------|
| 1 | Multi-output return contract | All 3 | ✅ Fixed (70c3bc3b) |
| 2 | fp8_amax non-pow2 pad+copy | Grok + Meta | ✅ Fixed (d764f88) |
| 3 | dsa_splitk sparse_loss `.item()` sync | Grok + Meta | Partial (`CPPMEGA_MLX_DSA_DEBUG` gate, but underlying scatter still risky) |
| 4 | reduce_prod hard crash, not warn | Meta | OPEN (wave-8 9a7d1d3e added kMul; must verify the C++ path closes the warning) |
| 5 | `_REGISTRY_LOCK` race (registry skew) | Meta | OPEN |
| 6 | Grid overflow / tainted shapes | Meta | OPEN |
| 7 | Warp reduction with non-mult-32 BLOCK | Meta | OPEN |
| 8 | ASq=0 clamp in stage-2 | Meta | OPEN |
| 9 | env-override bypasses fragment budget | Meta | OPEN |
| 10 | `_expose_to_globals` race | Meta | OPEN — wave-8 #1 fix is single-thread |
| 11 | `tma_load_im2col` silent NaN | Meta | OPEN — wave-1/2 left as TODO |

## Recommendation

Wave-10 should land:
- **P0**: items 5 (registry race), 6 (grid overflow), 7 (warp reduction lane mask)
- **P1**: items 8 (ASq=0 stage-2 clamp), 9 (runtime fragment budget assert), 10 (`_expose_to_globals` thread-local globals dict)
- **P2**: item 11 (`tma_load_im2col` impl OR assert-fail) — Meta's "4th category" supply-chain catch

Triple-LLM coverage validates that grok over-flags pattern matches but produces deeper code-grounded findings; chatgpt is shallow on this bundle; Meta is the strongest contextual reviewer for kernel-codegen specifics.

— /private/tmp/tl_poc_review/docs/research/wave78_meta_review.md, wave-9 third-provider review
