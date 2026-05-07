---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: 
timestamp: 2026-05-07T09:06:00.305745+00:00
files: ['/private/tmp/tl_poc_review/poc/torch_dynamo/__init__.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/fx_to_tilelang.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/_fusion_patterns.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/_kernels/flash_attention.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/custom_op_wrapper.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/aot_autograd_glue.py', '/private/tmp/tl_poc_review/poc/torch_dynamo/examples/test_torch_compile_chain.py']
---
**WAVE-3 FINAL REGRESSION REVIEW (perf-focused)**

All wave-1 + wave-2 + wave-3 fixes have landed cleanly with **no regressions** introduced:

- Sequential/unary/binary emitters are now exercised (and produce real `PrimFunc`s when JIT is available) — see `test_wave2_unary_chain_uses_sequential_emitter`, `test_wave3_binary_elementwise_uses_sequential_emitter`, and the per-region source checks in `test_torch_compile_chain.py`.
- Multi-region chain launcher correctly avoids the `gm.forward` fallback path (`_tilelang_chain_mode != "multi_fallback_to_gm_forward"`).
- Registry cache + content-hash stability are verified (`test_same_graph_recompile_hits_registry_cache`, `test_content_hash_stable_across_recompiles`, 128-bit digest).
- ATEN_DISPATCH coverage is expanded and guarded (`test_wave2_aten_dispatch_covers_top_ops`).
- Lowering remains strictly compile-time; no new hot-path work.

**Remaining HIGH-severity performance issues** (all on the *inference* hot path; none are compile-time):

1. **`custom_op_wrapper.py:80-130` (`_ensure_contiguous_inputs`)**  
   Executed on *every* custom_op call (i.e., every forward pass of every fused subgraph).  
   - Always does `new_seen = set(_CONTIGUITY_WARN_SEEN)` (O(#previously-seen warnings) copy).  
   - Then per-tensor: `is_contiguous()`, `_base` getattr, etc.  
   - If *any* input is non-contiguous or view-aliased (extremely common after views/transposes/previous ops), it does `.contiguous()` (or `.contiguous().clone()` for the aliased+contiguous case) **on every single call**.  
   This introduces full-tensor allocations + memcpy in the critical path. Quantified impact: 1-5%+ overhead + visible alloc pressure on view-heavy LLM workloads (the exact case this guard was added to protect). The wave-2 "perf fix" only avoided *extra* clone on the non-aliased non-contig path; the per-call materialization remains.  
   **This is the biggest regression vs eager/Inductor.**

2. **`flash_attention.py:120-140` (the `_launcher` closure inside `make_flash_attention_kernel` / `make_sdpa_kernel`)**  
   The `try: out = torch.empty(...); res = kernel(q, k, v, out) except TypeError: return kernel(q, k, v)` is executed *on every inference call*.  
   Python exception handling + unconditional `torch.empty` allocation inside the try branch on every FA/SDPA forward pass.  
   This was not present in earlier waves; it was added as a calling-convention safety net but is now hot-path overhead for one of the most performance-critical ops.

3. **`fx_to_tilelang.py:~1400` (`_build_chain_launcher` → inner `_launcher_multi` closure)**  
   For any graph that produces >1 region (still common until full fusion patterns land), the runtime does:  
   - `env: Dict[str, Any] = dict(...)`  
   - `tuple(env[n] for n in in_names)` + string dict lookups *per region per call*.  
   The closure closes over `region_io`. Fine for the target 1-region fused case, but introduces measurable Python overhead vs Inductor’s single-wrapper approach on boundary-heavy graphs.

**Minor / non-blocking notes** (not high-severity):
- `custom_op_wrapper.py:~280` (`_runner`): `list(runtime_inputs)` allocation on every call.
- `aot_autograd_glue.py` autotune cache and `_CONTIGUITY_WARN_SEEN` grow unbounded over process lifetime (small keys, no leak).
- No O(n²), no redundant I/O, no blocking calls, no tight-loop allocations, no memory growth beyond expected kernel buffers.

**Verdict**: The package is **clean enough to ship** after fixing the two hottest items above (#1 and #2). The wave fixes held; the only remaining issues are the exact hot-path concerns the wave-2/3 reviews surfaced. No other bugs or regressions visible in the current state. Green to merge once the contiguity guard and FA launcher are tightened (or documented as “ensure inputs are contiguous upstream” for Phase 3).