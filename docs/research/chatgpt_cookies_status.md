# ChatGPT cookies status — Wave-8 #7 probe

**Date**: 2026-05-07
**Probe**: `code-review-async submit --provider chatgpt --model gpt-5-5-pro` against
`poc/triton_frontend/op_mapping.py` with single `--aspects security`.

## Verdict: WORKING

`~/.env.gpt` size 8009 bytes, mode 0600, mtime `May 7 12:00`. Cookies were
refreshed by the user during the wave-7/8 cycle (fresh enough that the
`token_invalidated` 401 from wave-1 is no longer returned).

**Probe `rev_a33e1949e4`**:

- `status: done`
- `error: no_error` (was `RuntimeError: ChatGPTWeb runner error: POST /backend-api/files 401 token_invalidated` on wave-1 probes)
- `review_text` returns a substantive security review (header: "Security Review of Code"). Excerpt:

> "The code provided relates to the Triton TTIR -> TileLang TIR op-by-op dispatch table, with a number of functions mapping different Triton TTIR operations to TIR-level abstractions in TileLang."

> "Potential Injection via `printf`": this is the same finding wave-3 grok flagged and wave-3 #01 (commit 3078b95d) fixed via `_sanitize_printf_format`. ChatGPT independently re-discovered it — confirms the cookie path is producing a real model response, not a cached/empty placeholder.

## Implication for prior reviews

The 5 wave-1..wave-7 review cycles ran grok-only because of the 401. Now that
ChatGPT is reachable, **wave-8+ can run dual-provider** (grok-4 + gpt-5-5-pro)
for cross-validation. The wave-7/8 bugfix bundle currently in flight as
`rev_38ff59759f` (grok-4) can be paired with a gpt-5-5-pro submission of the
same file set for second-opinion verification.

## Recipe (for future re-issue)

If `token_invalidated` returns later:

```bash
mg-cookies update-env --chatgpt-profile <user_email> ~/.env.gpt
```

(profile: `davidgornshtein@gmail.com` per memory-stored user info).
