# Upstream Codegen-Reorg Absorption Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realign our fork's source layout to upstream tile-ai/tilelang's multi-backend-decoupled layout by absorbing the refactor PR chain in dependency order, so that future upstream cherry-picks (Blackwell/Hopper/ROCm features) apply cleanly.

**Architecture:** Absorb the refactor as an ordered sequence of single-commit cherry-picks (each upstream PR is a squash-merge = one commit) onto a dedicated branch. Each task applies one PR, resolves the known conflict hotspots against our divergent tree, rebuilds, runs the Metal regression suite + CUDA codegen string tests, then commits. We never run a single `git merge upstream/main` (confirmed to produce a broad conflict ball). We keep OUR `3rdparty/tvm` pin throughout — `tirx` is already present in our TVM and tilelang, so the only TVM-fork coupling is already satisfied.

**Tech Stack:** C++ (TileLang `src/`), TVM (`DatasunriseOU/tvm`, apache+tirx based), CMake+Ninja build in `build/`, Python (editable from `build/`), pytest. Platform: macOS/Metal (CUDA/ROCm paths are build-only + codegen-string verified, not runtime-executed here).

---

## Pre-flight context (read before starting)

**Why this plan exists.** Our fork forked at `936ae921` (2026-04-30). Since then upstream landed 101 commits, including a multi-backend decoupling refactor. We already absorbed two of its stages independently:
- `tirx` migration — DONE (164 files in our `src/` use `tirx`, our TVM carries `tvm/tirx/` + `s_tir/`).
- backend **op**-split — DONE (`src/backend/{cuda,rocm,metal,...}/op/` exists in our tree).

We have NOT absorbed:
- backend **codegen**-split — our CUDA codegen still lives at `src/target/codegen_cuda.cc`; upstream moved it to `src/backend/cuda/codegen/` and added `src/backend/cuda/{runtime.cc,stubs/}`.
- `src/backend/common/` shared lowerers.
- the backend-aware transform pipeline (`#2189`).

**Asymmetry that helps us:** upstream kept `codegen_metal.cc/.h` in `src/target/` (same as us). Only `rt_mod_metal.cc` + Metal ops moved into `src/backend/metal/`. So the reorg's blast radius on OUR primary area (Metal codegen) is small; the real work is CUDA/common/pipeline.

**TVM pin rule (applies to EVERY cherry-pick):** if a cherry-pick touches `3rdparty/tvm`, always keep ours:
```bash
git checkout --ours 3rdparty/tvm && git add 3rdparty/tvm
```
Only `#2216` and `#2273` touch the submodule in this range; `#2273` is Windows-only and out of scope.

**Verification model on macOS:**
- `BUILD`: `cmake --build build -j` (ninja). A green build proves the CUDA/ROCm C++ reorg is internally consistent (host compiles the codegen emitters; it does not run device kernels).
- `METAL_REGRESSION`: `python -m pytest testing/python/metal/ -x -q` — our area must never regress.
- `CODEGEN_SMOKE`: targeted codegen string tests that emit CUDA source without a GPU (see Task 0 for the exact selection found in the repo).

**Branch + build caveat:** Do this on a branch in THIS checkout (not a worktree) because the `build/` tree is bound to this path and a worktree would need a full rebuild of TVM. Each task rebuilds incrementally.

**Ordered PR chain (dependency order, with commit hashes):**
| # | PR | Commit | Role |
|---|----|--------|------|
| 1 | #2088 | `2905fde9` | Refactor register annotation lowering |
| 2 | #2138 | `48998d26` | Split `tl.copy` lowering by backend |
| 3 | #2153 | `960b9c98` | Split GEMM implementations by backend |
| 4 | #2121 | `9e2cf420` | **CodeGen multi-backend decoupling (codegen move — biggest)** |
| 5 | #2156 | `40acae4d` | Split remaining TileOps by backend |
| 6 | #2163 | `38983368` | Share common GPU tile op lowerers (adds `src/backend/common/`) |
| 7 | #2164 | `0fdd0f8a` | Move backend stubs out of codegen |
| 8 | #2165 | `a797e512` | Move backend GEMM impls + transforms into backend dirs |
| 9 | #2161 | `1fbd994c` | Refactor TensorCoreIntrinEmitter (atom-level mma) |
| 10 | #2048 | `f11954cb` | Refactor `gemm_sp` |
| 11 | #2216 | `b939fa01` | tirx + reorg residual (keep our tvm pin; resolve layout/CMake only) |
| 12 | #2189 | `0a9b6512` | Backend-aware Transform Pipeline |

After the chain: Task 13 reconciles our Metal backend + open PR #2252 onto the new layout; Task 14 is the final full-suite gate.

---

## Task 0: Branch, baseline, and verification harness

**Files:**
- Create: branch `merge/upstream-codegen-reorg`
- Modify: none (baseline capture only)

- [ ] **Step 1: Confirm clean tree and create the branch**

Run:
```bash
cd /Volumes/external/sources/tilelang
git status --porcelain        # expect empty
git fetch upstream --no-recurse-submodules
git switch -c merge/upstream-codegen-reorg
```
Expected: empty status; branch created off `main`.

- [ ] **Step 2: Record the baseline build is green**

Run:
```bash
cmake --build build -j 2>&1 | tail -5
```
Expected: build completes, `build/lib/libtilelang.dylib` newer than sources. If it does NOT build clean on a fresh checkout, STOP — fix baseline before merging anything.

- [ ] **Step 3: Capture the baseline Metal regression result**

Run:
```bash
python -m pytest testing/python/metal/ -q 2>&1 | tail -15 | tee /tmp/baseline_metal.txt
```
Expected: record the pass/fail/skip counts. This is the regression bar — every later task must match or beat it. Note any pre-existing failures here so they are not blamed on the merge.

- [ ] **Step 4: Identify the CUDA codegen-string smoke tests (no GPU needed)**

Run:
```bash
grep -rln "codegen\|emit\|cuda_source\|@T.prim_func" testing/python/ | grep -iE "codegen|lower" | head
ls testing/python/transform/ | grep -iE "lower|codegen" | head
```
Expected: a short list of tests that compile a prim_func to source and assert on the emitted string (these run host-only). Record the chosen invocation as `CODEGEN_SMOKE`, e.g.:
```bash
python -m pytest testing/python/transform/ -q -k "lower or codegen" 2>&1 | tail -10
```
If none are host-runnable on macOS, set `CODEGEN_SMOKE` to "build-only" and rely on `BUILD` for CUDA-path verification. Write the final decision into this checkbox before proceeding.

- [ ] **Step 5: Commit the baseline note**

```bash
mkdir -p .planning/merge-notes
cp /tmp/baseline_metal.txt .planning/merge-notes/baseline_metal.txt
git add .planning/merge-notes/baseline_metal.txt
git commit -m "chore(merge): record baseline metal regression before codegen-reorg absorption"
```

---

## Task 1: Absorb #2088 — register annotation lowering

**Files (expected hotspots):**
- Modify: `src/transform/*` (register annotation lowering pass), `src/op/*`
- Our risk: low — pre-codegen-move, touches transform/op which exist in both layouts.

- [ ] **Step 1: Cherry-pick with provenance**

Run:
```bash
git cherry-pick -x 2905fde9
```
Expected: applies clean OR stops with conflicts in `src/transform/`/`src/op/`.

- [ ] **Step 2: Resolve conflicts (if any)**

For each conflicted file, keep our divergent logic but adopt upstream's register-annotation lowering structure. Conflicts here are textual (our extra ops vs their refactor). Resolve, then:
```bash
git add -A
git cherry-pick --continue
```

- [ ] **Step 3: BUILD**

Run: `cmake --build build -j 2>&1 | tail -5`
Expected: green.

- [ ] **Step 4: METAL_REGRESSION**

Run: `python -m pytest testing/python/metal/ -x -q 2>&1 | tail -8`
Expected: matches baseline from Task 0 Step 3.

- [ ] **Step 5: The cherry-pick already committed.** Verify message carries `(cherry picked from commit 2905fde9...)`:
```bash
git log -1 --format="%s%n%b" | grep cherry
```

---

## Task 2: Absorb #2138 — split `tl.copy` lowering by backend

**Files (expected hotspots):**
- Move/create: `src/backend/{cuda,rocm,cpu}/op/copy.cc` (we already have these dirs)
- Modify: `src/op/copy.*`
- Our risk: MEDIUM — our Metal copy lives at `src/backend/metal/op/copy.cc`; ensure upstream's per-backend split does not clobber Metal-specific copy logic.

- [ ] **Step 1: Cherry-pick**

```bash
git cherry-pick -x 48998d26
```

- [ ] **Step 2: Resolve — protect Metal copy**

If conflict touches `src/backend/metal/op/copy.cc`, keep OURS (our Metal copy is ahead of upstream's). For `src/backend/{cuda,rocm}/op/copy.cc` and `src/op/copy.*`, take upstream's split structure but re-apply any fork-specific copy fixes we had in the monolithic file. Then:
```bash
git add -A && git cherry-pick --continue
```

- [ ] **Step 3: BUILD** — `cmake --build build -j 2>&1 | tail -5` → green.
- [ ] **Step 4: METAL_REGRESSION** — `python -m pytest testing/python/metal/ -x -q 2>&1 | tail -8` → matches baseline.
- [ ] **Step 5: Verify cherry-pick provenance line present.**

---

## Task 3: Absorb #2153 — split GEMM implementations by backend

**Files (expected hotspots):**
- Move/create: `src/backend/{cuda,rocm,cpu}/op/gemm.cc`
- Modify: `src/op/gemm.*`
- Our risk: MEDIUM-HIGH — `src/backend/metal/op/gemm.cc` is OURS and central to open PR #2252. Keep ours; only adopt the CUDA/ROCm split + the `src/op/gemm.*` dispatch shape.

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 960b9c98`
- [ ] **Step 2: Resolve — `src/backend/metal/op/gemm.cc` stays OURS.** Adopt upstream dispatch in `src/op/gemm.*`; re-apply our Metal dispatch hook. `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** → matches baseline (Metal GEMM tests `test_metal_gemm_v2*.py` are the canary).
- [ ] **Step 5: Verify provenance.**

---

## Task 4: Absorb #2121 — CodeGen multi-backend decoupling (the codegen move)

**This is the largest and riskiest task.** It moves CUDA codegen from `src/target/codegen_cuda.*` to `src/backend/cuda/codegen/` and reshapes codegen entrypoints.

**Files (expected hotspots):**
- Move: `src/target/codegen_cuda.{cc,h}` → `src/backend/cuda/codegen/codegen_cuda.{cc,h}`
- Move/create: `src/backend/cuda/codegen/{ptx.cc,intrin_rule_cuda.cc,codegen_py.*,rt_mod_cuda.cc,codegen_cutedsl.*}`
- Create: `src/backend/cuda/{runtime.cc,runtime.h,stubs/}`
- Modify: `CMakeLists.txt`, `src/backend/cuda/CMakeLists.txt`, target registration in `src/target/`
- KEEP OURS: `src/target/codegen_metal.{cc,h}` (upstream also keeps Metal codegen in `src/target/` — do NOT let this PR move or delete it)

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 9e2cf420`
- [ ] **Step 2: Resolve the codegen move**

Strategy:
1. Accept upstream's new files under `src/backend/cuda/codegen/` and `src/backend/cuda/{runtime,stubs}`.
2. For our `src/target/codegen_cuda.*`: if upstream deletes it (rename), accept the deletion AFTER porting any fork-specific CUDA codegen edits we made into the new `src/backend/cuda/codegen/codegen_cuda.cc`. Diff our pre-merge version against upstream's to find fork-only hunks:
   ```bash
   git show main:src/target/codegen_cuda.cc > /tmp/ours_codegen_cuda.cc
   diff /tmp/ours_codegen_cuda.cc src/backend/cuda/codegen/codegen_cuda.cc | head -80
   ```
   Port any of OUR unique hunks into the new location.
3. Leave `src/target/codegen_metal.*` untouched (ours).
4. Merge `CMakeLists.txt`: adopt upstream's backend-codegen source globbing; preserve our Metal/MLX/tvm-ffi build wiring (the lines that build `libtilelang_mlx_tvm_ffi_c_api`, Metal sources, etc.).

```bash
git add -A && git cherry-pick --continue
```

- [ ] **Step 3: BUILD** — `cmake --build build -j 2>&1 | tail -15`
  Expected: green. This is the step most likely to reveal dangling includes (`#include "target/codegen_cuda.h"` → must become `backend/cuda/codegen/codegen_cuda.h`). Fix include paths until it builds.
- [ ] **Step 4: METAL_REGRESSION** — `python -m pytest testing/python/metal/ -x -q 2>&1 | tail -10` → matches baseline.
- [ ] **Step 5: CODEGEN_SMOKE** — run the `CODEGEN_SMOKE` invocation recorded in Task 0 Step 4 → matches baseline (or build-only if that was the decision).
- [ ] **Step 6: Amend the cherry-pick with our resolution note**

```bash
git commit --amend --no-edit
git log -1 --format="%s" # confirm still references #2121
```

---

## Task 5: Absorb #2156 — split remaining TileOps by backend

**Files (expected hotspots):**
- Move/create: `src/backend/{cuda,rocm,cpu}/op/{atomic_add,reduce,fill,transpose,...}.cc`
- Modify: `src/op/*.cc` (remaining ops), `src/op/finalize_reducer.cc`, `src/op/atomic_reduce.cc`
- KEEP OURS: corresponding `src/backend/metal/op/*` (reduce, finalize_reducer, copy already ours)

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 40acae4d`
- [ ] **Step 2: Resolve** — adopt CUDA/ROCm/CPU op splits; keep Metal ops ours; re-apply fork-only op fixes. `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** → matches baseline (`test_metal_reduce.py`, `test_metal_local_var.py` canaries).
- [ ] **Step 5: Verify provenance.**

---

## Task 6: Absorb #2163 — share common GPU tile op lowerers

Introduces `src/backend/common/` (we currently lack it). This is the shared layer Metal should eventually sit on.

**Files (expected hotspots):**
- Create: `src/backend/common/op/{reduce,fill,transpose,atomic_reduce,cumsum,finalize_reducer}.h`
- Modify: `src/backend/{cuda,rocm,cpu}/op/*` to consume the shared headers; `src/backend/*/CMakeLists.txt`

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 38983368`
- [ ] **Step 2: Resolve** — accept `src/backend/common/` wholesale. For `src/backend/metal/op/*`, DO NOT auto-rewire to common yet (deferred to Task 13); keep Metal ops self-contained so this task stays green. `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green (Metal still compiles standalone).
- [ ] **Step 4: METAL_REGRESSION** → matches baseline.
- [ ] **Step 5: Verify provenance.**

---

## Task 7: Absorb #2164 — move backend stubs out of codegen

**Files (expected hotspots):**
- Create: `src/backend/{cuda,rocm}/stubs/*`
- Modify: codegen files to reference stubs

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 0fdd0f8a`
- [ ] **Step 2: Resolve** — accept stub extraction for CUDA/ROCm; ensure no Metal stub is expected (Metal has no stub split upstream). `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** → matches baseline.
- [ ] **Step 5: Verify provenance.**

---

## Task 8: Absorb #2165 — move backend GEMM impls + transforms into backend dirs

**Files (expected hotspots):**
- Move: backend-specific GEMM transforms into `src/backend/{cuda,rocm}/`
- Modify: `src/transform/*` (GEMM-related), `src/op/gemm*`
- KEEP OURS: Metal GEMM transforms (`src/transform/metal_*`, `tilelang/transform/metal/*`)

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x a797e512`
- [ ] **Step 2: Resolve** — adopt CUDA/ROCm GEMM transform relocation; keep all `metal_*` transforms ours. `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** → matches baseline (`test_metal_fragment_to_simdgroup_fp8.py`, `test_metal_pass_pipeline.py` canaries).
- [ ] **Step 5: Verify provenance.**

---

## Task 9: Absorb #2161 — refactor TensorCoreIntrinEmitter (atom-level mma)

**Files (expected hotspots):**
- Modify: `tilelang/intrinsics/*`, `tilelang/cuda/intrinsics/*`, `src/backend/cuda/op/gemm.cc`
- Our note: open PR #2252 uses a unified `MPSIntrinEmitter` for Metal that parallels this; keep our Metal emitter, adopt the CUDA atom-level interface.

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 1fbd994c`
- [ ] **Step 2: Resolve** — adopt CUDA emitter refactor; preserve our `tilelang/intrinsics/metal_macro_generator.py` and Metal emitter. `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** → matches baseline.
- [ ] **Step 5: Verify provenance.**

---

## Task 10: Absorb #2048 — refactor gemm_sp

**Files (expected hotspots):**
- Modify: `src/op/gemm_sp*.{cc,h}`, `src/backend/cuda/op/gemm_sp*`
- Our risk: low for Metal (sparse GEMM is CUDA-side).

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x f11954cb`
- [ ] **Step 2: Resolve** — adopt upstream `gemm_sp` shape. `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** → matches baseline.
- [ ] **Step 5: Verify provenance.**

---

## Task 11: Absorb #2216 — tirx + reorg residual (KEEP OUR TVM PIN)

We already use `tirx`; this task is about absorbing the residual reorg + CMake/codegen path changes WITHOUT changing our TVM submodule.

**Files (expected hotspots):**
- Modify: `CMakeLists.txt`, `cmake/load_tvm.cmake`, `src/backend/common/op/*`, many `src/backend/*/codegen/*`
- DO NOT CHANGE: `3rdparty/tvm` (keep ours)

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x b939fa01`
- [ ] **Step 2: Keep our TVM pin immediately**

```bash
git checkout --ours 3rdparty/tvm 2>/dev/null && git add 3rdparty/tvm
```

- [ ] **Step 3: Resolve remaining**

Since our code is ALREADY on `tirx`, most `tir::` → `tirx::` hunks in conflicted files are already applied on our side — take OURS for namespace lines, take UPSTREAM for genuine reorg/structure lines. For `CMakeLists.txt` / `cmake/load_tvm.cmake`, keep our TVM discovery (apache+tvm-ffi paths, MLX wiring) but adopt any new backend source registration.
```bash
git add -A && git cherry-pick --continue
```

- [ ] **Step 4: BUILD** — `cmake --build build -j 2>&1 | tail -15` → green. Watch for `tir::`/`tirx::` mismatches; our TVM provides `tvm/tirx/` so includes must resolve.
- [ ] **Step 5: METAL_REGRESSION** → matches baseline.
- [ ] **Step 6: CODEGEN_SMOKE** → matches baseline.
- [ ] **Step 7: Verify provenance AND that `3rdparty/tvm` still points at our pin:**

```bash
git ls-tree HEAD 3rdparty/tvm   # expect 66438efa... (our pin), NOT 0be33607 (upstream)
```

---

## Task 12: Absorb #2189 — backend-aware Transform Pipeline

The capstone: makes the transform pipeline dispatch per backend.

**Files (expected hotspots):**
- Modify: `tilelang/engine/*` (pipeline assembly), `src/transform/*`, backend pipeline registration
- Our area: our Metal pass pipeline (`tilelang/transform/metal/*`, `test_metal_pass_pipeline.py`) must slot into the new backend-aware pipeline.

- [ ] **Step 1: Cherry-pick** — `git cherry-pick -x 0a9b6512`
- [ ] **Step 2: Resolve** — adopt the backend-dispatch pipeline; register our Metal pipeline as the Metal backend's pipeline within the new structure (rather than a hard-coded branch). `git add -A && git cherry-pick --continue`
- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION** — `python -m pytest testing/python/metal/ -x -q 2>&1 | tail -10` → matches baseline. `test_metal_pass_pipeline.py` and `test_metal_merge_round_barrier.py` are the canaries.
- [ ] **Step 5: Verify provenance.**

---

## Task 13: Reconcile our Metal backend onto the new layout

Now that the layout matches upstream, wire our Metal backend into `src/backend/common/` where it duplicates shared lowerers, and reconcile with open PR #2252 (M5 cooperative tensor).

**Files:**
- Modify: `src/backend/metal/op/*` (consume `src/backend/common/op/*` headers where equivalent)
- Modify: `src/target/codegen_metal.{cc,h}` (align entrypoint signatures with the new codegen dispatch)
- Review: `tilelang/transform/metal/*`, `tilelang/backend/metal/gemm.py`

- [ ] **Step 1: Diff our Metal ops against the new common headers**

Run:
```bash
ls src/backend/common/op/
diff <(sed -n '1,80p' src/backend/metal/op/reduce.cc) <(sed -n '1,80p' src/backend/common/op/reduce.h) | head -40
```
Expected: identify which Metal op logic can delegate to common vs must stay Metal-specific (simdgroup/fp8 paths stay Metal).

- [ ] **Step 2: Rewire only the safe overlaps** — for each Metal op that is a thin specialization, include the common header and override only the Metal-specific hook. Leave fp8/simdgroup logic Metal-local.

- [ ] **Step 3: BUILD** → green.
- [ ] **Step 4: METAL_REGRESSION (full, not -x)** — `python -m pytest testing/python/metal/ -q 2>&1 | tail -15` → matches or beats baseline.
- [ ] **Step 5: Decide on open PR #2252**

Compare our current Metal GEMM against upstream open PR #2252 to confirm it is our own upstreaming and whether the M5 cooperative-tensor path is already in our tree:
```bash
gh pr diff 2252 --repo tile-ai/tilelang | grep -E "^\+\+\+|^diff" | head -40
git grep -l "cooperative_tensor\|matmul2d\|MPSIntrinEmitter" -- src tilelang | head
```
If the cooperative-tensor path is NOT in our tree, file a follow-up task to port it (out of scope for this reorg plan). Record the decision in `.planning/merge-notes/`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(metal): reconcile Metal backend onto upstream common/codegen layout"
```

---

## Task 14: Final full-suite gate

**Files:** none (verification + summary only).

- [ ] **Step 1: Clean rebuild from scratch**

```bash
cmake --build build -j --clean-first 2>&1 | tail -10
```
Expected: green from clean.

- [ ] **Step 2: Full Metal suite**

```bash
python -m pytest testing/python/metal/ -q 2>&1 | tail -20 | tee /tmp/final_metal.txt
diff .planning/merge-notes/baseline_metal.txt /tmp/final_metal.txt || true
```
Expected: no new failures vs baseline.

- [ ] **Step 3: Broader host-runnable suites (transform/analysis)**

```bash
python -m pytest testing/python/transform/ testing/python/analysis/ -q 2>&1 | tail -20
```
Expected: green (these are layout-sensitive and exercise the reorg).

- [ ] **Step 4: Confirm TVM pin unchanged across the whole branch**

```bash
git ls-tree HEAD 3rdparty/tvm    # must be our pin 66438efa..., never 0be33607
git log --oneline main..HEAD | grep -c cherry  # sanity: count absorbed PRs
```

- [ ] **Step 5: Write the merge summary**

Create `.planning/merge-notes/codegen-reorg-summary.md` listing: PRs absorbed (with hashes), files now matching upstream layout, Metal-specific deviations retained, and the #2252 decision from Task 13 Step 5. Commit it.

```bash
git add .planning/merge-notes/codegen-reorg-summary.md
git commit -m "docs(merge): summarize upstream codegen-reorg absorption"
```

---

## Post-plan: what this unlocks (not part of this plan)

Once the layout matches upstream, the hardware/feature PRs you flagged become tractable cherry-picks (separate plan), tiered by conflict surface established earlier:
- **Easy** (`src/tl_templates/` only): #2198 (SM75 MMA), parts of #2126.
- **Medium** (`src/op/`, `src/target/ptx.cc`, ROCm codegen): #2126 (TCGEN5 F8F6F4), #2132 (CDNA4 MXFP4), #2280 (TF32 MMA fix).
- **Now-easy after reorg** (`src/backend/cuda/codegen/*`): #2129 (TMA gather4/scatter4), #2032 (tfloat32), #2271 (fp4 unpacked), #2126 plumbing — these land cleanly only because Task 4 aligned the codegen path.
- **Open PRs to track**: #2252 (our Metal M5), #2155 (Hexagon — backend pattern), #1831/#1983 (scheduling), #2260 (stochastic fp8/fp4 rounding), #2253/#2182/#2171 (SM120/Blackwell FP4 for GB10/B200).
