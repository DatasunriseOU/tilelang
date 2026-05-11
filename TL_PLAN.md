# TL_PLAN.md - TileLang Metal/tvm-ffi completion plan

This file is the working contract for making cppmega Path C a real
TileLang path instead of a cppmega-side Metal text rewrite path.

## Non-negotiable invariants

- Production cppmega Path C must call TileLang kernels through
  `tilelang -> tvm -> tvm-ffi`; it must not depend on cppmega-side
  post-MSL regex/text transforms.
- Adapters must not hide large tensor allocations, CPU staging, GPU copies, or
  dtype casts. Inputs are existing GPU buffers; outputs are either explicit
  owner-provided buffers or TileLang-owned outputs whose ownership is clear.
- MLX integration uses standard tensor ownership mechanisms first:
  DLPack/tvm-ffi handles for data pointers and explicit Metal runtime handles
  only when synchronization cannot be represented by DLPack alone.
- The MLX route must not go through Torch, Torch-MPS stream hooks, or CUDA-only
  assumptions.
- Z3-backed reasoning lives in the TileLang/TVM pass pipeline. It proves
  barrier elision, vectorization legality, launch geometry, alias safety, and
  reduction rewrites before codegen. If proof is absent, keep the conservative
  path and mark why.
- Path C AUTO promotion is allowed only when parity passes and the measured
  receipt is not slower than the matching Path B receipt for the target shape.

## Current debt inventory

- cppmega currently has `_msl_transform.py`, `_mlx_runtime.py`, and
  Mamba3-specific lowered-body transforms. These are acceptable only as
  migration scaffolding and must not become the long-term production boundary.
- Mamba3 Path C reaches parity and near-Path-B performance only after
  cppmega-side extraction, launch geometry, and local CSE fixes.
- M2RNN still needs the same TileLang-first treatment as Mamba3, including
  forward/backward parity and no hidden staging.
- Sparse-MLA Path C has BF16, FP8, and blockscaled surfaces, but the full
  end-to-end training graph still has dtype/scale producer ownership gaps.
- FP8 matmul/vecmat Path C exists as standalone surfaces; end-to-end producer
  ownership and backward routing need to be first-class rather than wrapper
  quantization.
- Full `local_gb10` receipt coverage exists for selected routes only. The
  optimizer/dtype matrix still needs broader repeatable coverage.

## Workstreams

| ID | Stream | Primary repo | Write ownership | Goal | Acceptance |
|---|---|---|---|---|---|
| TL-1 | Metal tvm-ffi MLX ABI | TileLang | `tilelang/jit`, `tilelang/contrib`, `3rdparty/tvm/3rdparty/tvm-ffi`, runtime glue | Accept MLX arrays through DLPack/tvm-ffi on Metal without Torch adapters. Preserve device/owner semantics and fail with typed errors, not segfaults. | New MLX DLPack import/export smoke passes; no hidden copy/cast in adapter path; bad device produces a Python-side error. |
| TL-2 | Owner-provided output buffers | TileLang | runtime ABI, packed API lowering, tests | Support explicit output buffers/aliasing for Metal kernels so cppmega can pass existing GPU tensors through the graph. | Test proves a kernel writes into caller-owned MLX/Metal buffer and does not allocate a replacement output. |
| TL-3 | Metal pass pipeline | TileLang | `tilelang/transform`, `src/transform`, pass config | Move launch geometry, scalar CSE/LICM, elem-offset legalization, and MSL cleanup into IR passes before codegen. | Generated MSL for Mamba3 no longer needs cppmega postprocessing; transform tests cover the old fragment-buffer `*_elem_offset` blocker. |
| TL-4 | Z3 scheduler/barrier optimizer | TileLang | Z3 transform helpers, scheduler metadata, tests | Centralize Z3 proof hooks for vectorization, barrier minimization, async eligibility, and alias/shape proofs. | Tests show proof-enabled barrier removal and conservative fallback when proof fails. |
| TL-5 | Metal reductions | TileLang | `src/transform/lower_thread_allreduce.cc`, Metal codegen, reduction tests | Lower CUDA-style `tl::AllReduce<...>::run` to Metal simdgroup/cross-simdgroup reductions, including one-simdgroup-per-row no-barrier templates where legal. | Row reduce/QK reduction parity tests pass; emitted MSL has no unnecessary threadgroup barriers on the proven one-simdgroup route. |
| TL-6 | Mamba3 and M2RNN TileLang-first lowering | cppmega + TileLang | cppmega Path C modules/tests plus TileLang passes needed by TL-3/TL-5 | Remove production reliance on Mamba3 manual transforms and add complete M2RNN Path C forward/backward routing. | Path B vs Path C parity for Mamba3 and M2RNN; cppmega dispatch logs show `path_c`; no `mx.fast.metal_kernel` production fallback for these surfaces. |
| TL-7 | Sparse-MLA FP8/blockscaled composition | cppmega + TileLang | Sparse-MLA Path C modules/tests, FP8/blockscaled TileLang lowering | Make q/kv FP8 + scale first-class producer tensors and consume them directly in Sparse-MLA forward/backward. Avoid wrapper quantization allocations. | FP8/blockscaled parity tests pass; receipt shows Path C uses existing FP8 buffers and stays finite for 20-step real parquet. |
| TL-8 | Receipt matrix and regression gates | cppmega | scripts/tests/docs/bench receipts | Make failures visible: route dispatch, dtype, optimizer, memory peak, tok/sec, and fallback reason. | Repeatable 20-step `local_gb10` receipts for bf16/fp8/int8 routes and adamw/muon/lion/lion8bit/adam8bit where supported. |

## Implementation order

1. Lock current behavior with tests before deleting shims.
2. Build the Metal tvm-ffi ABI and owner-output support first; all higher
   layers depend on correct pointer ownership and synchronization.
3. Move the existing Mamba3 cppmega-side text fixes into TileLang IR passes.
4. Add Z3 proof plumbing around the new passes, then enable optimized rewrites
   only behind proof/config gates.
5. Replace reduction-heavy lowering with Metal simdgroup/online-softmax
   templates where the proof says the route is legal.
6. Port Mamba3 and M2RNN to the TileLang-first route and remove production
   fallback to cppmega-side MSL transforms.
7. Promote FP8/blockscaled producers above Sparse-MLA so Path C consumes the
   right dtype/shape directly.
8. Run the cppmega optimizer/dtype receipt matrix and promote Path C AUTO only
   where it is parity-clean and not slower than Path B.

## Execution status - 2026-05-11 TL-W

- TL-1/TL-2 advanced: MLX Metal buffers now have typed DLPack/tvm-ffi coverage,
  bad devices and consumed capsules fail in Python, and owner-output tests cover
  caller-provided output buffers. This proves the bridge substrate, not a full
  cppmega production switchover.
- TL-3/TL-5 advanced in TileLang: Metal scalar intrinsic binding, pass-pipeline
  scalar cleanup, loop-vectorization caps, merge-round lowering, AllReduce
  lowering, and targeted Metal transform/JIT tests are in place. The remaining
  production cppmega call boundary still uses MLX fast-kernel wrappers for most
  lowered MSL.
- TL-6 advanced in cppmega: Mamba3 and M2RNN dispatch use TileLang-derived
  lowering metadata in the Path C routes, but Path B remains the production
  route for the main scans. Mamba3 Path C is a proof/override path, not a global
  replacement.
- TL-7 advanced in cppmega: Sparse-MLA BF16 Path C is row-gated and promotes
  only receipt-covered green shapes. FP8/blockscaled routes keep prepared
  `q_fp8/q_scale/kv_fp8/kv_scale` style buffers explicit, but full FP8/e8m0
  forward/backward training composition is not closed.
- TL-8 advanced in cppmega: the training receipt now emits route, dtype,
  optimizer, fallback, memory, finite/loss, and tok/sec fields. The current
  green 20-step receipts are partial, not the full requested matrix.
- Current green cppmega 20-step receipts from `/tmp/cppmega_e2e_matrix_20260511`:
  bf16 `adamw` 314.14 tok/s, `muon_adamw` 40.26 tok/s, `nam56r` 40.51 tok/s,
  `lion` 370.82 tok/s, `adam8bit` 629.65 tok/s, `lion8bit` 686.21 tok/s, and
  `int8` 40.58 tok/s; all completed 20/20 steps, stayed finite, and decreased
  loss. After the VJP fix, fp8_path_c 20-step receipts also completed for
  `adamw` 255.02 tok/s, `muon_adamw` 36.73 tok/s, `nam56r` 36.72 tok/s,
  `lion` 293.15 tok/s, `adam8bit` 452.72 tok/s, `lion8bit` 482.59 tok/s, and
  `int8` 36.71 tok/s.
- Do not overclaim the matrix: a matched bf16 vs fp8_path_c lion8bit repro
  still has bf16 green at 916.56 tok/s while the fp8_path_c matched run was
  stopped after 1216.9s (`returncode=-15`), and no full final 20-step or
  100-step real-parquet optimizer/dtype matrix has been run end to end.
- Current FP8 micro-kernel blocker: the production owner-output/tvm-ffi path is
  still too slow for promotion. Treat FP8 matmul as roughly 14x slower than the
  shipped MLX/audiohacking-style route, and vecmat as roughly 1.7x slower, until
  fresh strict receipts prove otherwise. Older packed-dot4 probe receipts are
  useful diagnostics but are not the production gate.
- Remaining `mx.fast.metal_kernel` wrappers to remove or explicitly bless:
  `_mlx_runtime.py`, `_msl_transform.py`, `_mamba3_helpers_tilelang.py`,
  `mamba3_path_c.py`, `m2rnn.py`, `fp8_msl_kernels.py`,
  `fp8_matmul_path_c.py`, `fp8_vecmat_path_c.py`, `topk_selector.py`,
  `sparse_mla.py`, `sparse_mla_path_c.py`, `sparse_mla_fp8.py`,
  `sparse_mla_fp8_path_c.py`, `sparse_mla_blockscaled.py`, and
  `sparse_mla_blockscaled_path_c.py`.

## Temporary scaffolding to delete

- cppmega Mamba3 local lowered-body optimization is a bridge only. The target
  replacement is TL-3 plus TL-5.
- cppmega `_mlx_runtime.py` must shrink to a thin call boundary once TL-1 and
  TL-2 are done.
- Any Path C fallback that silently re-enters Path B or direct `mx.fast`
  kernels must become an explicit disabled route or a test failure.

## Required verification

TileLang:

- `cmake -S . -B build`
- `cmake --build build -j$(nproc)`
- `PYTHONPATH=$(pwd):$PYTHONPATH python -m pytest testing/python/metal/ -x`
- Targeted transform/JIT tests added by each stream.

cppmega:

- `python -m pytest tests/test_tilelang_mamba3_path_c.py tests/test_mamba3_dispatch.py -q`
- `python -m pytest tests/test_tilelang_m2rnn_path_c.py tests/test_m2rnn_dispatch.py -q`
- `python -m pytest tests/test_tilelang_sparse_mla_fp8.py tests/test_tilelang_sparse_mla_blockscaled_path_c.py -q`
- `python -m pytest tests/test_m04_train_step.py -q`
- `ruff check` on touched Python files.
- Real parquet receipts:
  `scripts/m04_train_step.py --receipt local_gb10_quarter --steps 20 --batch-size 1 --seq-len 4096`
  across supported dtype/optimizer routes. Run 100-step only after all 20-step
  routes are finite and dispatch-clean.

## Reporting contract

Each agent must report:

- Files changed.
- Which plan rows it advanced.
- Tests run and exact pass/fail status.
- Any remaining blocker with the failing command and shortest reproducible
  case.
- Any place it avoided a copy/allocation by moving dtype/shape ownership
  higher or by consuming the existing buffer lower.
