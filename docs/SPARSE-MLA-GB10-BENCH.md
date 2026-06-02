# Fused Sparse-MLA (DeepSeek-V3.2 / DSA) — GB10 sm_121a kernel benchmark

**Kernel-level** speed + memory benchmark of the re-tiled fused sparse-MLA
forward and backward kernels (the GB10 path fixed and proven in commit
`823c807c`, fwd cos 0.998–1.000 / bwd dq,dkv cos 1.000) **vs the O(n²) dense
MLA reference** they replace, at the `local_gb10_quarter` model shapes.

> **Scope — read this first.** Every number below is a **single attention
> operator** measured in isolation on the GPU. This is **NOT** a full-model
> training step. Megatron's **3399 tok/s @ 26 GB** is the entire 1.8B model
> (all layers + every other op + optimizer + activations, fwd+bwd). The two are
> not directly comparable; see *Kernel-level vs full-model* below for the honest
> framing and what is still required to produce a full-model tok/s/memory number
> with this fused attention wired in.

## Environment (measured, no fabrication)

- Host: gb10 (`gx10-9cd4`), `/home/dave/cppmega-venv/bin/python` (py3.13)
- Device: **NVIDIA GB10**, compute capability **(12, 1)** = sm_121a (Blackwell SoC, unified memory)
- CUDA **13.2** runtime / ptxas; tilelang branch `merge/upstream-codegen-reorg` @ `823c807c`, submodule tvm `62cc0314`
- Repo: `/home/dave/source/tilelang`, kernels `examples/deepseek_v32/sparse_mla_fwd.py` + `sparse_mla_bwd.py` (gb10 path)
- Timing: `tilelang.profiler.do_bench`, **kernel built once** (static shapes baked), 50 warmup + 50 timed reps — kernel-launch latency only, no recompilation in the loop.
- Memory: `torch.cuda.max_memory_allocated()` peak (allocator high-water mark over the call), reset per case.
- Bench harness: `examples/deepseek_v32/bench_sparse_mla_gb10.py` (tracked in-repo).

## Model shapes (`local_gb10_quarter` — the model Megatron trains at 3399 tok/s)

From `cppmega_mlx/cppmega_mlx/recipes/model_factory.py`: `num_attention_heads=28`,
`head_dim=128`, `hidden=3584`, `max_seq=4096`, DSA on A-layers (ranks 1,2,3).
This kernel is **MLA**: Q/K carry a latent `dim=512` + `tail=64` (= **576**),
value/out `dim=512`. The model's **28 heads** map directly to the kernel's
`heads`. (head_dim=128 is the *attention* head dim; MLA's KV latent is 512+64,
which is what the GEMMs run over.)

- **B=1, S=4096, S_kv=4096, H=28, DQK=576, DV=512, bf16**, `threads=128`
  (H=28 → padded_H=32; at the production `threads=256` the GEMM trips
  `warp_col_tiles must be > 8`, so the H<=32 tile runs at 4 warps).
- `topk` swept over **256 / 512 / 1024 / 2048** — the model config carries a
  placeholder `attention_sparse_topk=16`; the realistic DSA sparsity is the
  256–2048 range the deepseek_v32 examples parametrize (example default 2048).
  We report the full curve.
- Indices: causal, **in-bounds** (no OOB sentinel), so memory/latency aren't
  distorted by the degenerate early-row artifact the parity harness uses.

Parity re-confirmed at the model head count: **H=28, S=1024, topk=512 →
cos=0.999998, rel-err=0.0022**, `out_nan=0` (fully-filled rows, fused vs dense).

## Results — fused sparse-MLA vs O(n²) dense reference (S=4096, S_kv=4096, H=28)

### Forward

| topk | fused fwd (ms) | TFLOP/s | tok/s | peak GB | dense ref (ms) | dense GB | **speedup** | **mem ↓** |
|------|---------------:|--------:|------:|--------:|---------------:|---------:|------------:|----------:|
| 256  | **2.86**  | 22.4 | 1,432,921 | **0.259** | 152.59 | 6.75 | **53.4×** | **26.1×** |
| 512  | **4.95**  | 25.8 |   827,728 | **0.297** | 152.77 | 6.76 | **30.9×** | **22.8×** |
| 1024 | **9.26**  | 27.6 |   442,361 | **0.305** | 152.62 | 6.76 | **16.5×** | **22.2×** |
| 2048 | **18.10** | 28.2 |   226,324 | **0.322** | 152.34 | 6.78 | **8.4×**  | **21.1×** |

### Backward (preprocess + main bwd `AtomicAddx4` + postprocess)

| topk | fused bwd (ms) | TFLOP/s | tok/s | peak GB |
|------|---------------:|--------:|------:|--------:|
| 256  | **8.31**  | 19.4 | 492,727 | 0.523 |
| 512  | **14.09** | 22.9 | 290,617 | 0.561 |
| 1024 | **26.33** | 24.6 | 155,584 | 0.569 |
| 2048 | **50.80** | 25.4 |  80,634 | 0.586 |

### Combined fwd + bwd (one full attention fwd+bwd)

| topk | fwd+bwd (ms) | tok/s | peak GB |
|------|-------------:|------:|--------:|
| 256  | **11.17** | 366,650 | 0.523 |
| 512  | **19.04** | 215,096 | 0.561 |
| 1024 | **35.59** | 115,101 | 0.569 |
| 2048 | **68.90** |  59,453 | 0.586 |

All cases `out_nan=0`, `dq_nan=0`, `dkv_nan=0`. Raw measured JSON:
`docs/sparse_mla_bench_H28_gb10.json`.

### Shared-memory footprint (ptxas, measured — from SPARSE-MLA-GB10-RETILE.md)

The GB10 cap is **99 KiB/block** dynamic smem (`0x18c00` = 101376 B). Measured by
ptxas on real sm_121a:

| kernel | config | dynamic smem | vs 99 KiB cap |
|--------|--------|-------------:|---------------|
| **fwd** | `block_I=16, num_stages=1, drop-O, aggressive-merge` | **93.0 KiB** (95216 B) | FITS (5 KiB spare) |
| **bwd** | `threads=128, block_size=16, block_H=32` | **89.0 KiB** (merged ~86 KiB live) | FITS (10 KiB spare) |

Both fit; the `AtomicAddx4` dKV path is preserved (verified in generated CUDA).

## The headline numbers

At the model shapes (S=4096, H=28), the fused sparse-MLA **forward** is, vs the
O(n²) dense MLA reference at the same shapes:

- **53× faster and 26× less memory at topk=256**, scaling to
- **8.4× faster and 21× less memory at topk=2048**.

The speedup tracks sparsity exactly: the dense reference is **flat at ~152 ms /
~6.76 GB regardless of topk** (it always materializes the full S×S_kv attention
over *all* keys — O(S²)), while the fused kernel cost is **O(S·topk)** — it only
touches the `topk` gathered keys per query. So the advantage is largest when the
model is most sparse. Global-memory footprint: fused **~0.26–0.59 GB** (inputs +
small outputs, no S×S_kv matrix) vs dense **~6.75 GB** (dominated by the
[H, S, S_kv] fp32 score/softmax intermediate).

## Kernel-level vs full-model (the honest framing vs Megatron 3399 tok/s @ 26 GB)

**This benchmark is one attention operator, not a training step.** Concretely:

- **What it measures:** a single fused sparse-MLA fwd (and bwd) call on the GPU,
  timed in isolation, peak memory = just that op's tensors.
- **What Megatron's 3399 tok/s @ 26 GB measures:** the full 1.8B `local_gb10_quarter`
  forward **+** backward **+** optimizer step, across **all 13 layers** — and only
  3 of those layers (DSA A-layer ranks 1,2,3) even use sparse-MLA; the rest are
  Mamba3/M2RNN/MoE/FFN/embeddings, none of which this kernel touches. The 26 GB
  includes all weights, all activations, optimizer state, and every other op.

So you **cannot** read a full-model tok/s off this table. What this kernel
*does* establish is the per-op ceiling for the attention piece: at topk=2048,
S=4096, one fwd+bwd of sparse-MLA for **all 28 heads of one A-layer** costs
**~69 ms / ~0.59 GB**; at topk=256 it is **~11 ms / ~0.52 GB**. Three A-layers
⇒ roughly **3×** that per training step for the attention component (~33–207 ms
fwd+bwd depending on topk), a small and bounded slice of the full-step budget,
at <1 GB peak each.

### What is still needed for a full-model number with fused attention

1. **Wire the kernel into path_c's sparse_mla op.** Today path_c uses the
   gather/dense MLA path; the fused TileLang kernel must replace it in
   `cppmega_mlx/cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` (fwd+bwd) so the
   model actually calls it.
2. **Run the full `local_gb10_quarter` step** with that op active and measure
   end-to-end tok/s + peak — the apples-to-apples vs Megatron's 3399 / 26 GB.
   Note this is **independent** of the attention kernel: the model currently
   OOMs in MLX-eager at >118 GB, which is the loop/graph materialization limit,
   **not** an attention-memory problem (this kernel's attention peak is <1 GB).
   Replacing dense MLA with the fused kernel **removes the O(S²) ~6.8 GB
   attention intermediate per A-layer**, but the >118 GB eager OOM is elsewhere
   and must be addressed separately (graph segmentation / path_c fusion).

**Bottom line:** the fused sparse-MLA kernel is real, correct (cos 0.998–1.000),
and on GB10/sm_121a is **8–53× faster** and uses **21–26× less memory** than the
O(n²) dense reference at the model's S=4096/H=28 shapes. Translating that into a
full-model tok/s vs Megatron requires steps 1–2 above; this document delivers the
kernel-level number, not the full-model number.

## Reproduce

```bash
# on gb10, from <repo>/examples/deepseek_v32
/home/dave/cppmega-venv/bin/python bench_sparse_mla_gb10.py \
  --S 4096 --SKV 4096 --H 28 --threads 128 \
  --topks 256,512,1024,2048 --reps 50 --warmup 50 \
  --do_bwd --do_ref --out /tmp/sparse_mla_bench_H28.json
```
