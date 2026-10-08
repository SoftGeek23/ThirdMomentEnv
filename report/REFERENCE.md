# Reference solution: all-INT8 Qwen3-1.7B-Base on A10G

The reference solution scores documents **1.933x faster than `torch.compile`d BF16** (geometric
mean over batch 16, 32 and 64), with a **mean per-source dNLL of 0.009 nats/token**.

| Batch | 16 | 32 | 64 | Geomean |
|---|---|---|---|---|
| Speedup vs compiled BF16 | 2.031x | 1.901x | 1.869x | **1.933x** |

| Source | WikiText | MBPP | GSM8K | Mean |
|---|---|---|---|---|
| dNLL vs BF16 (nats/token) | +0.0160 | +0.0059 | +0.0064 | **+0.0094** |

These figures come from the development benchmark on an NVIDIA A10G. BF16 and the reference were
timed interleaved on the same GPU, five rounds per batch size, on 256-token windows, and NLL was
measured on held-out WikiText-2, MBPP and GSM8K text.

## Code and reproduction

| What | Where |
|---|---|
| Reference solution (submission files) | `report/reference-solution/`: `recipe.py`, `fused.py`, `custom_int8.py`, `int8_kernel.py`, `act_stats.pt` |
| Installer | `report/reference-solution/solve.sh`: copies the files to `/app/submission/` and writes `calib.json` from `/app/dev/dev.json` |
| Development benchmark | `pilot/agentref_experiment.py`, variant `agent-b8-all` (run: `modal run pilot/agentref_experiment.py`) |
| Raw benchmark result | `pilot/logs/agentref1.log`, line `[agent-b8-all] RESULT {...}` |
| Grade it with the task grader | copy `report/reference-solution/` into `tasks/qwen3-w8a8-recipe/solution/`, then `N_ORACLE=3 scripts/validate.sh oracle` |

## What the reference solution is

The solution is a recipe, `build(model) -> fn`, that takes the pinned BF16 Qwen3-1.7B-Base and
returns a function from token IDs `[B, 256]` to final hidden states. It quantizes the whole
decoder to INT8 weights and activations, keeps accuracy with SmoothQuant, and runs on custom
Triton kernels inside a CUDA graph.

| File | Contents |
|---|---|
| `recipe.py` | `build()`: model rewrite, calibration of the boundary layers, INT8 conversion, per-block compile, CUDA graphs |
| `fused.py` | Restructured Qwen3 backbone: merged QKV and gate/up projections, flash attention |
| `custom_int8.py` | `Int8Linear`: INT8 weights, per-token INT8 activations, SmoothQuant folding, kernel dispatch |
| `int8_kernel.py` | Three custom Triton kernels: INT8 GEMM, fused SwiGLU quantizer, dual gate/up GEMM |
| `act_stats.pt`, `calib.json` | Per-channel activation maxima and calibration tokens, both from the development split |

### Why INT8 and why it is hard

Scoring is a single prefill pass over 256 tokens at batch 16 to 64, so the matmuls are
compute-bound. The A10G (Ampere, sm_86) has no FP8, but its INT8 tensor cores run at twice the
BF16 rate, so INT8 is the only 8-bit route to a speedup. The difficulty is accuracy. Qwen3-1.7B
has activation outliers up to 180x the channel RMS in its early MLP layers. Plain INT8 W8A8
raises NLL by 0.033 nats/token, and keeping the most sensitive modules in BF16 recovers only 20
to 37% of that, because the error is spread across many layers.

### Optimization path

| Stage | Speedup | Worst dNLL | What changed |
|---|---|---|---|
| torchao INT8 W8A8 + `torch.compile` | 1.439x | 0.033 | Baseline INT8, accuracy out of band |
| + SmoothQuant (alpha 0.5) | 1.488x | 0.012 | Outliers moved from activations into weights |
| + merged QKV / gate-up projections | 1.495x | | Fewer, larger GEMMs; one quantization per input |
| + Inductor max-autotune | 1.630x | 0.014 | Compiler-generated Triton epilogues |
| Custom Triton kernels + CUDA graphs, layers 0/27 BF16 | 1.678x | 0.012 | INT32 round trip and launch overhead removed |
| **All 28 layers INT8 (reference)** | **1.933x** | **0.016** | BF16 boundary layers, the largest remaining cost, removed |

The profile at the torchao stage (batch 64, GPU ms) shows where the time went:

| | Matmuls | Attention | Elementwise | Total |
|---|---|---|---|---|
| BF16 | 614 | 74 | 97 | 785 |
| INT8 + norm/quant kernel | 354 | 46 | 148 | 548 |

The INT8 matmuls were already 1.73x faster than BF16. The rest of the time was elementwise work
around them, mostly the INT32 matmul output being written to memory and read back for rescaling.
The matmul-plus-attention floor on this GPU is about 1.96x. The reference reaches 1.933x.

### Component 1: model rewrite (`fused.py`)

- **Q, K and V are one linear** (2048 to 4096) and **gate and up are one linear** (2048 to
  12288). Each layer runs two large GEMMs instead of five small ones, and each input is
  quantized once instead of three or two times.
- RMSNorm, the per-head q/k norms, RoPE and causal `scaled_dot_product_attention` (flash path,
  no mask) keep the model's exact semantics. GQA heads are expanded the way `transformers`
  does it.

### Component 2: INT8 W8A8 with SmoothQuant (`custom_int8.py`)

- **Weights:** INT8, one scale per output channel, quantized once at build time.
- **Activations:** INT8, one scale per token, computed at runtime in FP32.
- **SmoothQuant (alpha 0.45):** input channel j is divided by
  `s_j = act_max_j^0.45 / w_max_j^0.55`, and weight column j is multiplied by `s_j`. The
  product is unchanged, but the outlier channels' range moves from the activations, where a
  per-token scale cannot absorb it, into the weights, where per-channel scales can.
- **Calibration:** per-channel activation maxima are measured on the development split
  (`act_stats.pt`). The first and last layers are calibrated at build time on the shipped
  development tokens (`calib.json`), so **all 28 layers run in INT8**.

### Component 3: custom Triton kernels (`int8_kernel.py`)

**`mm_kernel`: INT8 GEMM with fused epilogue.** Used for every projection.
- INT8 tensor-core `tl.dot` with an INT32 accumulator.
- **Grouped-M tile ordering:** program IDs are remapped so neighbouring blocks reuse weight
  tiles from L2. On the gate/up GEMM this runs in 6.7 ms against 9.2 ms for cuBLAS.
- **Dequantization in the epilogue:** per-token and per-channel scales are applied in registers
  and BF16 is stored directly. The 4-byte INT32 intermediate never reaches global memory.
- **Residual add in the epilogue** for `o_proj` and `down_proj`, saving a read and a write of
  the hidden state per layer.
- **Tile shape chosen per GEMM shape:** 256x128x128 or 128x256x128 for the wide projections,
  128x128x64 for the narrow ones, with 3-stage software pipelining.

**`quant_kernel`: fused SwiGLU quantizer.** Used for the `down_proj` input at batch 32 and 64.
- One pass per token row computes SiLU(gate)·up, applies the SmoothQuant scale, takes the row
  maximum and writes INT8 values plus one FP32 scale. The widest activation in the model (6144
  channels) is read once and written once as INT8.

**`gu_kernel`: dual GEMM with SwiGLU epilogue.** Used at batch 16.
- Computes the gate and up accumulators in the same tile loop, dequantizes both and applies
  SwiGLU in the epilogue, so the 12288-wide intermediate is never written to memory.

### Component 4: execution (`recipe.py`)

- **`torch.compile` per decoder block.** Inductor fuses each RMSNorm with the SmoothQuant
  scaling and per-token quantization that follow it. Compiling per block keeps the build in
  minutes. Compiling the whole graph at once took over 10 minutes.
- **One CUDA graph per batch size** (16, 32, 64). The full 28-layer forward pass is captured
  once and replayed as a single launch, which removes several hundred kernel launches of CPU
  overhead per call.

## Levers for further gains

At 1.933x the remaining time splits roughly into INT8 GEMMs (about 70%), activation quantization
(about 10%), attention and its surrounding kernels (about 10%), and norms, RoPE and the rest.
Ranked by expected payoff:

### Speed

1. **Static activation scales fused into RMSNorm.** Per-token dynamic quantization needs a row
   reduction before every GEMM input. Calibrated per-tensor (or per-channel-group) static scales
   turn quantization into a pure elementwise multiply-round. That can live in the RMSNorm
   kernel's epilogue, which writes INT8 directly, and in the previous GEMM's epilogue for the
   `down_proj` input. Expected: most of the ~10% quantization share. This needs stronger
   outlier handling (item 7) to hold accuracy, because static scales cannot follow outlier
   tokens.
2. **Fused SwiGLU at every batch size.** The dual gate/up GEMM with the SwiGLU epilogue runs
   only at batch 16. Extending it to 32 and 64, with the INT8 quantization of the `down_proj`
   input also in that epilogue, removes the 12288-wide intermediate and a full quantization
   pass at the batch sizes that dominate throughput.
3. **GEMM scheduling: split-K and persistent kernels.** `down_proj` (K = 6144, N = 2048) and
   `o_proj` have a narrow N, so at batch 16 there are few output tiles per streaming
   multiprocessor. Split-K or stream-K decomposition, plus a persistent kernel that keeps tiles
   resident across the four projections, raises SM occupancy. Per-batch-size autotuning
   (`triton.autotune` over tile, warp and stage counts) and swizzled shared-memory layouts
   tighten the inner loop. Expected: 5 to 10% on the GEMM share, the largest absolute block of
   time.
4. **Attention-side fusion.** The q/k RMSNorms and RoPE can move into the QKV GEMM's epilogue,
   writing Q, K and V in the head-major layout attention expects. That removes three
   elementwise kernels and a transpose per layer. INT8 or FP16-accumulate attention is a
   further option at longer sequence lengths. At 256 tokens attention is a small share.
5. **Final norm in the last GEMM.** The final RMSNorm can be fused into the last `down_proj`
   epilogue together with the residual add.
6. **2:4 structured sparsity.** Ampere's sparse tensor cores double INT8 throughput on 2:4
   pruned weights. That could take the GEMM share down by up to half, but pruning a 1.7B model
   to 2:4 without fine-tuning costs accuracy, so it needs sparsity-aware calibration or a short
   recovery fine-tune to stay inside the quality band.

### Accuracy (headroom that buys speed)

7. **Hadamard rotations (QuaRot/SpinQuant style).** Rotating the residual stream and the
   `down_proj` input by orthogonal Hadamard matrices spreads outliers across channels. The
   rotations fold into the weights, except one online Hadamard before `down_proj`, which is
   cheap as a fast Walsh-Hadamard transform in the quantizer kernel. Flatter activations make
   per-tensor static scales viable (item 1) and lower dNLL, especially on unseen sources.
8. **Per-layer SmoothQuant strength.** A single alpha (0.45) is a compromise. Searching alpha
   per layer against end-to-end NLL on the development split, rather than per-module output
   error, puts strong smoothing only where outliers are severe.
9. **Better weight rounding (GPTQ / AWQ-style).** Error-compensating weight quantization lowers
   the INT8 weight error at the same speed. That tightens the worst-source dNLL (currently
   WikiText, +0.016) toward the 0.01 full-credit line.
10. **Calibration that generalizes.** Calibrating on a broader mix (code, math and web-style
    text, not only the development sources) and using percentile-clipped maxima instead of raw
    maxima makes the scales robust on unseen text. That is where the web-text evaluation slice
    is most sensitive.

### Engineering

11. **Whole-model capture with fewer graph breaks.** Exporting the backbone once
    (`torch.export` plus AOTInductor), with the custom kernels registered as `torch.library`
    ops, removes per-block compile overhead from the build and lets Inductor fuse across block
    boundaries, for example the residual add into the next RMSNorm.
12. **Kernel-friendly weight layout.** Storing INT8 weights pre-swizzled for the GEMM's load
    pattern removes the strided weight loads in the inner loop.
