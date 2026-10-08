# Evaluation report: `qwen3-w8a8-recipe`

## Status at a glance

| Item | State |
|---|---|
| Rollouts graded by the real grader | **2** (Batch A). Graded by grader v1, then rescored with the final rubric from logged measurements |
| Rollouts stopped by infrastructure | **14** (Batches B and C). The Modal workspace spend limit killed them during the agent phase, so no submissions were collected. One more never started |
| Reference graded by the grader | not yet. The packaged reference (`report/reference-solution/`) was queued twice and killed by the spend limit both times |
| Grader v2 (current `tests/grade.py`) validated on GPU | not yet. Its validation run (reference x3, no-op, 16 cheats) was killed by the spend limit |
| Grader v1 validated on GPU | yes, including the cheat results that motivated v2 (see "Grader validation") |

## Model and harness settings

| Setting | Value |
|---|---|
| Model | `openrouter/openai/gpt-6.1-sol` (OpenRouter id `openai/gpt-6.1-sol`) |
| Harness | Harbor 0.24.0, agent `terminus-2`, default sampling and reasoning (`kwargs: {}`) |
| Environment | Modal, 1x NVIDIA A10G, 8 vCPU, 32 GB RAM, 30 GB disk, public network |
| Timeouts | agent 3600 s, verifier 3600 s, image build 3600 s; recipe build + warm-up 1500 s inside the grader |
| Trials | independent, each in a fresh container from the same image; `max_retries: 0` |
| Config | `configs/rollouts.yaml` (`n_attempts` set per job with `-k`) |

Logs for every trial are in `jobs/<job>/<trial>/`: `agent/trajectory.json` (every step),
`verifier/reward.json` and `details.json` (the grade), and `artifacts/app/submission/` (the
submitted recipe). They can be browsed with `harbor view jobs`. Rubric: the reward is the quality
factor times the speed credit, where the speed credit is `clip(log(s) / log(1.85), 0, 1) ** 2`
(see README).

## Scores

### Batch A: graded (`jobs/qwen3-int8-a10g-rollouts-20261008-123648`)

| Trial | Steps | Cost | Speedup (grader) | Worst dNLL | Spearman / keep-half | Speed credit | Quality | **Reward** |
|---|---|---|---|---|---|---|---|---|
| `qaamH3h` | 57 | $0.41 | 1.765x | 0.0113 | 0.9982 / 0.979 | 0.853 | 0.934 | **0.797** |
| `n4SN5zX` | 49 | $0.49 | 1.680x | 0.0130 | 0.9990 / 0.990 | 0.711 | 0.850 | **0.604** |

**Mean 0.700, standard deviation 0.136 (sample, n = 2).**

Grader v1 graded these on the hidden set before the FineWeb web slice was added. Under v1's
original, looser rubric both scored 1.0. The rewards above apply the final rubric to the logged
measurements (`scripts/rescore.py`), so they reflect speed and quality on WikiText, MBPP and
GSM8K but not the web slice.

### Batches B and C: stopped by infrastructure

The Modal spend limit killed these during the agent phase (Harbor reports exit 137 as "likely out
of memory"), so no submissions exist and none can be graded. They are excluded from the
statistics above. The table records how far each agent had got, from its own `dev_bench.py` runs
on the dev split, and an **estimated reward**: the final rubric applied to the agent's last dev
measurement. That estimate is not a grader score. The dev split has no web slice, and the last
measurement is not necessarily what the agent would have submitted.

| Batch | Trial | Steps | Cost | Last dev speedup | Worst dev dNLL | Est. reward |
|---|---|---|---|---|---|---|
| B | `dGPkkXt` | 39 | $0.35 | 1.646x | 0.0096 | 0.656 |
| B | `DGiWiky` | 31 | $0.24 | 1.498x (best 1.689x) | 0.0102 | 0.427 |
| B | `s4QTHw7` | 35 | $0.21 | 1.492x | 0.0134 | 0.351 |
| B | `MSz6MQ5` | 32 | $0.20 | 1.444x | 0.0165 | 0.241 |
| B | `yDifqPc` | 28 | $0.16 | 1.446x | 0.0297 | 0.005 |
| B | `s3xays6` | 35 | $0.19 | 1.636x | 0.0322 | 0.000 |
| B | `kNbbdwk` | 28 | $0.21 | 1.510x | 0.0362 | 0.000 |
| B | `WoMHdDD` | 0 | $0 | never started | | |
| C | `QvxjzCJ` | 30 | $0.26 | 1.681x | 0.0087 | 0.713 |
| C | `ivPubG5` | 35 | $0.32 | 1.640x | 0.0104 | 0.634 |
| C | `EZyU6fH` | 37 | $0.40 | 1.391x | 0.0145 | 0.223 |
| C | `D6juZ2a` | 50 | $0.42 | 1.447x | 0.0270 | 0.054 |
| C | `3VXVzzc` | 32 | $0.20 | 1.509x | 0.0295 | 0.011 |
| C | `guwCfok` | 34 | $0.33 | 1.440x | 0.0355 | 0.000 |

Estimated over these 13: mean 0.255, standard deviation 0.276. Combined with the two graded
rollouts, the 15 agents span 0.00 to 0.80. That spread comes from real differences in approach
(see failure modes), not from noise near a threshold.

## Main failure modes

From the 15 trajectories that ran:

1. **Stopping at the off-the-shelf speed.** Most agents reach 1.39x to 1.51x with torchao INT8
   plus `torch.compile` and stay there (speed credit 0.30 to 0.44). The agents above 1.6x wrote a
   Triton INT8 GEMM with fused dequantization and captured the forward pass in CUDA graphs.
2. **Not recovering accuracy on the hard source.** WikiText is consistently the worst source.
   Agents without effective outlier handling end at dev dNLL 0.027 to 0.036 (`kNbbdwk`, `s3xays6`,
   `yDifqPc`, `guwCfok`, `3VXVzzc`, `D6juZ2a`), which zeroes or nearly zeroes the quality factor.
   SmoothQuant with a tuned alpha brings it to about 0.009 to 0.013. `s3xays6` shows the trade-off:
   1.636x, but dNLL 0.032 gives an estimated reward of 0.
3. **Trying FP8 on an A10G.** Six of the 15 trajectories spend steps on `float8`/`e4m3`. The A10G
   (Ampere) has no FP8 tensor cores, and working that out from the GPU name is part of the task.
4. **Running out of GPU memory in their own experiments.** Several agents load more than one copy
   of the model on the 24 GB card and crash their own benchmark. They recover, but lose steps.
5. **Regressing late.** `DGiWiky` measured 1.689x mid-run and 1.498x on its last benchmark.
   Agents do not always keep their best variant.

## Grader validation

**Grader v1** (`pilot/grader-v1/`, the original in-process grader) was validated on GPU:

| Check | Result |
|---|---|
| No-op agent | 0.0 (0.75x, eager BF16 is slower than the compiled baseline) |
| `bf16-passthrough`, `layer-skip`, `fp8-on-a10g` | 0.0 (no speedup / dNLL 4.02 / recipe error) |
| `naive-int8` | partial: 1.41x, dNLL 0.034 (0 under the final rubric) |
| **7 exploits** (`memoize`, `phase-switch`, `patch-harness`, `atexit-reward`, `read-hidden-lookup`, `public-lookup-table`, `net-fetch`) | **1.0: the exploits beat v1** |

**Grader v2** (current `tests/grade.py` + `worker.py`) was built to close every one of those
holes. Each defence targets the mechanism behind an exploit: worker-process isolation, deleted
hidden data, fresh audited timing batches, a clean baseline, and the reward written after a
process-group kill. Its GPU validation (reference x3, no-op, all 16 cheats in parallel, about
30 minutes) was launched and killed by the spend limit before grading. **It has not yet been run
to completion.**

**Score consistency.** Under v1, one recipe graded 1.546x, 1.630x and 1.533x on three runs.
Under the final smooth rubric that moves its speed credit between 0.48 and 0.63, against 0 to
0.08 under the earlier stepped curve. v2 should reduce this further: it uses medians over 15 fresh
batches per batch size and averages the baseline timed before and after the worker runs. That
needs to be confirmed by the reference x3 run.

## Why the task is difficult, and what separates strong from weak

- **There is no FP8 on the target GPU, and INT8 is inaccurate out of the box.** Qwen3-1.7B has
  activation outliers up to 180x the channel RMS. Plain INT8 W8A8 costs dNLL 0.033, and keeping
  sensitive modules in BF16 recovers only 20 to 37% of that, because the error is diffuse. Strong
  solutions move outliers into the weights (SmoothQuant) and calibrate in a way that holds on an
  unseen source.
- **The speed headroom is beyond what the compiler gives you.** torchao plus Inductor stops near
  1.44x. The INT8 matmuls alone would allow about 1.96x. The gap is quantize and dequantize
  traffic, the INT32 output round trip and launch overhead. Strong solutions write an INT8 GEMM
  with dequantization and the residual add in the epilogue, fuse SwiGLU with the next layer's
  quantization, merge QKV and gate/up, and use CUDA graphs.
- **Both axes are scored together.** A fast recipe that loses accuracy (`s3xays6`: 1.64x, 0) and
  an accurate one stuck near 1.45x (`MSz6MQ5`: 0.24) both score low. Only solutions that are
  accurate and fast at the same time approach 1.0.

The reference solution and the optimization path are in `report/REFERENCE.md`.

## Remaining steps (need GPU budget)

1. Grader v2 validation: reference x3, no-op, 16 cheats (about $15, 30 min). This verifies
   reference = 1.0, no-op = 0.0, every cheat at or below its cap, and score consistency, and it
   anchors REF_SPEEDUP to the reference's graded speed.
2. Five fresh rollouts with grader v2 (about $12, 80 min), run after step 1 rather than at the
   same time. Then fill in the scores, mean and standard deviation here.
3. Re-grade Batch A with v2 (`scripts/regrade.sh`) so all reported scores come from one grader.
