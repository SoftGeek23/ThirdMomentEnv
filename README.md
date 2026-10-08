# ThirdMoment RL environment: `qwen3-w8a8-recipe`

A Terminal-Bench 4.0 / Harbor-format ML-engineering environment. The agent must make the
perplexity-scoring stage of a pretraining-data pipeline faster on an NVIDIA A10G. That means
running Qwen3-1.7B-Base over 256-token chunks at batch 16, 32 and 64, without changing which
documents a perplexity filter keeps. Any technique is allowed (quantization, kernel fusion,
custom Triton kernels, CUDA graphs). The grader measures speed against compiled BF16 and quality
against BF16 on hidden data.

## Repository layout

```
tasks/qwen3-w8a8-recipe/         the environment (Harbor task format)
  instruction.md                 the task prompt the agent receives
  task.toml                      resources, GPUs, timeouts, artifacts, separate verifier
  README.md                      difficulty, solution and verification explanations (rubric)
  environment/                   agent container
    Dockerfile, requirements.txt pinned stack; model weights + dev split baked in
    make_data.py                 builds the agent-visible dev split (no hidden data)
    data/                        copied to /app: starter submission/recipe.py, dev_bench.py, harness.py
  tests/                         verifier container (never visible to the agent)
    Dockerfile, requirements.txt pinned stack; model weights + hidden test split baked in
    make_data.py                 builds the hidden test split (WikiText, MBPP, GSM8K, FineWeb web slice)
    harness.py                   shared NLL / ranking / timing utilities (copy in environment/data/)
    grade.py, test.sh            the grader and its entry point
  solution/                      reference solution (solve.sh + recipe)
  cheat/<name>/                  shortcut and flawed solutions, each with an expected_max cap
configs/rollouts.yaml            the exact harness, model and settings used for graded rollouts
scripts/                         lint, validate, rollouts, regrade, rescore, summarize
report/                          EVAL_REPORT.md, REFERENCE.md, reference-solution/
  rollouts/<batch>/<trial>/      per-trial logs: trajectory, result, grade, submitted recipe
  validation/<job>/              grader results for reference, no-op and cheat runs
reference/tb-checks/             Terminal-Bench v4.0.0 CI checks (used by scripts/lint.sh)
pilot/                           pilot studies and reference experiments (Modal scripts, logs, results)
jobs/                            Harbor run outputs: logs, trajectories, rewards (git-ignored)
```

## Requirements

| What | Version / detail |
|---|---|
| Harbor | 0.24.0 (`uv tool install 'harbor[modal]'`), includes Modal 1.6.1 |
| Modal account | GPU access needs a payment method; runs use the `A10G` GPU type |
| OpenRouter API key | for agent rollouts (`openrouter/openai/gpt-6.1-sol`) |
| Local machine | any OS with Python 3.11+. Builds and runs happen on Modal, nothing heavy runs locally |

Pinned stack inside both containers: Python 3.12.11, torch 2.8.0 (CUDA 12.8, Triton 3.4),
torchao 0.13.0, transformers 4.56.2, huggingface-hub 0.35.3, datasets 4.1.1, numpy 2.3.3. Model:
`Qwen/Qwen3-1.7B-Base` at revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. All datasets are
pinned to commit hashes in `make_data.py`.

## One-time setup

```bash
uv tool install 'harbor[modal]'                       # Harbor 0.24.0 + Modal 1.6.1
~/.local/share/uv/tools/harbor/bin/modal token new    # Modal login (opens a browser)
cp .env.example .env && $EDITOR .env                  # OPENROUTER_API_KEY=sk-or-...
```

## Build

There is no separate build step. Harbor builds both images on Modal the first time a task runs,
from `environment/Dockerfile` (agent) and `tests/Dockerfile` (verifier), and caches them. The
first build takes about 10 minutes: torch, the 3.4 GB model, and the dataset downloads for the
dev and hidden splits. After that, only changed layers rebuild. Grader edits are cheap because
the weights and data layers come before `COPY . /tests/`.

To check that the task is well formed:

```bash
scripts/lint.sh          # Terminal-Bench v4.0.0 static checks (two TB-repo-only checks are skipped, see the script)
```

To open a shell in the agent's environment (A10G, pinned stack, model, dev data):

```bash
harbor tasks start-env -p tasks/qwen3-w8a8-recipe -e modal -a -i
python /app/dev_bench.py     # inside: score /app/submission/recipe.py on the dev split, same checks as the grader
```

## Run

**Graded rollouts** (the evaluation): terminus-2 + gpt-6.1-sol, settings in `configs/rollouts.yaml`.

```bash
scripts/rollouts.sh [job-name]                    # 5 trials (n_attempts in configs/rollouts.yaml)
harbor run -c configs/rollouts.yaml -k 6 -n 6 -o jobs --job-name <name> -y --env-file .env   # e.g. 6 trials
```

Each trial runs in a fresh agent container. The agent writes `/app/submission/` (`recipe.py`
plus any kernels and calibration files), and only that directory is passed to the verifier.

**Reference, no-op and cheat solutions** (grader validation):

```bash
scripts/validate.sh                  # all of the below
scripts/validate.sh oracle           # reference: N_ORACLE runs (default 3), must all score 1.0 and agree
scripts/validate.sh nop              # no-op agent: must score 0.0
scripts/validate.sh cheats           # each cheat/<name>/solve.sh: must score <= cheat/<name>/expected_max
```

## Grade

Grading is automatic. After each trial, Harbor starts a separate verifier container on a fresh
A10G, runs `tests/test.sh`, and stores the result in `jobs/<job>/<trial>/verifier/`:
`reward.json` (reward plus raw measurements) and `details.json` (per-batch speedups, per-source
dNLL, ranking checks, errors).

```bash
python3 scripts/summarize.py jobs/<job>          # table of rewards and components, mean, std
scripts/regrade.sh jobs/<job> [...]              # re-grade saved submissions with the CURRENT grader
python3 scripts/rescore.py --ref <x> jobs/<job>  # recompute rewards for a new REF_SPEEDUP, no GPU needed
harbor view jobs                                 # browse trajectories at http://127.0.0.1:8080
```

### Scoring rubric (`tests/grade.py`)

The grader scores outcomes only, so any technique is valid. Full rationale is in
`tasks/qwen3-w8a8-recipe/README.md` (Verification explanation).

1. **Ranking gate.** At every batch size, the Spearman correlation of per-document perplexity
   with BF16 must be at least 0.99, and agreement on the kept (lowest-perplexity) half must be at
   least 0.95. Otherwise the reward is 0.
2. **Quality factor.** Let d be the worst per-source mean dNLL vs BF16 (nats/token) over
   WikiText, MBPP, GSM8K and the unseen FineWeb web slice, at batch 16/32/64. The factor is 1 if
   d <= 0.01, falls linearly to 0 at d = 0.03, and is 0 beyond that.
3. **Speed credit.** The geometric-mean speedup over batch 16/32/64 is measured against
   `torch.compile`d BF16. Credit is `clip(log(s) / log(REF_SPEEDUP), 0, 1) ** 2`: smooth, log-scaled
   and convex. Every real speedup earns some credit, and the hardest gains earn the most.
4. **Reward** = quality factor x speed credit. The reference scores 1.0 and no speedup scores 0.

Worked examples (REF_SPEEDUP 1.85):

| Solution | Speedup | Worst dNLL | Speed credit | Quality | Reward |
|---|---|---|---|---|---|
| no-op / BF16 passthrough | 1.0x | 0 | 0 | 1 | 0 |
| torchao INT8 + compile | 1.44x | 0.033 | 0.35 | 0 | 0 |
| SmoothQuant + torchao | 1.49x | 0.012 | 0.42 | 0.90 | 0.38 |
| custom INT8 GEMM, BF16 boundary layers | 1.68x | 0.013 | 0.71 | 0.85 | 0.60 |
| custom GEMM + CUDA graphs, all INT8 | 1.765x | 0.011 | 0.85 | 0.93 | 0.80 |

**Grader hardening.** The submission runs in a separate, unprivileged worker process. Timing uses
fresh batches the submission has never seen, and every timed output is audited against BF16. The
baseline is timed while no submission code is loaded. The hidden data is deleted from disk once
loaded, and the reward is written only after the worker's process group is killed. Details are in
the task README. The cheat solutions in `cheat/` exercise each defence.

## Task configuration (`tasks/qwen3-w8a8-recipe/task.toml`)

| Setting | Agent environment | Verifier environment |
|---|---|---|
| GPU | 1x NVIDIA A10G (24 GB, Ampere sm_86: INT8 tensor cores, no FP8) | 1x A10G, a fresh container |
| CPUs / memory / disk | 8 vCPU / 32 GB / 30 GB | 8 vCPU / 32 GB / 30 GB |
| Image build timeout | 3600 s | 3600 s |
| Run timeout | agent: 3600 s | verifier: 3600 s |
| Network | public (open internet, Terminal-Bench convention) | public |
| Inputs | instruction, `/app` (starter recipe, dev split, dev_bench) | `/app/submission` only (the declared artifact) |

Limits enforced inside the grader:

| Limit | Value |
|---|---|
| `recipe.build()` plus the first call at each batch size | 1500 s (25 min) |
| Workload | `[B, 256]` token IDs, B in {16, 32, 64}; output `[B, 256, 2048]` final hidden states |
| Hidden evaluation set | 64 documents each from WikiText-2 test, MBPP test, GSM8K test and FineWeb (web) |

## Compute requirements and cost

| Run | GPUs | Typical wall time | Approximate Modal cost |
|---|---|---|---|
| One rollout | A10G for the agent (up to 60 min), then a fresh A10G for grading | 45 to 80 min | $1.5 to $2.5 |
| 5 to 6 rollouts in parallel | up to 6 A10Gs at once | about 80 min | $10 to $15 |
| Reference validation (`validate.sh oracle`, 3 runs) | 3 A10Gs | about 25 min | about $3 |
| Cheat validation (16 cheats; `validate.sh` runs them one at a time, which takes several hours; launched in parallel they finish in about 30 min) | 1 to 16 A10Gs | 30 min (parallel) | $10 to $15 |

The agent's OpenRouter cost is about $0.40 to $0.50 per rollout. The first image build adds
about 10 minutes to the first run. Set the Modal workspace spend limit with headroom: runs that
exceed it are killed mid-trial and lose their submissions.

## Rollout protocol

- Independent trials (`n_attempts`), each in a fresh container built from the same image, with
  the same harness (`terminus-2`), model (`openrouter/openai/gpt-6.1-sol`), default sampling
  (`kwargs: {}`) and the same task timeouts and resources. No retries (`max_retries: 0`).
- A trial killed by infrastructure (for example, a Modal spend limit) is reported separately
  rather than scored 0.
- Targets: mean reward < 0.5 and standard deviation > 0.08. `summarize.py` reports both the
  sample (ddof=1) and population standard deviation.
