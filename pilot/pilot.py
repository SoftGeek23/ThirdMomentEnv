"""Pilot: is mixed precision necessary for Qwen3-1.7B-Base on one L4?

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/pilot.py

Stages (each variant runs in its own L4 container, in parallel):
  0. prepare      (CPU)  download pinned model + datasets to a Volume, build fixed token sets
  1. evaluate     bf16 | int8 (W8A8) | fp8 (W8A8, per-row)   + sensitivity ranking for int8/fp8
  2. evaluate     int8/fp8 with the top-k most sensitive projections kept in BF16
Every evaluate container benchmarks BF16 on its own GPU first, so speedups are
within-device ratios, not cross-container comparisons.

Decision thresholds are fixed below, before any results are seen.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

import modal

# ----------------------------------------------------------------------------- pins
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
MODEL_REV = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")
MBPP = ("google-research-datasets/mbpp", "full", "4bb6404fdc6cacfda99d4ac4205087b89d32030c")
GSM8K = ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866")
GPU = "L4"

# --------------------------------------------------------------------- experiment
SEED = 1234
N_EVAL = 128  # per slice, from held-out (test) splits
N_CALIB = 16  # per slice, from train splits; used only for sensitivity ranking
WIKI_CTX, WIKI_CONT = 256, 256
MAX_LEN = 1024
BENCH_LEN, BENCH_BATCH = 256, 16
WARMUP, ITERS_B1, ITERS_B16 = 10, 50, 20
TOPK = (4, 8, 16)
BOOTSTRAP = 4000

# ------------------------------------------------- pre-registered decision thresholds
MIN_SPEEDUP = 1.20  # uniform low precision must be >= 1.2x on batch-1 latency or batch-16 tok/s
MIN_DNLL = 0.02  # a slice "degrades" if mean dNLL >= 0.02 nats/token AND its 95% CI excludes 0
RECOVER_FRAC = 0.75  # protection must remove >= 75% of the regression on every degraded slice
RETAIN_SPEED_FRAC = 0.75  # ...while keeping >= 75% of the uniform mode's speed gain

TARGET_RE = r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$"
CACHE = "/cache"
DATA_PATH = f"{CACHE}/pilot/data-{SEED}.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "torchao==0.13.0",
        "transformers==4.56.2",
        "huggingface-hub==0.35.3",
        "datasets==4.1.1",
        "numpy==2.3.3",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false"})
)
vol = modal.Volume.from_name("thirdmoment-pilot-cache", create_if_missing=True)
app = modal.App("thirdmoment-precision-pilot", image=image)


# =============================================================== stage 0: prepare
@app.function(volumes={CACHE: vol}, cpu=4, memory=16384, timeout=1800)
def prepare() -> dict:
    import hashlib
    import os

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_download(MODEL_ID, revision=MODEL_REV)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    rng = random.Random(SEED)

    def pair(slice_, idx, prompt, target):
        p, t = enc(prompt), enc(target)
        ids = (p + t)[:MAX_LEN]
        return {"slice": slice_, "idx": idx, "ids": ids, "start": len(p),
                "preview": target[:120].replace("\n", "\\n")}

    def wiki_windows(split, n, length):
        ds = load_dataset(WIKITEXT[0], WIKITEXT[1], revision=WIKITEXT[2], split=split)
        ids = enc("".join(ds["text"]))
        starts = rng.sample(range(len(ids) // length), n)
        return [ids[s * length:(s + 1) * length] for s in sorted(starts)]

    def wiki(split, n):
        out = []
        for i, w in enumerate(wiki_windows(split, n, WIKI_CTX + WIKI_CONT)):
            out.append({"slice": "wikitext", "idx": i, "ids": w, "start": WIKI_CTX,
                        "preview": tok.decode(w[WIKI_CTX:WIKI_CTX + 40]).replace("\n", "\\n")})
        return out

    def mbpp(split, n):
        ds = load_dataset(MBPP[0], MBPP[1], revision=MBPP[2], split=split)
        rows = rng.sample(range(len(ds)), n)
        return [pair("mbpp", int(ds[r]["task_id"]),
                     ds[r]["text"] + "\n" + "\n".join(ds[r]["test_list"]) + "\n",
                     ds[r]["code"]) for r in rows]

    def gsm(split, n):
        ds = load_dataset(GSM8K[0], GSM8K[1], revision=GSM8K[2], split=split)
        rows = rng.sample(range(len(ds)), n)
        return [pair("gsm8k", r, f"Question: {ds[r]['question']}\nAnswer:", " " + ds[r]["answer"])
                for r in rows]

    data = {
        "eval": wiki("test", N_EVAL) + mbpp("test", N_EVAL) + gsm("test", N_EVAL),
        "calib": wiki("train", N_CALIB) + mbpp("train", N_CALIB) + gsm("train", N_CALIB),
        "bench": wiki_windows("validation", BENCH_BATCH, BENCH_LEN),
    }
    blob = json.dumps(data, sort_keys=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    Path(DATA_PATH).write_text(blob)
    vol.commit()
    return {
        "data_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "n_eval": {s: sum(e["slice"] == s for e in data["eval"]) for s in ("wikitext", "mbpp", "gsm8k")},
        "scored_tokens": {s: sum(len(e["ids"]) - e["start"] for e in data["eval"] if e["slice"] == s)
                          for s in ("wikitext", "mbpp", "gsm8k")},
    }


# ============================================================ GPU-side helpers
def _setup():
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    torch.manual_seed(SEED)
    return json.loads(Path(DATA_PATH).read_text())


def _load_model():
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return m.cuda().eval()


def _config(mode):
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                      Int8DynamicActivationInt8WeightConfig, PerRow)
    if mode == "int8":
        # set_inductor_config=False: torchao would otherwise enable coordinate-descent autotuning
        # for the quantized model only, giving it kernel tuning the BF16 baseline never gets.
        return Int8DynamicActivationInt8WeightConfig(set_inductor_config=False)  # per-token act, per-channel weight
    if mode == "fp8":
        return Float8DynamicActivationFloat8WeightConfig(granularity=PerRow(), set_inductor_config=False)
    raise ValueError(mode)


def _quantize(model, mode, protect):
    import re

    import torch
    from torchao.quantization import quantize_

    pat, protect, chosen = re.compile(TARGET_RE), set(protect), []

    def keep(m, fqn):
        ok = isinstance(m, torch.nn.Linear) and bool(pat.match(fqn)) and fqn not in protect
        if ok:
            chosen.append(fqn)
        return ok

    quantize_(model, _config(mode), filter_fn=keep)
    return chosen


def _nll(model, items, bs=8):
    import torch

    order = sorted(range(len(items)), key=lambda i: (len(items[i]["ids"]), i))
    out = [None] * len(items)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            chunk = order[b:b + bs]
            L = max(len(items[i]["ids"]) for i in chunk)
            ids = torch.zeros((len(chunk), L), dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, i in enumerate(chunk):
                n = len(items[i]["ids"])
                ids[j, :n] = torch.tensor(items[i]["ids"])
                mask[j, :n] = 1
            h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
            for j, i in enumerate(chunk):
                it = items[i]
                n, s = len(it["ids"]), it["start"]
                logp = torch.log_softmax(model.lm_head(h[j, s - 1:n - 1]).float(), dim=-1)
                tgt = torch.tensor(it["ids"][s:n], device="cuda")
                tok_nll = -logp.gather(1, tgt[:, None]).squeeze(1)
                out[i] = {"slice": it["slice"], "idx": it["idx"], "preview": it["preview"],
                          "nll_sum": float(tok_nll.sum()), "n_tok": n - s}
    return out


def _profile(fn):
    """Count matmul ops and list top CUDA kernels for one call."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    ops = {e.key: e.count for e in ev
           if e.key in ("aten::_int_mm", "aten::_scaled_mm", "aten::mm", "aten::addmm", "aten::bmm")}
    dev = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    kern = sorted((e for e in ev if dev(e) > 0 and not e.key.startswith("aten::")), key=dev, reverse=True)
    return {"ops": ops, "top_kernels": [{"name": e.key[:110], "us": round(dev(e), 1)} for e in kern[:12]]}


def _bench(model, bench):
    import torch

    torch._dynamo.reset()
    cm = torch.compile(model, dynamic=False)
    x = torch.tensor(bench, device="cuda")
    x1, x16 = x[:1], x[:BENCH_BATCH]

    def fwd(t):
        return cm(input_ids=t, use_cache=False, logits_to_keep=1).logits

    def timed(t, iters):
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fwd(t)
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        ts.sort()
        return {"median_ms": statistics.median(ts), "p10_ms": ts[len(ts) // 10],
                "p90_ms": ts[(9 * len(ts)) // 10]}

    with torch.no_grad():
        t0 = time.time()
        for _ in range(WARMUP):
            fwd(x1)
            fwd(x16)
        torch.cuda.synchronize()
        compile_warmup_s = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        b1 = timed(x1, ITERS_B1)
        b16 = timed(x16, ITERS_B16)
        peak = torch.cuda.max_memory_allocated()
        prof = _profile(lambda: fwd(x1))
    return {"b1_latency": b1, "b16": b16,
            "b16_tok_per_s": BENCH_BATCH * BENCH_LEN / (b16["median_ms"] / 1e3),
            "peak_mem_gb": peak / 2**30, "compile_warmup_s": compile_warmup_s,
            "compiled_profile": prof}


def _log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def _free(*objs):
    import gc

    import torch

    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================== stage 1/2: evaluate
@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=2400)
def evaluate(variant: dict) -> dict:
    import torch

    data = _setup()
    name, mode, protect = variant["name"], variant["mode"], variant.get("protect", [])
    res = {"variant": variant, "gpu": torch.cuda.get_device_name()}

    model = _load_model()
    _log(name, "bench bf16 baseline")
    res["baseline_perf"] = _bench(model, data["bench"])  # same-GPU BF16 reference
    if mode == "bf16":
        res["quantized_modules"] = []
        res["weights_gb"] = torch.cuda.memory_allocated() / 2**30
        res["perf"] = res["baseline_perf"]
        res["eager_profile"] = _profile(lambda: model(input_ids=torch.tensor(data["bench"][:1], device="cuda"),
                                                      use_cache=False, logits_to_keep=1))
    else:
        _free(model)
        torch._dynamo.reset()
        model = _load_model()
        res["quantized_modules"] = _quantize(model, mode, protect)
        _log(name, f"quantized {len(res['quantized_modules'])} modules; profiling + bench")
        torch.cuda.synchronize()
        res["weights_gb"] = torch.cuda.memory_allocated() / 2**30
        x1 = torch.tensor(data["bench"][:1], device="cuda")
        with torch.no_grad():
            res["eager_profile"] = _profile(lambda: model(input_ids=x1, use_cache=False, logits_to_keep=1))
        res["perf"] = _bench(model, data["bench"])
    _log(name, "nll")
    res["nll"] = _nll(model, data["eval"])
    print(f"[{name}] done: b1 {res['perf']['b1_latency']['median_ms']:.2f} ms, "
          f"b16 {res['perf']['b16_tok_per_s']:.0f} tok/s")
    return res


# ========================================================== stage 1: sensitivity
@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=2400)
def sensitivity(mode: str) -> list[dict]:
    """Local output error of each projection when quantized alone, on calibration data.

    One BF16 forward pass per calibration example. A hook on every target Linear feeds
    the same input through a quantized copy and accumulates ||y_q - y||^2 / ||y||^2.
    """
    import copy
    import re

    import torch
    from torchao.quantization import quantize_

    data = _setup()
    model = _load_model()
    _log(f"sens-{mode}", "start")
    pat = re.compile(TARGET_RE)
    targets = {f: m for f, m in model.named_modules() if pat.match(f)}
    qcopy, stats, hooks = {}, {}, []
    for f, m in targets.items():
        seq = torch.nn.Sequential(copy.deepcopy(m))
        quantize_(seq, _config(mode))
        qcopy[f] = seq
        stats[f] = {"err": 0.0, "ref": 0.0, "amax": 0.0, "sq": 0.0, "n": 0}

    def make_hook(f):
        def hook(_m, inp, out):
            x = inp[0]
            d = qcopy[f](x).float() - out.float()
            s = stats[f]
            s["err"] += d.pow(2).sum().item()
            s["ref"] += out.float().pow(2).sum().item()
            xf = x.float()
            s["amax"] = max(s["amax"], xf.abs().max().item())
            s["sq"] += xf.pow(2).sum().item()
            s["n"] += xf.numel()
        return hook

    for f, m in targets.items():
        hooks.append(m.register_forward_hook(make_hook(f)))
    with torch.no_grad():
        for it in data["calib"]:
            model(input_ids=torch.tensor([it["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()

    rank = [{"fqn": f, "rel_err": s["err"] / max(s["ref"], 1e-30),
             "act_absmax": s["amax"], "act_amax_over_rms": s["amax"] / math.sqrt(s["sq"] / max(s["n"], 1)),
             "params": targets[f].weight.numel()} for f, s in stats.items()]
    rank.sort(key=lambda r: r["rel_err"], reverse=True)
    _log(f"sens-{mode}", "done")
    return rank


# =================================================================== analysis
SLICES = ("wikitext", "mbpp", "gsm8k")


def _per_example(nll):
    return {(r["slice"], r["idx"]): r["nll_sum"] / r["n_tok"] for r in nll}


def _boot_ci(xs, rng):
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def _quality(base, var):
    b, v = _per_example(base["nll"]), _per_example(var["nll"])
    rng = random.Random(SEED)
    out = {}
    for s in SLICES:
        keys = sorted(k for k in b if k[0] == s)
        d = [v[k] - b[k] for k in keys]
        tok = {r["idx"]: r["n_tok"] for r in base["nll"] if r["slice"] == s}
        bsum = sum(r["nll_sum"] for r in base["nll"] if r["slice"] == s)
        vsum = sum(r["nll_sum"] for r in var["nll"] if r["slice"] == s)
        lo, hi = _boot_ci(d, rng)
        prev = {r["idx"]: r["preview"] for r in base["nll"] if r["slice"] == s}
        worst = sorted(zip(d, keys), reverse=True)[:5]
        out[s] = {
            "base_nll": statistics.fmean(b[k] for k in keys),
            "dnll_mean": statistics.fmean(d), "ci95": [lo, hi],
            "dnll_token_weighted": (vsum - bsum) / sum(tok.values()),
            "worst": [{"idx": k[1], "dnll": x, "preview": prev[k[1]]} for x, k in worst],
        }
        out[s]["degraded"] = out[s]["dnll_mean"] >= MIN_DNLL and lo > 0
    return out


def _speed(r):
    bl, p = r["baseline_perf"], r["perf"]
    return {"b1_speedup": bl["b1_latency"]["median_ms"] / p["b1_latency"]["median_ms"],
            "b16_speedup": p["b16_tok_per_s"] / bl["b16_tok_per_s"]}


def _lowp_calls(r):
    ops = r["eager_profile"]["ops"]
    return ops.get("aten::_int_mm", 0) + ops.get("aten::_scaled_mm", 0)


def analyze(results, ranks):
    base = results["bf16"]
    rows, verdict = {}, {}
    for name, r in results.items():
        rows[name] = {"quality": _quality(base, r), "speed": _speed(r),
                      "n_quantized": len(r["quantized_modules"]), "lowp_matmul_calls_eager": _lowp_calls(r),
                      "lowp_ops_compiled": r["perf"]["compiled_profile"]["ops"],
                      "peak_mem_gb": r["perf"]["peak_mem_gb"], "weights_gb": r["weights_gb"],
                      "b1_ms": r["perf"]["b1_latency"]["median_ms"], "b16_tok_s": r["perf"]["b16_tok_per_s"],
                      "b16_ms": r["perf"]["b16"]["median_ms"],
                      "b1_p10_p90": [r["perf"]["b1_latency"]["p10_ms"], r["perf"]["b1_latency"]["p90_ms"]],
                      "b16_p10_p90": [r["perf"]["b16"]["p10_ms"], r["perf"]["b16"]["p90_ms"]],
                      "bf16_b1_ms_same_gpu": r["baseline_perf"]["b1_latency"]["median_ms"]}

    for mode in ("int8", "fp8"):
        u = rows[mode]
        sp = u["speed"]
        metric = "b16_speedup" if sp["b16_speedup"] >= sp["b1_speedup"] else "b1_speedup"
        fast = max(sp.values()) >= MIN_SPEEDUP
        kernels_ok = u["lowp_matmul_calls_eager"] >= u["n_quantized"] > 0
        degraded = [s for s in SLICES if u["quality"][s]["degraded"]]
        rec = []
        for k in TOPK:
            p = rows.get(f"{mode}-protect{k}")
            if not p:
                continue
            recovered = all(p["quality"][s]["dnll_mean"] <= (1 - RECOVER_FRAC) * u["quality"][s]["dnll_mean"]
                            for s in degraded) if degraded else False
            gain_u = sp[metric] - 1
            kept = (p["speed"][metric] - 1) / gain_u if gain_u > 0 else 0.0
            rec.append({"k": k, "recovered": recovered, "speed_gain_retained": kept,
                        "ok": recovered and kept >= RETAIN_SPEED_FRAC})
        ok_k = [x["k"] for x in rec if x["ok"]]
        if not kernels_ok:
            v = "INVALID: low-precision matmuls not confirmed"
        elif not fast:
            v = "STOP: not meaningfully faster -> change the inference stack"
        elif not degraded:
            v = "NO PUZZLE: fast and quality-safe uniformly"
        elif ok_k:
            v = f"CONTINUE: degrades on {degraded}; protecting top-{min(ok_k)} recovers with speed kept"
        else:
            v = f"WEAK: degrades on {degraded} but no top-k protection recovered within the speed budget"
        verdict[mode] = {"verdict": v, "fast": fast, "speed_metric": metric, "kernels_confirmed": kernels_ok,
                         "degraded_slices": degraded, "protection": rec}

    overall = any(v["verdict"].startswith("CONTINUE") for v in verdict.values())
    trivial = [m for m, v in verdict.items() if v["verdict"].startswith("NO PUZZLE")]
    return {"rows": rows, "per_mode": verdict, "continue": overall,
            "caveat": (f"uniform {trivial} is already fast and safe; a full task must exclude it "
                       "or it is a trivial solution") if overall and trivial else None,
            "top_sensitive": {m: ranks[m][:16] for m in ranks}}


def render(meta, a) -> str:
    L = ["# Precision pilot: Qwen3-1.7B-Base on L4", "",
         f"- model `{MODEL_ID}@{MODEL_REV[:7]}`, torch 2.8.0, torchao 0.13.0, transformers 4.56.2",
         f"- eval tokens/slice: {meta['scored_tokens']}, data sha256 `{meta['data_sha256'][:12]}`",
         f"- thresholds: speedup >= {MIN_SPEEDUP}, dNLL >= {MIN_DNLL} with CI>0, "
         f"recover >= {RECOVER_FRAC:.0%}, keep >= {RETAIN_SPEED_FRAC:.0%} of speed gain", "",
         "## Variants", "",
         "| variant | #quant | lowp calls | b1 ms [p10-p90] | b1 x | b16 ms [p10-p90] | b16 tok/s | b16 x | peak GB | "
         + " | ".join(f"dNLL {s} [95% CI]" for s in SLICES) + " |",
         "|" + "---|" * (9 + len(SLICES))]
    for name, r in a["rows"].items():
        q = r["quality"]
        b1r, b16r = r["b1_p10_p90"], r["b16_p10_p90"]
        L.append(f"| {name} | {r['n_quantized']} | {r['lowp_matmul_calls_eager']} | "
                 f"{r['b1_ms']:.2f} [{b1r[0]:.2f}-{b1r[1]:.2f}] | {r['speed']['b1_speedup']:.2f} | "
                 f"{r['b16_ms']:.1f} [{b16r[0]:.1f}-{b16r[1]:.1f}] | {r['b16_tok_s']:.0f} | "
                 f"{r['speed']['b16_speedup']:.2f} | "
                 f"{r['peak_mem_gb']:.2f} | "
                 + " | ".join(f"{q[s]['dnll_mean']:+.4f} [{q[s]['ci95'][0]:+.4f}, {q[s]['ci95'][1]:+.4f}]"
                              + (" *" if q[s]["degraded"] else "") for s in SLICES) + " |")
    L += ["", "`*` = degraded beyond threshold. Speedups are vs BF16 measured on the same GPU.", "",
          "## Verdict", "", f"**{'CONTINUE' if a['continue'] else 'DO NOT CONTINUE (as specified)'}**", ""]
    for m, v in a["per_mode"].items():
        L.append(f"- **{m}**: {v['verdict']}")
        for p in v["protection"]:
            L.append(f"  - top-{p['k']}: recovered={p['recovered']}, "
                     f"speed gain retained={p['speed_gain_retained']:.0%}")
    if a["caveat"]:
        L += ["", f"Caveat: {a['caveat']}"]
    for m in ("int8", "fp8"):
        L += ["", f"## Most sensitive projections ({m}, calibration data)", "",
              "| fqn | rel err | act amax/rms |", "|---|---|---|"]
        L += [f"| {r['fqn']} | {r['rel_err']:.2e} | {r['act_amax_over_rms']:.0f} |"
              for r in a["top_sensitive"][m][:10]]
        L += ["", f"Worst examples, uniform {m}:"]
        for s in SLICES:
            w = a["rows"][m]["quality"][s]["worst"][:3]
            L += [f"- {s}: " + "; ".join(f"#{x['idx']} {x['dnll']:+.3f} `{x['preview'][:50]}`" for x in w)]
    return "\n".join(L) + "\n"


def rerender(json_path: str) -> str:
    """Rebuild the markdown report from a saved results JSON (no GPU)."""
    blob = json.loads(Path(json_path).read_text())
    md = render(blob["meta"], analyze(blob["raw"], blob["ranks"]))
    Path(json_path).with_suffix(".md").write_text(md)
    return md


# ================================================================= orchestration
@app.local_entrypoint()
def main():
    t0 = time.time()
    meta = prepare.remote()
    print("prepared:", meta)

    sens = {m: sensitivity.spawn(m) for m in ("int8", "fp8")}
    stage1 = [{"name": "bf16", "mode": "bf16"}, {"name": "int8", "mode": "int8"}, {"name": "fp8", "mode": "fp8"}]
    results = {r["variant"]["name"]: r for r in evaluate.map(stage1)}
    ranks = {m: h.get() for m, h in sens.items()}
    print(f"stage 1 done at {time.time() - t0:.0f}s")

    stage2 = [{"name": f"{m}-protect{k}", "mode": m, "protect": [r["fqn"] for r in ranks[m][:k]]}
              for m in ("int8", "fp8") for k in TOPK]
    results.update({r["variant"]["name"]: r for r in evaluate.map(stage2)})
    print(f"stage 2 done at {time.time() - t0:.0f}s")

    a = analyze(results, ranks)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{ts}.json").write_text(json.dumps({"meta": meta, "analysis": a, "raw": results, "ranks": ranks},
                                               indent=1, default=str))
    md = render(meta, a)
    (out / f"{ts}.md").write_text(md)
    print(md)
    print(f"wrote pilot/results/{ts}.md and .json  ({time.time() - t0:.0f}s total)")
