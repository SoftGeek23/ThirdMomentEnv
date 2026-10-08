"""Speed sweep: BF16 vs INT8 vs FP8 (W8A8, torchao) at batch 16/32/64 on one L4.

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/bench_batches.py

All three modes live in ONE container and are timed in interleaved rounds (mode order
shuffled per round), so speed ratios are free of cross-GPU and thermal/clock drift.
Each (mode, batch) is compiled once with torch.compile (static shapes, identical Inductor
config for all modes). Workload: prefill of 256-token WikiText-2 windows,
use_cache=False, logits for the last position only (same as pilot.py).
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
MODEL_REV = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")
GPU = "L4"
SEED = 1234
SEQ = 256
BATCHES = (16, 32, 64)
MODES = ("bf16", "int8", "fp8")
WARMUP = 3
ROUNDS = 5
ITERS = 8  # timed passes per (mode, batch) per round
TARGET_RE = r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$"
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2",
                 "huggingface-hub==0.35.3", "datasets==4.1.1", "numpy==2.3.3")
    .env({"HF_HOME": f"{CACHE}/hf", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
          # Persist Inductor's compile cache so re-runs skip most compile time.
          "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/inductor-cache"})
)
vol = modal.Volume.from_name("thirdmoment-pilot-cache", create_if_missing=True)
app = modal.App("thirdmoment-batch-sweep", image=image)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _inputs(n):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    ds = load_dataset(WIKITEXT[0], WIKITEXT[1], revision=WIKITEXT[2], split="validation")
    ids = tok("".join(ds["text"]), add_special_tokens=False)["input_ids"]
    starts = sorted(random.Random(SEED).sample(range(len(ids) // SEQ), n))
    return [ids[s * SEQ:(s + 1) * SEQ] for s in starts]


def _model(mode):
    import re

    import torch
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                      Int8DynamicActivationInt8WeightConfig, PerRow, quantize_)
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()
    if mode != "bf16":
        cfg = (Int8DynamicActivationInt8WeightConfig(set_inductor_config=False) if mode == "int8"
               else Float8DynamicActivationFloat8WeightConfig(granularity=PerRow(), set_inductor_config=False))
        pat = re.compile(TARGET_RE)
        quantize_(m, cfg, filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear) and bool(pat.match(fqn)))
    return m


def _category(name):
    n = name.lower()
    if "fmha" in n or "attention" in n or "flash" in n:
        return "attention"
    if "gemv" in n:
        return "gemv (lm_head, 1 token)"
    if "gemm" in n or "cutlass" in n or "_mm" in n or "xmma" in n:
        return "gemm"
    if n.startswith("triton_"):
        return "triton fused (norm/rope/act/quant)"
    return "other"


def _profile(fn):
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    dev = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    cats: dict[str, float] = {}
    for e in prof.key_averages():
        if dev(e) > 0 and not e.key.startswith(("aten::", "Torch-Compiled", "cuda", "Memcpy", "Memset")):
            c = _category(e.key)
            cats[c] = cats.get(c, 0.0) + dev(e) / 1e3
    total = sum(cats.values())
    return {"gpu_ms": total, "by_category_ms": dict(sorted(cats.items(), key=lambda kv: -kv[1]))}


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=3600)
def sweep() -> dict:
    import torch

    for k in ("recompile_limit", "cache_size_limit"):  # 3 models x 3 shapes share one code object
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, 64)
    torch.manual_seed(SEED)
    x = torch.tensor(_inputs(max(BATCHES)), device="cuda")
    out = {"gpu": torch.cuda.get_device_name(), "compile_s": {}}

    fwd, models = {}, {}
    for mode in MODES:
        models[mode] = _model(mode)
        torch.cuda.synchronize()
        cm = torch.compile(models[mode], dynamic=False)
        for b in BATCHES:
            f = (lambda cm=cm, b=b: cm(input_ids=x[:b], use_cache=False, logits_to_keep=1).logits)
            t0 = time.time()
            with torch.no_grad():
                for _ in range(WARMUP):
                    f()
            torch.cuda.synchronize()
            out["compile_s"][f"{mode}/{b}"] = round(time.time() - t0, 1)
            _log(f"compiled {mode} b={b} in {out['compile_s'][f'{mode}/{b}']}s")
            fwd[(mode, b)] = f
    out["weights_gb_all_three_models"] = torch.cuda.memory_allocated() / 2**30

    # Interleaved timing rounds.
    times = {k: [] for k in fwd}
    per_round = {k: [] for k in fwd}
    rng = random.Random(SEED)
    with torch.no_grad():
        for r in range(ROUNDS):
            for b in BATCHES:
                order = list(MODES)
                rng.shuffle(order)
                for mode in order:
                    f, ts = fwd[(mode, b)], []
                    for _ in range(ITERS):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        f()
                        e1.record()
                        torch.cuda.synchronize()
                        ts.append(e0.elapsed_time(e1))
                    times[(mode, b)] += ts
                    per_round[(mode, b)].append(statistics.median(ts))
            _log(f"round {r + 1}/{ROUNDS} done")

        prof, peak = {}, {}
        for (mode, b), f in fwd.items():
            torch.cuda.reset_peak_memory_stats()
            f()
            torch.cuda.synchronize()
            peak[f"{mode}/{b}"] = torch.cuda.max_memory_allocated() / 2**30
            prof[f"{mode}/{b}"] = _profile(f)

    res = {}
    for (mode, b), ts in times.items():
        ts = sorted(ts)
        base_rounds = per_round[("bf16", b)]
        ratios = [br / mr for br, mr in zip(base_rounds, per_round[(mode, b)])]
        res[f"{mode}/{b}"] = {
            "median_ms": statistics.median(ts), "p10_ms": ts[len(ts) // 10], "p90_ms": ts[(9 * len(ts)) // 10],
            "tok_per_s": b * SEQ / (statistics.median(ts) / 1e3),
            "speedup_vs_bf16_median": statistics.median(ratios),
            "speedup_vs_bf16_range": [min(ratios), max(ratios)],
        }
    out.update({"results": res, "profile": prof, "peak_mem_gb_incl_all_models": peak})
    return out


@app.local_entrypoint()
def main():
    t0 = time.time()
    out = sweep.remote()
    r = out["results"]
    lines = [f"# Batch sweep on {out['gpu']} (prefill, {SEQ} tokens/seq, interleaved x{ROUNDS} rounds)", "",
             "| batch | mode | latency ms [p10-p90] | tok/s | speedup vs BF16 [min-max over rounds] | "
             "GPU ms: gemm / attn / triton-fused / lm_head / other |",
             "|---|---|---|---|---|---|"]
    for b in BATCHES:
        for mode in MODES:
            k = f"{mode}/{b}"
            x, p = r[k], out["profile"][k]["by_category_ms"]
            g = lambda c: p.get(c, 0.0)  # noqa: E731
            lines.append(
                f"| {b} | {mode} | {x['median_ms']:.1f} [{x['p10_ms']:.1f}-{x['p90_ms']:.1f}] | {x['tok_per_s']:.0f} | "
                f"{x['speedup_vs_bf16_median']:.3f} [{x['speedup_vs_bf16_range'][0]:.3f}-"
                f"{x['speedup_vs_bf16_range'][1]:.3f}] | {g('gemm'):.1f} / {g('attention'):.1f} / "
                f"{g('triton fused (norm/rope/act/quant)'):.1f} / {g('gemv (lm_head, 1 token)'):.1f} / {g('other'):.1f} |")
    lines += ["", "INT8 vs FP8 (same container, same rounds):"]
    for b in BATCHES:
        i8, f8 = r[f"int8/{b}"]["median_ms"], r[f"fp8/{b}"]["median_ms"]
        lines.append(f"- batch {b}: FP8 is {i8 / f8:.3f}x the speed of INT8 ({f8:.1f} ms vs {i8:.1f} ms)")
    md = "\n".join(lines) + "\n"
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    (outdir / f"sweep-{ts}.json").write_text(json.dumps(out, indent=1))
    (outdir / f"sweep-{ts}.md").write_text(md)
    print(md)
    print(f"wrote pilot/results/sweep-{ts}.md ({time.time() - t0:.0f}s)")
