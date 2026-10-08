"""Reference search on top of rollout qaamH3h's 1.765x recipe (A10G). See agentref2/recipe_plus.py.

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/agentref2_experiment.py
"""

import json
import random
import statistics
import time
from pathlib import Path

import modal

MODEL_ID, MODEL_REV = "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
CACHE, SEQ, BATCHES, PAD, GPU = "/cache", 256, (16, 32, 64), 151643, "A10G"
VARIANTS = ["q", "q-res", "q-res-tiles", "q-res-tiles-static"]

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential")
         .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2", "huggingface-hub==0.35.3",
                      "numpy==2.3.3")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(str(Path(__file__).parent / "agentref2"), remote_path="/agentref2"))
vol = modal.Volume.from_name("thirdmoment-pilot-cache")
app = modal.App("thirdmoment-agentref2-experiment", image=image)


def load():
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()


def build(variant, calib):
    import sys

    import torch

    for k in ("recompile_limit", "cache_size_limit"):  # same as the grader's harness.setup_torch()
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, 64)
    sys.path.insert(0, "/agentref2")
    if variant == "q":
        import recipe
        return recipe.build(load())
    import recipe_plus as rp
    rp.RESIDUAL = "res" in variant
    rp.TILES = "tiles" in variant
    rp.STATIC = "static" in variant
    return rp.build(load(), "/agentref2/stats.pt")


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=3600)
def evaluate(variant: str) -> dict:
    import torch

    data = json.loads(Path(f"{CACHE}/pilot/data-1234.json").read_text())
    calib = [it["ids"][:SEQ] for it in data["calib"]]
    items = []
    for it in data["eval"]:
        if it["slice"] == "wikitext":
            items.append({"slice": "wikitext", "ids": it["ids"][:SEQ], "start": SEQ // 2})
        elif it["start"] + 8 <= SEQ:
            items.append(it)
    x = torch.tensor([w[:SEQ] for w in data["bench"]] * 4, device="cuda")[:max(BATCHES)]

    base_model = load()
    head = base_model.lm_head.weight.detach().clone()
    bcm = torch.compile(base_model.model, dynamic=False)
    base = lambda ids: bcm(input_ids=ids, use_cache=False).last_hidden_state  # noqa: E731
    t0 = time.time()
    fn = build(variant, calib)
    out = {"variant": variant}
    with torch.no_grad():
        for b in BATCHES:
            base(x[:b])
            fn(x[:b])
        out["build_warmup_s"] = round(time.time() - t0, 1)

        def nll(f):
            res = []
            for i in range(0, len(items), 16):
                chunk = items[i:i + 16]
                ids = torch.full((16, SEQ), PAD, dtype=torch.long)
                for j, it in enumerate(chunk):
                    ids[j, :min(len(it["ids"]), SEQ)] = torch.tensor(it["ids"][:SEQ])
                h = f(ids.cuda()).clone()
                for j, it in enumerate(chunk):
                    n, s = min(len(it["ids"]), SEQ), it["start"]
                    lp = torch.log_softmax((h[j, s - 1:n - 1].to(torch.bfloat16) @ head.T).float(), -1)
                    tgt = torch.tensor(it["ids"][s:n], device="cuda")
                    res.append(float(-lp.gather(1, tgt[:, None]).mean()))
            return res

        bn, sn = nll(base), nll(fn)
        by = {}
        for it, b, s in zip(items, bn, sn):
            by.setdefault(it["slice"], []).append(s - b)
        out["dnll"] = {k: statistics.fmean(v) for k, v in by.items()}

        rng, per = random.Random(0), {(w, b): [] for w in "bs" for b in BATCHES}
        for _ in range(5):
            for b in BATCHES:
                order = ["b", "s"]
                rng.shuffle(order)
                for who in order:
                    f, ts = (base if who == "b" else fn), []
                    for _ in range(6):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record(); f(x[:b]); e1.record(); torch.cuda.synchronize()
                        ts.append(e0.elapsed_time(e1))
                    per[(who, b)].append(statistics.median(ts))
        out["speedup"] = {str(b): statistics.median(bt / st for bt, st in zip(per[("b", b)], per[("s", b)]))
                          for b in BATCHES}
    gm = 1.0
    for v in out["speedup"].values():
        gm *= v
    out["speedup_geomean"] = gm ** (1 / len(BATCHES))
    Path(f"{CACHE}/pilot/agentref2-{variant}.json").write_text(json.dumps(out))
    vol.commit()
    print(f"[{variant}] RESULT {json.dumps(out)}", flush=True)
    return out



@app.local_entrypoint()
def main():
    rows = list(evaluate.map(VARIANTS, return_exceptions=True))
    for v, r in zip(VARIANTS, rows):
        if isinstance(r, Exception):
            print(f"| {v} | ERROR {str(r)[:300]} |")
        else:
            print(f"| {v} | {r['speedup_geomean']:.3f} | {r['speedup']} | {r['dnll']} | {r['build_warmup_s']} |")
