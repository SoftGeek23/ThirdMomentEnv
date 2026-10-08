"""Push the rollout agent's 1.677x recipe further on A10G (speed + quality, same harness).

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/agentref_experiment.py

Base = rollout n4SN5zX's submission (pilot/agentref/): fused QKV/gate-up backbone, Triton INT8
GEMM with dequant + residual epilogue, SmoothQuant alpha 0.45, layers 0 and 27 in BF16,
per-block compile + CUDA graphs. Variants:
  agent              as submitted
  agent-b8-attn      also quantize layers 0/27 attention (qkv, o); their MLP stays BF16
  agent-b8-all       quantize layers 0/27 entirely
  agent-b8-attn-a50  as agent-b8-attn with SmoothQuant alpha 0.5
Boundary-layer activation stats come from the pilot's train-split calibration tokens.
"""

import json
import random
import statistics
import time
from pathlib import Path

import modal

MODEL_ID, MODEL_REV = "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
CACHE, SEQ, BATCHES, PAD, GPU = "/cache", 256, (16, 32, 64), 151643, "A10G"
VARIANTS = ["agent", "agent-b8-attn", "agent-b8-all", "agent-b8-attn-a50"]

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential")
         .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2", "huggingface-hub==0.35.3",
                      "numpy==2.3.3")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_dir(str(Path(__file__).parent / "agentref"), remote_path="/agentref"))
vol = modal.Volume.from_name("thirdmoment-pilot-cache")
app = modal.App("thirdmoment-agentref-experiment", image=image)


def load():
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()


def build(variant, calib):
    import sys

    import torch
    from torch import nn

    sys.path.insert(0, "/agentref")
    import custom_int8
    import recipe as agent_recipe

    if variant == "agent":
        return agent_recipe.build(load())

    alpha = 0.5 if variant.endswith("a50") else 0.45
    static_on = (("qkv", "gu", "o") if "s-norm-o" in variant else ("qkv", "gu")) if "s-norm" in variant else ()
    clip = 0.9 if "c0.9" in variant else 1.0

    class StaticInt8Linear(custom_int8.Int8Linear):
        """Per-tensor STATIC activation scale = max_j(act_max_j / smooth_j) * clip / 127, from the
        same calibration maxima as SmoothQuant. Quantization becomes elementwise (no per-token
        reduction), so Inductor fuses it into the preceding RMSNorm."""
        def __init__(self, m, act, alpha):
            super().__init__(m, act, alpha=alpha)
            s = float((act.cuda().float() * self.inv_smooth).max()) * clip / 127.0
            self.s = s
            self.register_buffer("as_buf", torch.full((64 * 256, 1), s, device="cuda", dtype=torch.float32))

        def forward(self, x, gu=False, residual=None, swiglu=False):
            if gu or swiglu:
                return super().forward(x, gu=gu, residual=residual, swiglu=swiglu)
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            q = torch.round(x.float() * self.inv_smooth / self.s).clamp(-127, 127).to(torch.int8)
            scale = self.as_buf[:x.shape[0]]
            mm = custom_int8.matmul
            if q.shape[1] > 2048 and q.shape[0] >= 8192:
                y = mm(q, self.w, scale, self.scale, 128, 256, 128, 8, 3, gm=4, residual=residual)
            elif q.shape[1] > 2048 or self.w.shape[1] > 4096:
                y = mm(q, self.w, scale, self.scale, 256, 128, 128, 8, 3, residual=residual)
            else:
                y = mm(q, self.w, scale, self.scale, 128, 128, 64, 4, 3, residual=residual)
            return y.reshape(*shape[:-1], -1)
    # The agent's MLP calls down_proj with INT8-only kwargs when gate/up is INT8, so gu and down
    # must share a precision: keep the boundary MLPs whole in BF16.
    keep_bf16 = ({f"layers.{i}.mlp.{n}" for i in (0, 27) for n in ("gu", "down")}
                 if "b8-attn" in variant else set())
    if variant.startswith("agent-s-"):  # agent's layout: boundary layers fully BF16
        keep_bf16 = {f"layers.{i}.{n}" for i in (0, 27) for n in ("self_attn.qkv", "self_attn.o", "mlp.gu", "mlp.down")}
    stats = torch.load("/agentref/act_stats.pt", weights_only=True)

    # Backbone rewrite first, then measure activation maxima for the boundary layers' linears.
    model = load()
    m = agent_recipe.Backbone(model.model)
    need = [f"layers.{i}.{n}" for i in (0, 27) for n in ("self_attn.qkv", "self_attn.o", "mlp.gu", "mlp.down")]
    hooks = []
    for name in need:
        mod = m.get_submodule(name)
        def hook(_m, inp, _o, k=name):
            a = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).amax(0)
            stats[k] = torch.maximum(stats[k], a) if k in stats and stats[k].shape == a.shape and k in fresh else a
            fresh.add(k)
        hooks.append(mod.register_forward_hook(hook))
    fresh = set()
    with torch.no_grad():
        for ids in calib:
            m(torch.tensor([ids], device="cuda"))
    for h in hooks:
        h.remove()

    def replace(mod, prefix=""):
        for name, c in list(mod.named_children()):
            full = prefix + name
            if isinstance(c, nn.Linear):
                if full in keep_bf16:
                    continue
                cls = StaticInt8Linear if name in static_on else custom_int8.Int8Linear
                setattr(mod, name, cls(c, stats[full], alpha=alpha))
            else:
                replace(c, full + ".")

    replace(m)
    blocks = [torch.compile(agent_recipe.Block(l), dynamic=False) for l in m.layers]

    def fn(ids):
        x = m.emb(ids)
        cos, sin = m.rot(x, torch.arange(ids.shape[1], device=ids.device)[None, :])
        for b in blocks:
            x = b(x, cos, sin)
        return m.norm(x)

    graphs = {}

    def run(ids):
        b = ids.shape[0]
        if b not in graphs:
            static = ids.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    fn(static)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = fn(static)
            graphs[b] = (g, static, out)
        g, static, out = graphs[b]
        static.copy_(ids)
        g.replay()
        return out

    return run


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
    Path(f"{CACHE}/pilot/agentref-{variant}.json").write_text(json.dumps(out))
    vol.commit()
    print(f"[{variant}] RESULT {json.dumps(out)}", flush=True)
    return out


@app.local_entrypoint()
def main():
    rows = list(evaluate.map(VARIANTS, return_exceptions=True))
    print("| variant | b16 x | b32 x | b64 x | geomean x | dNLL wikitext | mbpp | gsm8k | build s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v, r in zip(VARIANTS, rows):
        if isinstance(r, Exception):
            print(f"| {v} | ERROR {str(r)[:200]} |")
            continue
        s, d = r["speedup"], r["dnll"]
        print(f"| {v} | {s['16']:.3f} | {s['32']:.3f} | {s['64']:.3f} | **{r['speedup_geomean']:.3f}** | "
              f"{d.get('wikitext', float('nan')):+.4f} | {d.get('mbpp', float('nan')):+.4f} | "
              f"{d.get('gsm8k', float('nan')):+.4f} | {r['build_warmup_s']:.0f} |")


VARIANTS2 = ["agent-b8-attn-s-norm", "agent-b8-attn-s-norm-o", "agent-b8-attn-s-norm-c0.9", "agent-s-norm"]


@app.local_entrypoint()
def static_main():
    rows = list(evaluate.map(VARIANTS2, return_exceptions=True))
    for v, r in zip(VARIANTS2, rows):
        if isinstance(r, Exception):
            print(f"| {v} | ERROR {str(r)[:300]} |")
        else:
            print(f"| {v} | {r['speedup_geomean']:.3f} | {r['speedup']} | {r['dnll']} | {r['build_warmup_s']} |")
