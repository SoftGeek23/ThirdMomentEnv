"""Which INT8 recipe should be the reference? Speed (vs compiled BF16, same GPU) + quality, on A10G.

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/ref_experiment.py

One container per candidate. Each container times its candidate against compiled BF16,
interleaved, at batch 16/32/64 x 256 tokens, then measures dNLL on the pilot's held-out data
(truncated to 256 tokens) through the candidate's fn at batch 16.

A10G = Ampere sm_86: INT8 tensor cores, no FP8. Candidates:
  hf-int8                  torchao INT8 W8A8 on the HF model + torch.compile   (the "obvious" answer)
  fused-bf16               custom forward: merged QKV and gate/up linears, causal flash SDPA (no quant)
  fused-int8               custom forward + torchao INT8 W8A8 on the merged linears
  fused-int8-sq{a}         + SmoothQuant folded into norms/weights (alpha a), calibrated on train data
  fused-int8-sq0.5-cg      + CUDA graphs (torch.compile mode="reduce-overhead")
  fused-int8-sq0.5-ma      + Inductor max-autotune (Triton GEMM templates with fused epilogues)
"""

import json
import random
import statistics
import time
from pathlib import Path

import modal

MODEL_ID, MODEL_REV = "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
CACHE, SEQ, BATCHES, PAD = "/cache", 256, (16, 32, 64), 151643
VARIANTS = ["hf-int8", "fused-bf16", "fused-int8", "fused-int8-sq0.5", "fused-int8-sq0.8",
            "fused-int8-sq0.5-cg", "fused-int8-sq0.5-ma"]
GPU = "A10G"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2", "huggingface-hub==0.35.3",
                      "numpy==2.3.3")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_OFFLINE": "1",
               "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/inductor-cache"}))
vol = modal.Volume.from_name("thirdmoment-pilot-cache")
app = modal.App("thirdmoment-ref-experiment", image=image)


# ------------------------------------------------------------------ recipe building blocks
def load():
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()


def quant_config(fmt):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    assert fmt == "int8"
    return Int8DynamicActivationInt8WeightConfig(set_inductor_config=False)


def smooth(model, calib, alpha):
    """SmoothQuant folded exactly into RMSNorm / up_proj weights (no runtime cost)."""
    import torch

    stats, hooks = {}, []
    for i, L in enumerate(model.model.layers):
        for key, mod in (("qkv", L.self_attn.q_proj), ("gu", L.mlp.gate_proj), ("down", L.mlp.down_proj)):
            def hook(_m, inp, _o, k=(i, key)):
                a = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).amax(0)
                stats[k] = torch.maximum(stats[k], a) if k in stats else a
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for ids in calib:
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()

    def scale(x, ws):
        w = torch.stack([w.abs().float().amax(0) for w in ws]).amax(0)
        return (x.clamp(min=1e-5) ** alpha / w.clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)

    def cols(ws, s):
        for w in ws:
            w.copy_((w.float() * s[None, :]).to(w.dtype))

    with torch.no_grad():
        for i, L in enumerate(model.model.layers):
            a, m = L.self_attn, L.mlp
            s = scale(stats[(i, "qkv")], [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            L.input_layernorm.weight.copy_((L.input_layernorm.weight.float() / s).to(torch.bfloat16))
            cols([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], s)
            s = scale(stats[(i, "gu")], [m.gate_proj.weight, m.up_proj.weight])
            L.post_attention_layernorm.weight.copy_((L.post_attention_layernorm.weight.float() / s).to(torch.bfloat16))
            cols([m.gate_proj.weight, m.up_proj.weight], s)
            s = scale(stats[(i, "down")], [m.down_proj.weight])
            m.up_proj.weight.copy_((m.up_proj.weight.float() / s[:, None]).to(torch.bfloat16))
            cols([m.down_proj.weight], s)


def fused_backbone(model, fmt=None):
    """Custom Qwen3 forward: merged QKV / gate-up linears, causal SDPA (flash), no mask."""
    import torch
    import torch.nn.functional as F
    from torchao.quantization import quantize_
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    cfg = model.config
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    inter = cfg.intermediate_size

    class Layer(torch.nn.Module):
        def __init__(self, L):
            super().__init__()
            a, m = L.self_attn, L.mlp
            self.ln1, self.ln2, self.qn, self.kn = L.input_layernorm, L.post_attention_layernorm, a.q_norm, a.k_norm
            self.qkv = torch.nn.Linear(cfg.hidden_size, (nh + 2 * nkv) * hd, bias=False, dtype=torch.bfloat16,
                                       device="cuda")
            self.qkv.weight.data = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach()
            self.o = a.o_proj
            self.gu = torch.nn.Linear(cfg.hidden_size, 2 * inter, bias=False, dtype=torch.bfloat16, device="cuda")
            self.gu.weight.data = torch.cat([m.gate_proj.weight, m.up_proj.weight]).detach()
            self.down = m.down_proj

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.norm, self.rope = model.model.embed_tokens, model.model.norm, model.model.rotary_emb
            self.layers = torch.nn.ModuleList(Layer(L) for L in model.model.layers)

        def forward(self, ids):
            B, T = ids.shape
            h = self.emb(ids)
            cos, sin = self.rope(h, torch.arange(T, device=ids.device)[None])
            for L in self.layers:
                x = L.ln1(h)
                q, k, v = L.qkv(x).split([nh * hd, nkv * hd, nkv * hd], dim=-1)
                q = L.qn(q.view(B, T, nh, hd)).transpose(1, 2)
                k = L.kn(k.view(B, T, nkv, hd)).transpose(1, 2)
                v = v.view(B, T, nkv, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k, v = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                h = h + L.o(o.transpose(1, 2).reshape(B, T, nh * hd))
                g, u = L.gu(L.ln2(h)).split(inter, dim=-1)
                h = h + L.down(F.silu(g) * u)
            return self.norm(h)

    bb = Backbone().eval()
    for L in model.model.layers:  # free the now-duplicated unmerged weights
        L.self_attn.q_proj = L.self_attn.k_proj = L.self_attn.v_proj = None
        L.mlp.gate_proj = L.mlp.up_proj = None
    torch.cuda.empty_cache()
    if fmt:
        quantize_(bb, quant_config(fmt), filter_fn=lambda m, f: isinstance(m, torch.nn.Linear) and f.startswith("layers."))
    return bb


def build(variant, calib):
    """Returns fn(ids[B,256]) -> final hidden states."""
    import re

    import torch
    from torchao.quantization import quantize_

    model = load()
    if variant.startswith("hf-"):
        pat = re.compile(r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$")
        quantize_(model, quant_config(variant[3:]),
                  filter_fn=lambda m, f: isinstance(m, torch.nn.Linear) and bool(pat.match(f)))
        cm = torch.compile(model.model, dynamic=False)
        return lambda ids: cm(input_ids=ids, use_cache=False).last_hidden_state
    parts = variant.split("-")  # fused-<fmt>[-sq<a>][-cg|-ma]
    fmt = None if parts[1] == "bf16" else parts[1]
    sq = next((float(p[2:]) for p in parts if p.startswith("sq")), None)
    if sq is not None:
        smooth(model, calib, sq)
    bb = fused_backbone(model, fmt)
    mode = "reduce-overhead" if "cg" in parts else "max-autotune-no-cudagraphs" if "ma" in parts else None
    cm = torch.compile(bb, dynamic=False, mode=mode)
    if "cg" in parts:
        return lambda ids: cm(ids).clone()  # cudagraph outputs are overwritten by the next replay
    return cm


def profile(f):
    """GPU time (ms) of one call, split into gemm / attention / triton-fused / other kernels."""
    import torch
    from torch.profiler import ProfilerActivity, profile as tprof

    with tprof(activities=[ProfilerActivity.CUDA]) as prof:
        f()
        torch.cuda.synchronize()
    dev = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    cats = {}
    for e in prof.key_averages():
        n = e.key.lower()
        if dev(e) <= 0 or n.startswith(("aten::", "torch-compiled", "cuda", "memcpy", "memset")):
            continue
        c = ("attention" if ("fmha" in n or "flash" in n or "attention" in n) else
             "gemm" if ("gemm" in n or "cutlass" in n or "_mm" in n or "xmma" in n or "triton_tem" in n) else
             "triton" if n.startswith("triton_") else "other")
        cats[c] = cats.get(c, 0.0) + dev(e) / 1e3
    return cats


# --------------------------------------------------------------------------------- run
@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=3600)
def evaluate(variant: str) -> dict:
    import torch

    for k in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, 64)
    data = json.loads(Path(f"{CACHE}/pilot/data-1234.json").read_text())
    calib = [it["ids"][:SEQ] for it in data["calib"]]
    items = [it for it in data["eval"] if it["start"] + 8 <= SEQ]
    x = torch.tensor([w[:SEQ] for w in data["bench"]] * 4, device="cuda")[:max(BATCHES)]

    base_model = load()
    head = base_model.lm_head.weight.detach().clone()
    bcm = torch.compile(base_model.model, dynamic=False)
    base = lambda ids: bcm(input_ids=ids, use_cache=False).last_hidden_state  # noqa: E731
    t0 = time.time()
    fn = build(variant, calib)
    out = {"variant": variant, "gpu": torch.cuda.get_device_name()}
    with torch.no_grad():
        for b in BATCHES:
            base(x[:b]); fn(x[:b])
        out["build_warmup_s"] = round(time.time() - t0, 1)

        def nll(f):
            res = []
            for i in range(0, len(items) - len(items) % 16, 16):
                chunk = items[i:i + 16]
                ids = torch.full((16, SEQ), PAD, dtype=torch.long)
                for j, it in enumerate(chunk):
                    ids[j, :min(len(it["ids"]), SEQ)] = torch.tensor(it["ids"][:SEQ])
                h = f(ids.cuda())
                for j, it in enumerate(chunk):
                    n, s = min(len(it["ids"]), SEQ), it["start"]
                    lp = torch.log_softmax((h[j, s - 1:n - 1].to(torch.bfloat16) @ head.T).float(), -1)
                    res.append((it["slice"], float(-lp.gather(1, torch.tensor(it["ids"][s:n], device="cuda")[:, None]).mean())))
            return res

        bn, sn = nll(base), nll(fn)
        by = {}
        for (s, b), (_, v) in zip(bn, sn):
            by.setdefault(s, []).append(v - b)
        out["dnll"] = {s: statistics.fmean(d) for s, d in by.items()}

        rng, per = random.Random(0), {("b", b): [] for b in BATCHES} | {("s", b): [] for b in BATCHES}
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
        out["speedup"] = {b: statistics.median(bt / st for bt, st in zip(per[("b", b)], per[("s", b)]))
                          for b in BATCHES}
        out["ms"] = {b: statistics.median(per[("s", b)]) for b in BATCHES}
        out["bf16_ms"] = {b: statistics.median(per[("b", b)]) for b in BATCHES}
        out["profile_b64"] = {"bf16": profile(lambda: base(x[:64])), "variant": profile(lambda: fn(x[:64]))}
    gm = 1.0
    for v in out["speedup"].values():
        gm *= v
    out["speedup_geomean"] = gm ** (1 / len(BATCHES))
    print(f"[{variant}] geomean {out['speedup_geomean']:.3f}x dnll {out['dnll']}", flush=True)
    return out


@app.local_entrypoint()
def main():
    t0 = time.time()
    rows = list(evaluate.map(VARIANTS, return_exceptions=True))
    lines = ["| variant | b16 x | b32 x | b64 x | geomean x | dNLL wikitext | mbpp | gsm8k | build s | "
             "b64 GPU ms gemm/attn/triton/other (bf16 -> variant) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in zip(VARIANTS, rows):
        if isinstance(r, Exception):
            lines.append(f"| {v} | ERROR: {str(r)[:150]} |")
            continue
        s, d = r["speedup"], r["dnll"]
        lines.append(f"| {v} | {s[16]:.3f} | {s[32]:.3f} | {s[64]:.3f} | **{r['speedup_geomean']:.3f}** | "
                     f"{d['wikitext']:+.4f} | {d['mbpp']:+.4f} | {d['gsm8k']:+.4f} | {r['build_warmup_s']:.0f} | "
                     + " -> ".join("/".join(f"{p.get(c, 0):.0f}" for c in ("gemm", "attention", "triton", "other"))
                                   for p in (r["profile_b64"]["bf16"], r["profile_b64"]["variant"])) + " |")
    md = "\n".join(lines) + "\n"
    out = Path(__file__).parent / "results"
    ts = time.strftime("%Y%m%d-%H%M%S")
    (out / f"ref-{ts}.json").write_text(json.dumps([r if not isinstance(r, Exception) else repr(r) for r in rows],
                                                   indent=1, default=str))
    (out / f"ref-{ts}.md").write_text(md)
    print(md)
    print(f"wrote pilot/results/ref-{ts}.md ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------- quality-only re-run (eager)
QUALITY_VARIANTS = ["hf-int8", "fused-int8", "fused-int8-sq0.5", "fused-int8-sq0.65", "fused-int8-sq0.8"]


def _eval_items(data):
    """All three sources, fitted to 256 tokens. WikiText windows (512 tokens, scored from 256)
    become their first 256 tokens scored from 128."""
    out = []
    for it in data["eval"]:
        if it["slice"] == "wikitext":
            out.append({"slice": "wikitext", "ids": it["ids"][:SEQ], "start": SEQ // 2})
        elif it["start"] + 8 <= SEQ:
            out.append(it)
    return out


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=1800)
def quality(variant: str) -> dict:
    import torch

    data = json.loads(Path(f"{CACHE}/pilot/data-1234.json").read_text())
    calib = [it["ids"][:SEQ] for it in data["calib"]]
    items = _eval_items(data)
    base_model = load()
    head = base_model.lm_head.weight.detach().clone()
    base = lambda ids: base_model.model(input_ids=ids, use_cache=False).last_hidden_state  # noqa: E731

    model = load()
    if variant.startswith("hf-"):
        import re

        from torchao.quantization import quantize_
        pat = re.compile(r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$")
        quantize_(model, quant_config("int8"), filter_fn=lambda m, f: isinstance(m, torch.nn.Linear) and bool(pat.match(f)))
        fn = lambda ids: model.model(input_ids=ids, use_cache=False).last_hidden_state  # noqa: E731
    else:
        parts = variant.split("-")
        sq = next((float(p[2:]) for p in parts if p.startswith("sq")), None)
        if sq is not None:
            smooth(model, calib, sq)
        fn = fused_backbone(model, None if parts[1] == "bf16" else parts[1])

    def nll(f):
        res = []
        with torch.no_grad():
            for i in range(0, len(items), 16):
                chunk = items[i:i + 16]
                ids = torch.full((len(chunk), SEQ), PAD, dtype=torch.long)
                for j, it in enumerate(chunk):
                    ids[j, :min(len(it["ids"]), SEQ)] = torch.tensor(it["ids"][:SEQ])
                h = f(ids.cuda())
                for j, it in enumerate(chunk):
                    n, s = min(len(it["ids"]), SEQ), it["start"]
                    lp = torch.log_softmax((h[j, s - 1:n - 1].to(torch.bfloat16) @ head.T).float(), -1)
                    tgt = torch.tensor(it["ids"][s:n], device="cuda")
                    res.append(float(-lp.gather(1, tgt[:, None]).mean()))
        return res

    b, s = nll(base), nll(fn)
    by = {}
    for it, x, y in zip(items, b, s):
        by.setdefault(it["slice"], []).append(y - x)
    out = {"variant": variant, "n": {k: len(v) for k, v in by.items()},
           "dnll": {k: statistics.fmean(v) for k, v in by.items()}}
    Path(f"{CACHE}/pilot/quality-{variant}.json").write_text(json.dumps(out))
    vol.commit()
    print(f"[{variant}] QUALITY {json.dumps(out)}", flush=True)
    return out


@app.local_entrypoint()
def quality_main():
    rows = list(quality.map(QUALITY_VARIANTS, return_exceptions=True))
    print("| variant | dNLL wikitext | mbpp | gsm8k | n |\n|---|---|---|---|---|")
    for v, r in zip(QUALITY_VARIANTS, rows):
        if isinstance(r, Exception):
            print(f"| {v} | ERROR {str(r)[:120]} |")
        else:
            d = r["dnll"]
            print(f"| {v} | {d.get('wikitext', float('nan')):+.4f} | {d.get('mbpp', float('nan')):+.4f} | "
                  f"{d.get('gsm8k', float('nan')):+.4f} | {r['n']} |")
