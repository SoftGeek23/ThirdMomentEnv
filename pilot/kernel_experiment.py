"""Custom-kernel reference candidates on A10G: can explicit INT8 + a fused Triton RMSNorm+quant
kernel beat the torchao reference?

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/kernel_experiment.py

All variants share: SmoothQuant alpha 0.5 (calibrated on the pilot's train-split tokens), merged
QKV / gate-up linears, causal SDPA. They differ only in how the INT8 linears are executed:
  torchao-ma   torchao Int8DynamicActivationInt8Weight, compile max-autotune   (current reference)
  manual-ma    explicit int8: per-token quant in plain torch ops -> torch._int_mm -> dequant,
               so Inductor can fuse norm->quant and the dequant epilogue; compile max-autotune
  triton-ma    manual + Triton kernel fusing RMSNorm + per-token INT8 quant (ln1, ln2); max-autotune
  triton       same, default compile
Each container: speed vs compiled BF16 on the same GPU (interleaved, batch 16/32/64) + dNLL
(WikiText/MBPP/GSM8K, 256-token windows) + a GPU-time breakdown at batch 64.
"""

import json
import random
import statistics
import time
from pathlib import Path

import modal

MODEL_ID, MODEL_REV = "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
CACHE, SEQ, BATCHES, PAD, GPU = "/cache", 256, (16, 32, 64), 151643, "A10G"
VARIANTS = ["torchao-ma", "manual-ma", "triton-ma", "triton"]

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("build-essential")
         .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2", "huggingface-hub==0.35.3",
                      "numpy==2.3.3")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_OFFLINE": "1"}))
vol = modal.Volume.from_name("thirdmoment-pilot-cache")
app = modal.App("thirdmoment-kernel-experiment", image=image)


# ============================================================ Triton: RMSNorm + INT8 quant
def define_kernels():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_quant(X, W, Q, S, stride_x, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        # match HF Qwen3RMSNorm rounding: normalize in fp32, cast to bf16, multiply by bf16 weight
        xn = (x * tl.rsqrt(var + eps)).to(tl.bfloat16)
        w = tl.load(W + cols, mask=mask, other=0.0)
        y = (w * xn).to(tl.float32)
        amax = tl.max(tl.abs(y), axis=0)
        scale = tl.maximum(amax, 1e-5) / 127.0
        v = y / scale
        q = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(Q + row * N + cols, q.to(tl.int8), mask=mask)
        tl.store(S + row, scale)

    @torch.library.custom_op("tm::rmsnorm_quant", mutates_args=())
    def rmsnorm_quant(x: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x2.shape
        q = torch.empty((M, N), dtype=torch.int8, device=x.device)
        s = torch.empty((M,), dtype=torch.float32, device=x.device)
        _rmsnorm_quant[(M,)](x2, w, q, s, x2.stride(0), N, eps, BLOCK=triton.next_power_of_2(N), num_warps=8)
        return q, s

    @rmsnorm_quant.register_fake
    def _(x, w, eps):
        M = x.numel() // x.shape[-1]
        return x.new_empty((M, x.shape[-1]), dtype=torch.int8), x.new_empty((M,), dtype=torch.float32)

    return rmsnorm_quant


# ============================================================ model building blocks
def load():
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()


def smooth(model, calib, alpha=0.5):
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


def build(variant, calib):
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    model = load()
    smooth(model, calib, 0.5)
    cfg = model.config
    nh, nkv, hd, inter, H = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                             cfg.intermediate_size, cfg.hidden_size)
    use_triton = variant.startswith("triton")
    torchao = variant.startswith("torchao")
    norm_quant = define_kernels() if use_triton else None

    class QLinear(torch.nn.Module):
        """W8A8: int8 weight (per-output-channel), int8 activation (per-token), int32 accumulate."""
        def __init__(self, w):
            super().__init__()
            wf = w.detach().float()
            ws = wf.abs().amax(1).clamp(min=1e-5) / 127.0
            self.register_buffer("wq", (wf / ws[:, None]).round().clamp(-127, 127).to(torch.int8))
            self.register_buffer("ws", ws)

        def from_q(self, q, s):
            acc = torch._int_mm(q, self.wq.t())
            return (acc.float() * s[:, None] * self.ws[None, :]).to(torch.bfloat16)

        def forward(self, x):  # x [M, K] bf16
            xf = x.float()
            s = xf.abs().amax(-1).clamp(min=1e-5) / 127.0
            q = (xf / s[:, None]).round().clamp(-127, 127).to(torch.int8)
            return self.from_q(q, s)

    def lin(ws):
        w = torch.cat(ws) if len(ws) > 1 else ws[0]
        if torchao:
            m = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.bfloat16, device="cuda")
            m.weight.data = w.detach().clone()
            return m
        return QLinear(w)

    class Layer(torch.nn.Module):
        def __init__(self, L):
            super().__init__()
            a, m = L.self_attn, L.mlp
            self.ln1, self.ln2, self.qn, self.kn = L.input_layernorm, L.post_attention_layernorm, a.q_norm, a.k_norm
            self.qkv = lin([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            self.o = lin([a.o_proj.weight])
            self.gu = lin([m.gate_proj.weight, m.up_proj.weight])
            self.down = lin([m.down_proj.weight])

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.norm, self.rope = model.model.embed_tokens, model.model.norm, model.model.rotary_emb
            self.layers = torch.nn.ModuleList(Layer(L) for L in model.model.layers)

        def _norm_lin(self, ln, linear, h):
            if use_triton:
                q, s = norm_quant(h, ln.weight, ln.variance_epsilon)
                return linear.from_q(q, s)
            x = ln(h).reshape(-1, H)
            return linear(x)

        def forward(self, ids):
            B, T = ids.shape
            h = self.emb(ids)
            cos, sin = self.rope(h, torch.arange(T, device=ids.device)[None])
            for L in self.layers:
                q, k, v = self._norm_lin(L.ln1, L.qkv, h).view(B, T, -1).split([nh * hd, nkv * hd, nkv * hd], -1)
                q = L.qn(q.view(B, T, nh, hd)).transpose(1, 2)
                k = L.kn(k.view(B, T, nkv, hd)).transpose(1, 2)
                v = v.view(B, T, nkv, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k, v = k.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                h = h + L.o(o.transpose(1, 2).reshape(B * T, nh * hd)).view(B, T, H)
                g, u = self._norm_lin(L.ln2, L.gu, h).view(B, T, -1).split(inter, -1)
                h = h + L.down((F.silu(g) * u).reshape(B * T, inter)).view(B, T, H)
            return self.norm(h)

    bb = Backbone().eval()
    del model
    torch.cuda.empty_cache()
    if torchao:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
        quantize_(bb, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False),
                  filter_fn=lambda m, f: isinstance(m, torch.nn.Linear) and f.startswith("layers."))
    mode = "max-autotune-no-cudagraphs" if variant.endswith("-ma") else None
    return torch.compile(bb, dynamic=False, mode=mode)


def profile(f):
    import torch
    from torch.profiler import ProfilerActivity, profile as tprof

    with tprof(activities=[ProfilerActivity.CUDA]) as prof:
        f()
        torch.cuda.synchronize()
    dev = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    cats = {}
    for e in prof.key_averages():
        n = e.key.lower()
        if dev(e) <= 0 or n.startswith(("aten::", "torch-compiled", "cuda", "memcpy", "memset", "tm::")):
            continue
        c = ("attention" if ("fmha" in n or "flash" in n or "attention" in n) else
             "gemm" if ("gemm" in n or "cutlass" in n or "_mm" in n or "xmma" in n or "triton_tem" in n) else
             "rmsnorm_quant" if "rmsnorm_quant" in n else
             "triton" if n.startswith("triton_") else "other")
        cats[c] = cats.get(c, 0.0) + dev(e) / 1e3
    return cats


@app.function(gpu=GPU, cpu=8, memory=32768, volumes={CACHE: vol}, timeout=3600)
def evaluate(variant: str) -> dict:
    import torch

    for k in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, 64)
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
    out = {"variant": variant, "gpu": torch.cuda.get_device_name()}
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
                h = f(ids.cuda())
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
        out["ms"] = {str(b): statistics.median(per[("s", b)]) for b in BATCHES}
        out["profile_b64"] = {"bf16": profile(lambda: base(x[:64])), "variant": profile(lambda: fn(x[:64]))}
    gm = 1.0
    for v in out["speedup"].values():
        gm *= v
    out["speedup_geomean"] = gm ** (1 / len(BATCHES))
    Path(f"{CACHE}/pilot/kernel-{variant}.json").write_text(json.dumps(out))
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
    for v, r in zip(VARIANTS, rows):
        if not isinstance(r, Exception):
            print(v, "b64 GPU ms bf16:", {k: round(x) for k, x in r["profile_b64"]["bf16"].items()},
                  "-> variant:", {k: round(x) for k, x in r["profile_b64"]["variant"].items()})
