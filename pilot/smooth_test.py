"""Does SmoothQuant-style smoothing fix INT8 W8A8 quality on Qwen3-1.7B-Base? (quality only)

Run:  ~/.local/share/uv/tools/harbor/bin/modal run pilot/smooth_test.py

Reuses the pilot's cached model + token sets (Volume thirdmoment-pilot-cache):
calibration = pilot "calib" (train splits), evaluation = pilot "eval" (held-out test splits).
Smoothing is folded exactly into the weights:
  qkv / gate-up : RMSNorm.weight /= s,   proj.weight[:, j] *= s_j
  down_proj     : up_proj.weight[i, :] /= s_i,   down_proj.weight[:, i] *= s_i
  s_j = max|X_j|^a / max|W_j|^(1-a)
"""

import json
import time
from pathlib import Path

import modal

MODEL_ID, MODEL_REV = "Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
CACHE = "/cache"
ALPHAS = (0.5, 0.65, 0.8)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "torchao==0.13.0", "transformers==4.56.2", "huggingface-hub==0.35.3",
                      "numpy==2.3.3")
         .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_OFFLINE": "1"}))
vol = modal.Volume.from_name("thirdmoment-pilot-cache")
app = modal.App("thirdmoment-smooth-test", image=image)


@app.function(gpu="L40S", cpu=8, memory=32768, volumes={CACHE: vol}, timeout=3600)
def run() -> dict:
    import re

    import torch
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    from transformers import AutoModelForCausalLM

    data = json.loads(Path(f"{CACHE}/pilot/data-1234.json").read_text())
    target = re.compile(r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$")

    def load():
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").cuda().eval()

    @torch.no_grad()
    def nll(model):
        out = {}
        for it in data["eval"]:
            ids = torch.tensor([it["ids"]], device="cuda")
            s = it["start"]
            h = model.model(input_ids=ids).last_hidden_state[0, s - 1:-1]
            lp = torch.log_softmax(model.lm_head(h).float(), -1)
            out[(it["slice"], it["idx"])] = float(-lp.gather(1, ids[0, s:, None]).mean())
        return out

    @torch.no_grad()
    def act_absmax(model):
        """Per-channel max|x| of the inputs to q_proj (=k,v), gate_proj (=up) and down_proj."""
        stats, hooks = {}, []
        for i, layer in enumerate(model.model.layers):
            for key, mod in (("qkv", layer.self_attn.q_proj), ("gu", layer.mlp.gate_proj),
                             ("down", layer.mlp.down_proj)):
                def hook(_m, inp, _o, k=(i, key)):
                    a = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).amax(0)
                    stats[k] = torch.maximum(stats[k], a) if k in stats else a
                hooks.append(mod.register_forward_hook(hook))
        for it in data["calib"]:
            model(input_ids=torch.tensor([it["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
        for h in hooks:
            h.remove()
        return stats

    @torch.no_grad()
    def smooth(model, stats, alpha):
        def scale(xmax, ws):
            wmax = torch.stack([w.abs().float().amax(0) for w in ws]).amax(0)
            return (xmax.clamp(min=1e-5) ** alpha / wmax.clamp(min=1e-5) ** (1 - alpha)).clamp(min=1e-5)

        def fold_cols(ws, s):
            for w in ws:
                w.copy_((w.float() * s[None, :]).to(w.dtype))

        for i, L in enumerate(model.model.layers):
            a, m = L.self_attn, L.mlp
            s = scale(stats[(i, "qkv")], [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            L.input_layernorm.weight.copy_((L.input_layernorm.weight.float() / s).to(torch.bfloat16))
            fold_cols([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], s)
            s = scale(stats[(i, "gu")], [m.gate_proj.weight, m.up_proj.weight])
            L.post_attention_layernorm.weight.copy_((L.post_attention_layernorm.weight.float() / s).to(torch.bfloat16))
            fold_cols([m.gate_proj.weight, m.up_proj.weight], s)
            s = scale(stats[(i, "down")], [m.down_proj.weight])
            m.up_proj.weight.copy_((m.up_proj.weight.float() / s[:, None]).to(torch.bfloat16))
            fold_cols([m.down_proj.weight], s)

    def int8(model):
        quantize_(model, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False),
                  filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear) and bool(target.match(fqn)))

    t0 = time.time()
    base = nll(load())
    res = {}
    m = load(); int8(m); res["int8"] = nll(m); del m
    for alpha in ALPHAS:
        m = load()
        st = act_absmax(m)
        smooth(m, st, alpha)
        res[f"smooth{alpha}-bf16"] = nll(m)  # smoothing alone must be ~lossless
        int8(m)
        res[f"smooth{alpha}-int8"] = nll(m)
        del m
        torch.cuda.empty_cache()

    summary = {}
    for name, r in res.items():
        by = {}
        for k, v in r.items():
            by.setdefault(k[0], []).append(v - base[k])
        summary[name] = {s: sum(d) / len(d) for s, d in by.items()}
    return {"gpu": torch.cuda.get_device_name(), "seconds": time.time() - t0, "dnll": summary}


@app.local_entrypoint()
def main():
    out = run.remote()
    print(f"GPU {out['gpu']}, {out['seconds']:.0f}s. Mean dNLL vs BF16 (nats/token), eager:")
    print(f"{'variant':22} {'wikitext':>9} {'mbpp':>9} {'gsm8k':>9}")
    for name, d in out["dnll"].items():
        print(f"{name:22} {d['wikitext']:+9.4f} {d['mbpp']:+9.4f} {d['gsm8k']:+9.4f}")
    Path(__file__).with_name("results").mkdir(exist_ok=True)
    Path(__file__).with_name("results").joinpath(f"smooth-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(out, indent=1))
