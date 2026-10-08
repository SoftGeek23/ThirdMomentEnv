"""Rollout qaamH3h's 1.765x recipe + three additions (flags), for the reference search.

Base (qaamH3h): all 28 layers INT8 W8A8, SmoothQuant alpha 0.7 from stats.pt, one grouped Triton
INT8 GEMM with fused dequant (custom op, so Inductor compiles each whole layer), CUDA graphs.
  RESIDUAL  add the residual inside the o_proj / down_proj GEMM epilogue
  TILES     pick tile sizes per GEMM shape instead of one config
  STATIC    per-tensor static activation scales for the RMSNorm-fed inputs (qkv, gate_up),
            computed from the same calibration maxima: max_j(act_max_j / smooth_j) / 127
"""
import os

import torch
import triton
import triton.language as tl
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

RESIDUAL = False
TILES = False
STATIC = False
ALPHA = 0.7
MAX_M = 64 * 256


@triton.jit
def kernel(A, W, AS, WS, Y, R, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr, RES: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM); nn_ = tl.cdiv(N, BN)
    group = pid // (G * nn_)
    first = group * G
    gs = tl.minimum(nm - first, G)
    pm = first + (pid % (G * nn_)) % gs
    pn = (pid % (G * nn_)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for k in range(tl.cdiv(K, BK)):
        a = tl.load(A + rm[:, None] * K + (rk[None, :] + k * BK), mask=rm[:, None] < M, other=0)
        w = tl.load(W + rn[None, :] * K + (rk[:, None] + k * BK))
        acc = tl.dot(a, w, acc)
    ys = acc.to(tl.float32) * tl.load(AS + rm, mask=rm < M, other=0)[:, None] * tl.load(WS + rn)[None, :]
    if RES:
        r = tl.load(R + rm[:, None] * N + rn[None, :], mask=rm[:, None] < M, other=0)
        ys = ys.to(tl.bfloat16).to(tl.float32) + r.to(tl.float32)
    tl.store(Y + rm[:, None] * N + rn[None, :], ys, mask=rm[:, None] < M)


def _cfg(m, n, k):
    if not TILES:
        return 128, 256, 128, 8, 8, 3
    if n > 2048 and m >= 8192:
        return 128, 256, 128, 4, 8, 3
    if n > 4096 or k > 2048:
        return 256, 128, 128, 8, 8, 3
    return 128, 128, 64, 8, 4, 3


@torch.library.custom_op("fastint2::mm", mutates_args=())
def mm(a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    m, k = a.shape; n = w.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    bm, bn, bk, g, nw, ns = _cfg(m, n, k)
    kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](a, w, sa, sw, out, out, m, n, k, bm, bn, bk, g, False,
                                                       num_warps=nw, num_stages=ns, enable_fp_fusion=False)
    return out


@mm.register_fake
def _(a, w, sa, sw):
    return torch.empty((a.shape[0], w.shape[1]), device=a.device, dtype=torch.bfloat16)


@torch.library.custom_op("fastint2::mm_res", mutates_args=())
def mm_res(a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    m, k = a.shape; n = w.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    bm, bn, bk, g, nw, ns = _cfg(m, n, k)
    kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](a, w, sa, sw, out, r.contiguous(), m, n, k, bm, bn, bk, g,
                                                       True, num_warps=nw, num_stages=ns, enable_fp_fusion=False)
    return out


@mm_res.register_fake
def _(a, w, sa, sw, r):
    return torch.empty((a.shape[0], w.shape[1]), device=a.device, dtype=torch.bfloat16)


class Int8Linear(nn.Module):
    def __init__(self, m, act, static=False):
        super().__init__()
        w = m.weight.detach().float()
        smooth = (act.cuda().float().clamp(min=1e-5).pow(ALPHA) /
                  w.abs().amax(0).clamp(min=1e-5).pow(1 - ALPHA)).clamp(min=1e-3)
        self.register_buffer("smooth", smooth)
        w = w * smooth[None, :]
        scale = w.abs().amax(1).clamp(min=1e-8) / 127
        self.register_buffer("w", (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous().t())
        self.register_buffer("scale", scale)
        self.static = static
        if static:
            s = float((act.cuda().float() / smooth).max()) / 127.0
            self.s = s
            self.register_buffer("as_buf", torch.full((MAX_M, 1), s, device="cuda", dtype=torch.float32))

    def forward(self, x, residual=None):
        shape = x.shape
        xf = x.reshape(-1, shape[-1]).float() / self.smooth
        if self.static:
            scale = self.as_buf[:xf.shape[0]]
            q = (xf / self.s).round().clamp(-127, 127).to(torch.int8)
        else:
            scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
            q = (xf / scale).round().clamp(-127, 127).to(torch.int8)
        if residual is not None:
            y = mm_res(q, self.w, scale, self.scale, residual.reshape(-1, residual.shape[-1]))
        else:
            y = mm(q, self.w, scale, self.scale)
        return y.reshape(*shape[:-1], -1)


class Attention(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.qkv = nn.Linear(2048, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
        self.qkv.weight = nn.Parameter(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]))
        self.q_norm = a.q_norm; self.k_norm = a.k_norm; self.o_proj = a.o_proj

    def forward(self, x, pos, residual):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).split([2048, 1024, 1024], dim=-1)
        q = self.q_norm(q.reshape(b, s, 16, 128)).transpose(1, 2)
        k = self.k_norm(k.reshape(b, s, 8, 128)).transpose(1, 2)
        v = v.reshape(b, s, 8, 128).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *pos)
        k = k.repeat_interleave(2, dim=1); v = v.repeat_interleave(2, dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, s, 2048)
        if RESIDUAL:
            return self.o_proj(y, residual=residual)
        return self.o_proj(y) + residual


class MLP(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.gate_up = nn.Linear(2048, 12288, bias=False, device="cuda", dtype=torch.bfloat16)
        self.gate_up.weight = nn.Parameter(torch.cat([m.gate_proj.weight, m.up_proj.weight]))
        self.down_proj = m.down_proj

    def forward(self, x, residual):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        a = torch.nn.functional.silu(g) * u
        if RESIDUAL:
            return self.down_proj(a, residual=residual)
        return self.down_proj(a) + residual


def build(model, stats_path=None):
    backbone = model.model
    for l in backbone.layers:
        l.self_attn = Attention(l.self_attn)
        l.mlp = MLP(l.mlp)
    stats = torch.load(stats_path or os.path.join(os.path.dirname(__file__), "stats.pt"), weights_only=True)
    keys = {"qkv": "self_attn.q_proj", "o_proj": "self_attn.o_proj", "gate_up": "mlp.gate_proj",
            "down_proj": "mlp.down_proj"}
    for i, l in enumerate(backbone.layers):
        for m in (l.self_attn, l.mlp):
            for name, child in list(m.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(m, name, Int8Linear(child, stats[f"layers.{i}.{keys[name]}"],
                                                static=STATIC and name in ("qkv", "gate_up")))

    def layer(x, l, pos):
        x = l.self_attn(l.input_layernorm(x), pos, residual=x)
        return l.mlp(l.post_attention_layernorm(x), residual=x)

    cl = torch.compile(layer, dynamic=False)
    norm = torch.compile(backbone.norm, dynamic=False)
    pos = backbone.rotary_emb(torch.empty(1, 256, 2048, device="cuda", dtype=torch.bfloat16),
                              torch.arange(256, device="cuda")[None, :])

    def forward(ids):
        x = backbone.embed_tokens(ids)
        for l in backbone.layers:
            x = cl(x, l, pos)
        return norm(x)

    cache = {}

    def fn(ids):
        b = ids.shape[0]
        if b not in cache:
            inp = torch.empty_like(ids)
            inp.copy_(ids)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward(inp)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = forward(inp)
            cache[b] = (inp, graph, out)
        inp, graph, out = cache[b]
        inp.copy_(ids)
        graph.replay()
        return out

    return fn
