"""Smoothed dynamic INT8 backbone for Qwen3-1.7B.

Fixed channel maxima in stats.pt were calibrated ahead of time on development
texts. Activations use per-token symmetric INT8 and weights use per-output
channel INT8. Projection fusion, grouped Triton GEMMs with FP32 rescaling,
layerwise compilation and CUDA graph replay reduce inference cost.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(A,W,AS,WS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,G:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    group=pid//(G*nn)
    first=group*G
    gs=tl.minimum(nm-first,G)
    pm=first+(pid%(G*nn))%gs
    pn=(pid%(G*nn))//gs
    rm=pm*BM+tl.arange(0,BM)
    rn=pn*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.int32)
    for k in range(tl.cdiv(K,BK)):
        a=tl.load(A+rm[:,None]*K+(rk[None,:]+k*BK),mask=rm[:,None]<M,other=0)
        w=tl.load(W+rn[None,:]*K+(rk[:,None]+k*BK))
        acc=tl.dot(a,w,acc)
    ys=acc.to(tl.float32)*tl.load(AS+rm,mask=rm<M,other=0)[:,None]*tl.load(WS+rn)[None,:]
    tl.store(Y+rm[:,None]*N+rn[None,:],ys,mask=rm[:,None]<M)


@torch.library.custom_op('fastint::mm',mutates_args=())
def mm(a:torch.Tensor,w:torch.Tensor,sa:torch.Tensor,sw:torch.Tensor)->torch.Tensor:
    m,k=a.shape; n=w.shape[1]
    out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    kernel[(triton.cdiv(m,128)*triton.cdiv(n,256),)](a,w,sa,sw,out,m,n,k,128,256,128,8,num_warps=8,num_stages=3,enable_fp_fusion=False)
    return out

@mm.register_fake
def _(a,w,sa,sw):
    return torch.empty((a.shape[0],w.shape[1]),device=a.device,dtype=torch.bfloat16)

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

class Attention(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.qkv = nn.Linear(2048,4096,bias=False,device='cuda',dtype=torch.bfloat16)
        self.qkv.weight = nn.Parameter(torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]))
        self.q_norm=a.q_norm; self.k_norm=a.k_norm; self.o_proj=a.o_proj
    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        b,s,_=hidden_states.shape
        q,k,v=self.qkv(hidden_states).split([2048,1024,1024],dim=-1)
        q=self.q_norm(q.reshape(b,s,16,128)).transpose(1,2)
        k=self.k_norm(k.reshape(b,s,8,128)).transpose(1,2)
        v=v.reshape(b,s,8,128).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        k=k.repeat_interleave(2,dim=1); v=v.repeat_interleave(2,dim=1)
        y=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o_proj(y.transpose(1,2).reshape(b,s,2048)),None

class MLP(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.gate_up=nn.Linear(2048,12288,bias=False,device='cuda',dtype=torch.bfloat16)
        self.gate_up.weight=nn.Parameter(torch.cat([m.gate_proj.weight,m.up_proj.weight]))
        self.down_proj=m.down_proj
    def forward(self,x):
        g,u=self.gate_up(x).chunk(2,dim=-1)
        return self.down_proj(torch.nn.functional.silu(g)*u)

class Int8Linear(nn.Module):
    def __init__(self, m, a):
        super().__init__()
        w=m.weight.detach().float()
        smooth=(a.cuda().float().clamp(min=1e-5).pow(0.7)/w.abs().amax(0).clamp(min=1e-5).pow(0.3)).clamp(min=1e-3)
        self.register_buffer('smooth',smooth)
        w=w*smooth[None,:]
        scale=w.abs().amax(1).clamp(min=1e-8)/127
        self.register_buffer('w', (w/scale[:,None]).round().clamp(-127,127).to(torch.int8).contiguous().t())
        self.register_buffer('scale',scale)
    def forward(self,x):
        shape=x.shape
        xf=x.reshape(-1,shape[-1]).float()/self.smooth
        scale=xf.abs().amax(-1,keepdim=True).clamp(min=1e-8)/127
        q=(xf/scale).round().clamp(-127,127).to(torch.int8)
        y=mm(q,self.w,scale,self.scale)
        return y.reshape(*shape[:-1],-1)

def build(model):
    backbone=model.model
    for l in backbone.layers:
        l.self_attn=Attention(l.self_attn)
        l.mlp=MLP(l.mlp)
    import os
    stats=torch.load(os.path.join(os.path.dirname(__file__),'stats.pt'),weights_only=True)
    for i,l in enumerate(backbone.layers):
        for m in (l.self_attn,l.mlp):
            for name,child in list(m.named_children()):
                if isinstance(child,nn.Linear):
                    key={'qkv':'self_attn.q_proj','o_proj':'self_attn.o_proj','gate_up':'mlp.gate_proj','down_proj':'mlp.down_proj'}[name]
                    setattr(m,name,Int8Linear(child,stats[f'layers.{i}.{key}']))
    def layer(x, l, pos):
        r=x
        x=l.input_layernorm(x)
        x=l.self_attn(x,position_embeddings=pos)[0]+r
        r=x
        x=l.post_attention_layernorm(x)
        return l.mlp(x)+r
    cl=torch.compile(layer,dynamic=False)
    norm=torch.compile(backbone.norm,dynamic=False)
    pos=backbone.rotary_emb(torch.empty(1,256,2048,device='cuda',dtype=torch.bfloat16),torch.arange(256,device='cuda')[None,:])
    def forward(ids):
        x=backbone.embed_tokens(ids)
        for l in backbone.layers:
            x=cl(x,l,pos)
        return norm(x)
    cache={}
    def fn(ids):
        b=ids.shape[0]
        if b not in cache:
            inp=torch.empty_like(ids)
            inp.copy_(ids)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward(inp)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out=forward(inp)
            cache[b]=(inp,graph,out)
        inp,graph,out=cache[b]
        inp.copy_(ids)
        graph.replay()
        return out
    return fn
