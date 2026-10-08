"""Compiled mixed-precision backbone with reusable CUDA graphs.

Interior projections use calibrated SmoothQuant INT8; boundary layers remain
BF16. Compilation is per decoder block to bound build time and host memory.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch
from torch import nn
from fused import Backbone
from custom_int8 import replace

class Block(nn.Module):
    def __init__(self,l):
        super().__init__()
        self.att=l.self_attn; self.mlp=l.mlp
        self.n1=l.input_layernorm; self.n2=l.post_attention_layernorm
    def forward(self,x,cos,sin):
        x=self.att(self.n1(x),(cos,sin),residual=x)
        return self.mlp(self.n2(x),residual=x)

def build(model):
    m=Backbone(model.model)
    replace(m)
    blocks=[torch.compile(Block(l),dynamic=False) for l in m.layers]
    def fn(ids):
        x=m.emb(ids)
        cos,sin=m.rot(x,torch.arange(ids.shape[1],device=ids.device)[None,:])
        for l in blocks: x=l(x,cos,sin)
        return m.norm(x)
    graphs={}
    def run(ids):
        b=ids.shape[0]
        if b not in (16,32,64): return fn(ids)
        if b not in graphs:
            static=ids.clone()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3): fn(static)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.no_grad():
                out=fn(static)
            graphs[b]=(graph,static,out)
        graph,static,out=graphs[b]
        static.copy_(ids)
        graph.replay()
        return out
    return run
