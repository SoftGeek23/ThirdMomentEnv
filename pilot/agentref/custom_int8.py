"""Per-channel INT8 weights and per-token dynamic activation quantization.

SmoothQuant statistics are fixed calibration maxima, not evaluation inputs.
The diagonal transform balances channel ranges without changing the underlying
linear operation before quantization.
"""
import torch
from torch import nn
from int8_kernel import matmul, quantize, gate_up

class Int8Linear(nn.Module):
    def __init__(self,m,act=None,alpha=0.45):
        super().__init__()
        w=m.weight.detach().float()
        smooth=(act.cuda().clamp(min=1e-5).pow(alpha)/w.abs().amax(0).clamp(min=1e-5).pow(1-alpha)) if act is not None else torch.ones(w.shape[1],device=w.device)
        self.register_buffer('inv_smooth',1/smooth)
        w=w*smooth
        scale=w.abs().amax(1).clamp(min=1e-8)/127
        self.register_buffer('w',torch.round(w/scale[:,None]).clamp(-127,127).to(torch.int8).contiguous().t())
        self.register_buffer('scale',scale)
    def forward(self,x,gu=False,residual=None,swiglu=False):
        shape=x.shape
        x=x.reshape(-1,shape[-1])
        if gu:
            q,scale=quantize(x,self.inv_smooth,True)
        else:
            x=x.float()*self.inv_smooth
            scale=x.abs().amax(-1,keepdim=True).clamp(min=1e-8)/127
            q=torch.round(x/scale).clamp(-127,127).to(torch.int8)
        if swiglu:
            return gate_up(q,self.w,scale,self.scale).reshape(*shape[:-1],-1)
        if q.shape[1]>2048 and q.shape[0]>=8192:
            y=matmul(q,self.w,scale,self.scale,128,256,128,8,3,gm=4,residual=residual)
        elif q.shape[1]>2048 or self.w.shape[1]>4096:
            y=matmul(q,self.w,scale,self.scale,256,128,128,8,3,residual=residual)
        else:
            y=matmul(q,self.w,scale,self.scale,128,128,64,4,3,residual=residual)
        return y.reshape(*shape[:-1],-1)

def replace(m,stats=None,prefix=""):
    if stats is None:
        from pathlib import Path
        stats=torch.load(Path(__file__).parent/"act_stats.pt",weights_only=True)
    for name,c in list(m.named_children()):
        if isinstance(c,nn.Linear):
            if prefix.startswith("layers.0.") or prefix.startswith("layers.27."): continue
            setattr(m,name,Int8Linear(c,stats[prefix+name]))
        else: replace(c,stats,prefix+name+".")
