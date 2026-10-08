"""A10 INT8 tensor-core kernels with grouped scheduling and BF16 epilogues."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def mm_kernel(A,W,AS,WS,Y,R,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GM:tl.constexpr,RES:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    group=pid//(GM*nn)
    first=group*GM
    size=tl.minimum(nm-first,GM)
    pm=first+(pid%(GM*nn))%size
    pn=(pid%(GM*nn))//size
    mi=pm*BM+tl.arange(0,BM)
    ni=pn*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.int32)
    for kb in range(tl.cdiv(K,BK)):
        kk=kb*BK+ki
        a=tl.load(A+mi[:,None]*K+kk[None,:],mi[:,None]<M,other=0)
        w=tl.load(W+ni[None,:]*K+kk[:,None])
        acc+=tl.dot(a,w)
    s=tl.load(AS+mi,mi<M,other=0)
    t=tl.load(WS+ni)
    y=acc.to(tl.float32)*s[:,None]*t[None,:]
    if RES:
        r=tl.load(R+mi[:,None]*N+ni[None,:],mi[:,None]<M,other=0)
        y=y.to(tl.bfloat16).to(tl.float32)+r.to(tl.float32)
    tl.store(Y+mi[:,None]*N+ni[None,:],y,mi[:,None]<M)

def matmul(a,w,sa,sw,bm=64,bn=128,bk=64,warps=4,stages=3,gm=8,residual=None):
    m,k=a.shape; n=w.shape[1]
    y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    mm_kernel[(tr.cdiv(m,bm)*tr.cdiv(n,bn),)](a,w,sa,sw,y,residual,m,n,k,bm,bn,bk,gm,residual is not None,num_warps=warps,num_stages=stages)
    return y

@tr.jit
def quant_kernel(X,S,Q,AS,K:tl.constexpr,BK:tl.constexpr,GU:tl.constexpr=False):
    row=tl.program_id(0)
    k=tl.arange(0,BK)
    if GU:
        g=tl.load(X+row*(2*K)+k,k<K,other=0).to(tl.float32)
        u=tl.load(X+row*(2*K)+K+k,k<K,other=0).to(tl.float32)
        a=(g/(1.+tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        x=(a*u).to(tl.bfloat16).to(tl.float32)
    else:
        x=tl.load(X+row*K+k,k<K,other=0).to(tl.float32)
    s=tl.load(S+k,k<K,other=0)
    x=x*s
    scale=tl.maximum(tl.max(tl.abs(x),0),1.e-8)/127.
    q=tl.extra.cuda.libdevice.nearbyint(x/scale)
    q=tl.minimum(tl.maximum(q,-127.),127.).to(tl.int8)
    tl.store(Q+row*K+k,q,k<K)
    tl.store(AS+row,scale)

def quantize(x,smooth,gu=False):
    m,k=x.shape
    if gu: k=k//2
    q=torch.empty((m,k),device=x.device,dtype=torch.int8)
    s=torch.empty((m,1),device=x.device,dtype=torch.float32)
    quant_kernel[(m,)](x,smooth,q,s,k,tr.next_power_of_2(k),gu,num_warps=4 if k<=2048 else 8)
    return q,s

@tr.jit
def gu_kernel(A,W,AS,WS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GM:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    group=pid//(GM*nn); first=group*GM
    size=tl.minimum(nm-first,GM)
    pm=first+(pid%(GM*nn))%size
    pn=(pid%(GM*nn))//size
    mi=pm*BM+tl.arange(0,BM)
    ni=pn*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK)
    g=tl.zeros((BM,BN),tl.int32)
    u=tl.zeros((BM,BN),tl.int32)
    for kb in range(tl.cdiv(K,BK)):
        kk=kb*BK+ki
        a=tl.load(A+mi[:,None]*K+kk[None,:],mi[:,None]<M,other=0)
        w=tl.load(W+ni[None,:]*K+kk[:,None])
        v=tl.load(W+(ni[None,:]+N)*K+kk[:,None])
        g+=tl.dot(a,w)
        u+=tl.dot(a,v)
    s=tl.load(AS+mi,mi<M,other=0)
    sg=tl.load(WS+ni); su=tl.load(WS+ni+N)
    gf=(g.to(tl.float32)*s[:,None]*sg[None,:]).to(tl.bfloat16).to(tl.float32)
    uf=(u.to(tl.float32)*s[:,None]*su[None,:]).to(tl.bfloat16).to(tl.float32)
    gf=(gf/(1.+tl.exp(-gf))).to(tl.bfloat16).to(tl.float32)
    y=gf*uf
    tl.store(Y+mi[:,None]*N+ni[None,:],y,mi[:,None]<M)

def gate_up(a,w,sa,sw):
    m,k=a.shape; n=w.shape[1]//2
    y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    gu_kernel[(tr.cdiv(m,128)*tr.cdiv(n,64),)](a,w,sa,sw,y,m,n,k,128,64,64,8,num_warps=4,num_stages=2)
    return y
