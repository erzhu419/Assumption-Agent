import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


def fit_profile(C, a, start=None, support=None, steps=120):
    p=len(C); eye=np.eye(p); penalty=np.broadcast_to(a,(p,p)).copy(); np.fill_diagonal(penalty,0)
    d,U=eigh(C); H=(U*np.sqrt(d))@U.T
    def evaluate(S):
        try: R=np.linalg.cholesky(S)
        except np.linalg.LinAlgError: return np.inf,None
        V=H@S@H
        v,W=eigh(V,subset_by_index=[p-2,p-1],check_finite=False)
        v=np.maximum(v,1); B=H@W
        f=-2*np.log(R.diagonal()).sum()+np.trace(V)-np.sum(v-1-np.log(v))
        g=C-(B*(1-1/v))@B.T-np.linalg.solve(S,eye)
        return f,g
    if support is not None:
        i,j=np.where(np.triu(support | np.eye(p,dtype=bool))); scale=np.where(i==j,1.,2.**.5)
        def unpack(b):
            S=np.zeros_like(C); S[i,j]=S[j,i]=b/scale; return S
        def obj(b):
            f,g=evaluate(unpack(b)); return (f,g[i,j]*scale) if g is not None else (1e6,np.zeros_like(b))
        res=minimize(obj,start[i,j]*scale,method='L-BFGS-B',jac=True,options={'maxiter':steps,'ftol':1e-11,'gtol':1e-7})
        return unpack(res.x)
    S=np.eye(p) if start is None else start.copy(); Y=S.copy(); t=1.; step=1.
    for _ in range(steps):
        fy,g=evaluate(Y)
        if g is None: Y=S; t=1.; fy,g=evaluate(Y)
        while True:
            Z=Y-step*g; Z=np.sign(Z)*np.maximum(np.abs(Z)-step*penalty,0)
            fz,_=evaluate(Z); delta=Z-Y
            if fz <= fy+np.sum(g*delta)+np.sum(delta**2)/(2*step)+1e-12: break
            step*=.5
        if np.max(np.abs(Z-S))<2e-6: return Z
        tt=(1+np.sqrt(1+4*t*t))/2
        Y=Z+(t-1)/tt*(Z-S); S=Z; t=tt; step=min(step*1.12,4.)
    return S


def models(X):
    C=np.cov(X,rowvar=False,bias=True); d,U=eigh(C); V=U[:,-2:]
    R=(U[:,:-2]*d[:-2])@U[:,:-2].T; P=(U[:,:-2]/d[:-2])@U[:,:-2].T
    q=np.sqrt(np.diag(P)/(1-np.sum(V**2,axis=1))); K=C*np.outer(q,q)
    yield 'raw',np.linalg.inv(C)
    for m in [.75,1.,1.25]:
        yield f'restuff{m}',P+V@np.linalg.inv(m*(V.T@(1/q[:,None]**2*V)))@V.T
    for a in [.025,.045,.065,.085]:
        S=fit_profile(K,a)
        yield f'profile{a}',S
        Q=np.abs(S)/np.sqrt(np.outer(S.diagonal(),S.diagonal())); np.fill_diagonal(Q,0)
        for fraction in [.7]:
            weights=a*np.clip(1-Q/.17,fraction,1.)
            S2=fit_profile(K,weights,start=S,steps=70)
            yield f'adapt{a}',S2


def generate(seed, mode, n=900):
    rng=np.random.default_rng(seed); p=24; ix=rng.permutation(p); A=np.zeros((p,p),bool)
    def put(i,j): A[i,j]=A[j,i]=True
    if mode in ('ring','sign','weak','coherent','hetero','weaker_factors'):
        for i in range(p): put(ix[i],ix[(i+1)%p])
        for i,j in rng.choice(p,size=(4,2),replace=False): put(ix[i],ix[j])
    else:
        while A.sum()<2*(23+seed%10):
            i,j=rng.choice(p,2,replace=False)
            if A.sum(0)[[i,j]].max()< (5 if mode=='hubs' else 4): put(i,j)
    S=2.2*np.eye(p); i,j=np.where(np.triu(A,1)); weights=np.full(len(i),.52)
    if mode in ('weak','hubs'): weights=rng.uniform(.34,.59,len(i))
    if mode!='ring': weights*=rng.choice([-1.,1.],len(i))
    S[i,j]=S[j,i]=-weights
    if mode=='hetero':
        sc=np.exp(rng.uniform(-.55,.55,p)); S*=np.outer(sc,sc)
    B=rng.normal(size=(p,2))*.9
    if mode=='coherent': B[:5]*=3
    if mode=='weaker_factors': B*=.4
    X=rng.standard_normal((n,p))@np.linalg.cholesky(np.linalg.inv(S)).T + rng.standard_normal((n,2))@B.T
    return X,A,S


def scores(S,A):
    i,j=np.triu_indices(len(A),1); y=A[i,j]; Q=np.abs(S[i,j])/np.sqrt(S[i,i]*S[j,j]); idx=np.argsort(-Q); t=np.cumsum(y[idx]); f=2*t/(np.arange(len(y))+1+y.sum());
    z=[f.max(),f[25]]
    for cut in [.06,.08,.10,.12,.14,.16]:
        pred=Q>=cut; z.append(2*(pred&y).sum()/(pred.sum()+y.sum()))
    return z

if __name__=='__main__':
    from collections import defaultdict
    f=np.load('/holdout_task/public_graph.npz'); cases=[('public',[(f['X'],f['gt_adj'],f['gt_S'])])]+[(mode,[generate(seed,mode) for seed in range(15)]) for mode in ('ring','sign','random','weak','hubs','coherent','hetero','weaker_factors')]
    for mode,test in cases:
        D=defaultdict(list)
        for X,A,_ in test:
            for name,S in models(X): D[name].append(scores(S,A))
        print(mode,'columns = oracle, top26, thresholds .06 to .16; mean / minimum',flush=True)
        for name,v in D.items(): print(name,' '.join(f'{a:.3f}/{b:.3f}' for a,b in zip(np.mean(v,axis=0),np.min(v,axis=0))),flush=True)
