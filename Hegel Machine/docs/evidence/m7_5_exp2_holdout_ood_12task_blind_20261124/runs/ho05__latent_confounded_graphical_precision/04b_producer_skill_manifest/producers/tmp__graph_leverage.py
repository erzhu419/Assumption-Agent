import numpy as np
from scipy.linalg import eigh
from graph_equivariant import case, residual_scale
from deconfound_graph import _fit, _profile

I,J=np.triu_indices(24,1)
NAMES=('five','five270','avg','avg270','lev mild','lev strong','adaptive','adaptive lev','adaptive avg')

def predictions(x):
    n,p=x.shape
    z=x-x.mean(axis=0)
    z=z/np.sqrt(np.mean(z*z,axis=0))
    corr=z.T@z/n
    d0=residual_scale(corr)
    d=d0.copy()
    for _ in range(5):
        d*=residual_scale(corr*np.outer(d,d))**.75
    c=corr*np.outer(d,d)
    evaluate=_profile(c)
    alpha=.8775/np.sqrt(n)
    s=_fit(evaluate,np.eye(p),penalty=alpha)
    s0=_fit(_profile(corr*np.outer(d0,d0)),np.eye(p),penalty=alpha)
    r=np.sqrt(s0.diagonal())
    sr=_fit(_profile(corr*np.outer(d0*r,d0*r)),s0/np.outer(r,r),penalty=alpha)
    def partial(a):
        return np.abs(a)/np.sqrt(np.outer(a.diagonal(),a.diagonal()))*np.sqrt(n)
    q=partial(s)
    qr=partial(sr)
    avg=.7*q+.3*qr
    # Diagonal of the latent projector in the fitted conditional whitening.
    dv,u=eigh(s); root=(u*np.sqrt(dv))@u.T
    _,v=eigh(root@c@root,subset_by_index=(p-2,p-1))
    # Map pair directions into precision-whitened coordinates. The squared
    # retained length estimates the local information left after profiling.
    a=root.copy()/np.sqrt(s.diagonal())[None,:]
    b=a-v@(v.T@a)
    keep=b.T@b
    info=np.sqrt(np.clip(np.outer(keep.diagonal(),keep.diagonal())+keep**2,.16,1.))
    adaptive=alpha*np.clip(1-q/12,.72,1)
    sa=partial(_fit(evaluate,s,penalty=adaptive,iterations=100))
    sl=partial(_fit(evaluate,s,penalty=adaptive*info,iterations=100))
    for t in (q>=2.5,q>=2.7,avg>=2.5,avg>=2.7,q>=2.5*info**.25,q>=2.5*info**.5,sa>=2.75,sl>=2.85/info**.25,(sa+avg)/2>=2.65):
        yield t[I,J]

if __name__=='__main__':
    z=np.load('/holdout_task/public_graph.npz')
    print(NAMES,flush=True)
    for mode in ('public','coherent','hetero_coherent','hubs','weak','hetero','random','factor2_weak','ring','sign','weaker_factors'):
        stats=[]
        seeds=list(range(160,186))
        if 'coherent' in mode: seeds+= [33,37,63,69,96,102,106]
        cases=[(z['X'],z['gt_adj'])] if mode=='public' else (case(i,mode) for i in seeds)
        for x,a in cases:
            y=a[I,J]
            stats.append([2*(b&y).sum()/(b.sum()+y.sum()) for b in predictions(x)])
        v=np.array(stats)
        print(mode,'min',np.round(v.min(0),3),'avg',np.round(v.mean(0),3),flush=True)
