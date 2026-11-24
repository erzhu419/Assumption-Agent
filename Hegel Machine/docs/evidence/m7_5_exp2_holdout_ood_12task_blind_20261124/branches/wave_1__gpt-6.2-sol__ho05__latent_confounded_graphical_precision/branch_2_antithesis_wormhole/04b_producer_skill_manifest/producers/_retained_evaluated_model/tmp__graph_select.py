import numpy as np
from collections import defaultdict
from graph_probe import generate, fit_profile


def candidates(x):
    c=np.cov(x,rowvar=False,bias=True)
    d,u=np.linalg.eigh(c); v=u[:,-2:]; w=u[:,:-2]
    z=(w*w)@(1/d[:-2]); scale=np.sqrt(z/(1-(v*v).sum(axis=1)))
    k=c*np.outer(scale,scale)
    n=len(x); p=len(c)
    base=.045*np.sqrt(900/n)
    hc=(u*np.sqrt(d))@u.T; hi=(u/np.sqrt(d))@u.T
    m=np.linalg.inv(c)
    j=np.triu_indices(p,1)
    def partial(s): return np.abs(s[j])/np.sqrt(s[j[0],j[0]]*s[j[1],j[1]])
    for factor in [.65,1.,1.35]:
        s=fit_profile(k,base*factor); q=partial(s)
        for cut in [2.2,2.5,2.8,3.1,3.4]:
            yield f'profile {factor} {cut}', q>=cut/np.sqrt(n)
        r=s*np.outer(scale,scale)
        e,b=np.linalg.eigh(hc@r@hc); b=hi@b[:,-2:]
        corrected=m+(b*np.maximum(e[-2:]-1,0))@b.T
        for cut in [3.2,3.5,3.8,4.1,4.4]:
            yield f'correct {factor} {cut}', partial(corrected)>=cut/np.sqrt(n)
    for cut in [3.2,3.5,3.8,4.1,4.4]:
        yield f'restuff {cut}', partial((w/d[:-2])@w.T+v@np.linalg.inv(v.T@((1/scale[:,None]**2)*v))@v.T)>=cut/np.sqrt(n)

if __name__=='__main__':
    import sys
    public=np.load('/holdout_task/public_graph.npz'); x=public['X']; a=public['gt_adj']; modes=('ring','sign','random','weak','hubs','coherent','hetero','weaker_factors'); results=defaultdict(dict); ij=np.triu_indices(24,1)
    for mode in ('public','public450','public675',*modes):
        if mode.startswith('public'):
            size=int(mode[6:] or 900)
            cases=[(x[np.random.default_rng(i).permutation(len(x))[:size]],a) for i in range(12 if size<900 else 1)]
        else:
            cases=[generate(seed,mode)[:2] for seed in range(18)]
        data=defaultdict(list)
        for x1,a1 in cases:
            y=a1[ij]
            for name,yhat in candidates(x1):
                tp=np.sum(yhat&y); data[name].append((2*tp/(y.sum()+yhat.sum()),int(yhat.sum()-tp),int(y.sum()-tp)))
        for name,val in data.items():
            v=np.array(val); results[name][mode]=(v[:,0].min(),v[:,0].mean(),v[:,1].mean(),v[:,2].mean())
    for name,d in sorted(results.items(),key=lambda item:(min(item[1][m][0] for m in modes),np.mean([item[1][m][1] for m in modes])),reverse=True):
        print(name, 'pub',np.round([d[m][0] for m in ('public','public450','public675')],3),'stress minimums', ' '.join(f'{d[m][0]:.3f}' for m in modes),'mean',round(np.mean([d[m][1] for m in modes]),3),'weak/hub avg FP,FN',np.round([d[m][2:] for m in ('weak','hubs')],2).tolist())
