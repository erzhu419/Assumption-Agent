"""Reconstruct calibrated branch features and fit a regularized response."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def features(s, x):
    u, v = (s['panel_a'][k] for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    pars = []
    for k in range(u.shape[1]):
        d = u[i, k]-u[j, k]; ok = np.abs(d)>1e-8
        m = np.median((v[i[ok], k]-v[j[ok], k])/d[ok]); r = v[:,k]-m*u[:,k]
        r -= np.median(r); g = np.abs(r)<max(.001, 6*np.median(abs(r)))
        pars.append(_ls(_ones(u[g,k:k+1]),v[g,k]))
    off, scale = np.array(pars).T
    z = (x-off)/scale
    b, c = s['panel_b'], s['panel_c']
    def q(w): return _ones(w[:,[0,0,1,2]]*w[:,[1,3,2,3]])
    def r(w): return _ones(np.cos(2.4*w))
    L = _ones(z) @ _ls(_ones(b['u']), b['v'])
    Q = q(z) @ _ls(q(c['u']),c['v1'])
    R = r(z) @ _ls(r(c['u']),c['v2'])
    return z,L,Q,R,q(z)[:,1:],r(z)[:,1:]


def _dist(a):
    return np.maximum(0,((a[:,None,:]-a[None,:,:])**2).sum(-1))


def _models(blocks):
    z,l,q,r,p,c = blocks
    cross=(p[:,:,None]*c[:,None,:]).reshape(len(z),-1)
    ds=_dist(z)
    fs=[_ones(np.c_[p,c,cross]), _ones(np.c_[z,p,c,cross]), _ones(np.c_[p,c,(q*r).sum(1)]), np.ones((len(z),1))]
    for ix,F in enumerate(fs):
        lin=F@F.T/F.shape[1]
        for D in [ds, _dist(np.c_[q,r]), ds+.5*_dist(np.c_[q,r]), ds+.5*_dist(l)]:
            for h in [0.9,1.7,3.,5.]:
                for K in [np.exp(-D/h**2),(1+D/h**2)**(-2)]:
                    for w in [.1,1.,10.,100.] if ix<3 else [1.]:
                        yield K+w*lin
    yield fs[0]@fs[0].T/fs[0].shape[1]


def _predict(s, rows, xnew):
    t=s['train']; n=len(t['x'][rows]); blocks=features(s,np.vstack([t['x'][rows],xnew])); traj=t['traj'][rows]
    U,_,V=np.linalg.svd(traj,full_matrices=False); H=V[:3]; target=np.c_[t['y'][rows],traj@H.T]
    count=target.shape[1]; models=[[] for _ in range(count)]; I=np.eye(n)
    for K in _models(blocks):
        for lam in [1e-6,1e-4,.002,.025]:
            f=cho_factor(K[:n,:n]+lam*I,check_finite=False); alpha=cho_solve(f,target,check_finite=False); inv=cho_solve(f,I,check_finite=False).diagonal(); e=np.mean(abs(alpha/inv[:,None]),axis=0)
            pred=K[n:,:n]@alpha
            for j in range(count):models[j].append((e[j],pred[:,j]))
    out=[]
    for m in models:
        best=sorted(m,key=lambda v:v[0])[:10]; w=np.exp(-15*(np.array([a for a,b in best])/best[0][0]-1));out.append(np.average(np.stack([b for a,b in best]),axis=0,weights=w))
    pred=np.array(out).T
    return pred[:,0],pred[:,1:]@H


def solve_batch(study: dict) -> dict:
    y,state=_predict(study,slice(None),study['test']['x']); end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
