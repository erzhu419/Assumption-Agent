"""Recover latent coordinates, then fit the two calibrated response branches."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k]) for k in ('u','v'))
    w = np.roll(u, 2, axis=1)
    rng = np.random.default_rng(420)
    samples = [rng.choice(len(u),6,replace=False) for _ in range(90)]
    def basis(z, f):
        return np.stack((np.ones_like(z),z,np.sin(f*np.roll(z,2,axis=1))),axis=2)
    def coeff(a, ids):
        return np.stack([_ls(a[ids,j], v[ids,j]) for j in range(u.shape[1])])
    def pred(a,b):
        return np.einsum('njk,jk->nj',a,b)
    best = np.inf
    for f in np.arange(0.8,5.01,.15):
        a = basis(u,f)
        for ids in samples:
            b = coeff(a,ids)
            er = np.linalg.norm(pred(a,b)-v,axis=1)
            score = np.median(er)
            if score < best:
                best, start, er0 = score, f, er
    ids = np.argsort(er0)[:max(6,int(len(u)*.75))]
    for _ in range(3):
        def residual(p):
            a = basis(u,p[0]); return (pred(a, coeff(a,ids))[ids]-v[ids]).ravel()
        f = least_squares(residual,[start],bounds=(.1,8.)).x[0]
        a = basis(u,f); b = coeff(a,ids)
        er = np.linalg.norm(pred(a,b)-v,axis=1)
        ids = np.flatnonzero(er < max(.002, 4*np.median(er)))
        start=f
    print("freq",f,"coef",b,"quantiles",np.quantile(er,[.25,.5,.75,.85,.9,.95,1])); z = (x-b[:,0])/b[:,1]
    for _ in range(80):
        nxt = (x-b[:,0]-b[:,2]*np.sin(f*np.roll(z,2,axis=1)))/b[:,1]
        if np.max(np.abs(nxt-z),initial=0.) < 1e-12: break
        z = nxt
    return z


def _branch(panel, z, k):
    u, v = (np.asarray(panel[j]) for j in ('u','v'+str(k)))
    def design(w,p):
        q = w+p[1:]
        if k == 1:
            h = q/(1+p[0]*abs(q))
        else:
            h = q*q/(1+p[0]*q*q)
        return _ones(np.c_[w,h])
    def residual(p):
        a = design(u,p)
        return (a @ _ls(a,v)-v).ravel()
    p = least_squares(residual, [.85 if k == 1 else .165,0,0,0,0],
                      bounds=([.03,-.2,-.2,-.2,-.2],[3,.2,.2,.2,.2])).x
    return design(z,p) @ _ls(design(u,p),v)


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    x = np.vstack((s['train']['x'], s['test']['x']))
    z = _calibrate(s['panel_a'], x)
    b = s['panel_b']
    l = _ones(z) @ _ls(_ones(b['u']), b['v'])
    q, r = (_branch(s['panel_c'], z, k) for k in (1,2))
    return dict(z=z, L=l, Q=q, R=r, product=q, cosine=r)
