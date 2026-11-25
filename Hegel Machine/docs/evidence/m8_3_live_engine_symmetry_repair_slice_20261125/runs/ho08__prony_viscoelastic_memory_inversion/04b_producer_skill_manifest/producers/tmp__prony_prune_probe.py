import sys
from itertools import combinations
import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import prony_rheology as base


def fits(t, strain, y):
    drift = np.column_stack((np.ones(len(t)), (t-t.mean())/np.ptp(t)))
    grid = np.geomspace(0.02, 20., 96)
    A = base._design(t, strain, grid)
    keep, s = base._robust_screen(A, drift, y)
    Q = qr(drift[keep], mode='economic')[0]
    obs = base._project(Q, y[keep])
    B = base._project(Q, A[keep])
    g = base._positive_fit(B, obs)
    inds = np.flatnonzero(g[1:] > 1e-7 * g.max())
    groups = np.split(inds, np.flatnonzero(np.diff(inds)>2)+1) if len(inds) else []
    bounds = np.log(.02), np.log(20.)
    z0 = np.clip([np.average(np.log(grid[j]),weights=g[1:][j]) for j in groups], *bounds)

    def fit(z, strength=0):
        def calc(z):
            B = base._project(Q, base._design(t,strain,np.exp(z))[keep])
            c = base._positive_fit(B,obs)
            return c,(obs-B@c)/s
        def fun(z):
            r = calc(z)[1]
            return np.r_[r,strength*np.diff(np.sort(z),n=2)] if strength else r
        if len(z):
            z = least_squares(fun,z,bounds=bounds,max_nfev=65).x
        c,r = calc(z)
        return z,c,r@r

    best = fit(z0)
    models = [best]
    if 2 <= len(z0) <= 9:
        starts = best[0]
        for j in range(2):
            if len(starts) < 2:
                break
            opts = [fit(np.delete(starts,k)) for k in range(len(starts))]
            model = min(opts,key=lambda a:a[2]); models.append(model)
            if model[2] > best[2] + 12:
                break
            starts = model[0]
    out = {}
    for penalty in (1., 2., 4., 7.):
        z,c,cost = min(models,key=lambda a:a[2]+penalty*len(a[0]))
        if len(z) >= 3:
            for strength in (18,10,6,3):
                zz,cc,err=fit(z,strength)
                if err <= cost+len(z)-2:
                    z,c=zz,cc
                    break
        out[penalty] = np.exp(z),c
    return out, [(len(z), round(cost-best[2],2)) for z,_,cost in models]


def scores(t,x,y,tt,xx,true,g):
    proposed, profile = fits(t,x,y)
    b=base.fit_prony_and_predict_stress(dict(train_t=t,train_strain=x,train_stress_raw=y,test_t=tt,test_strain=xx))
    preds={'base':(b['G_inf'],b['test_stress'])}
    preds.update({k:(c[0],base._design(tt,xx,tau)@c) for k,(tau,c) in proposed.items()})
    rows={}
    for k,(mod,pred) in preds.items():
        delta=pred-true
        rows[k]=[abs(mod/g-1),np.linalg.norm(delta)/np.linalg.norm(true),max(np.linalg.norm(delta[i])/np.linalg.norm(true[i]) for i in np.array_split(np.arange(len(tt)),4))]
    return profile,rows
