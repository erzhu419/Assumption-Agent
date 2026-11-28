import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement


def affine(x):
    return np.column_stack((np.ones(len(x)), x))


def ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def calibrate(panel):
    u, v = panel['u'], panel['v']
    a = affine(u)
    w = ls(a, v)
    best = np.median(np.linalg.norm(a @ w - v, axis=1))
    if best > .001:
        rng = np.random.default_rng(218)
        for _ in range(160):
            j = rng.choice(len(u), 5, replace=False)
            wj = ls(a[j], v[j])
            e = np.median(np.linalg.norm(a @ wj - v, axis=1))
            if e < best:
                best, w = e, wj
    for _ in range(5):
        e = np.linalg.norm(a @ w - v, axis=1)
        good = e < max(1e-9, 4*np.median(e))
        if good.sum() >= 5:
            w = ls(a[good], v[good])
    return w


def quadratic(x):
    return np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]), 2)])


def outer(a, b):
    return (a[:,:,None]*b[:,None,:]).reshape(len(a), -1)


def spatial(study, x):
    w = calibrate(study['panel_a'])
    z = np.linalg.solve(w[1:].T, (x-w[0]).T).T
    panel = study['panel_c']
    u, v1, v2 = panel['u'], panel['v1'], panel['v2']
    q = quadratic(u)
    weights = ls(np.c_[affine(u),q],v1)
    keep = np.linalg.norm(weights,axis=1) > .003
    keep[0] = True
    p = np.c_[affine(z),quadratic(z)][:,keep] @ ls(np.c_[affine(u),q][:,keep],v1)
    def basis(a,k):
        q = a*a
        den = 1 + k[0]*q + k[1]*q*q
        return np.c_[np.ones(len(a)), q/den, q*q/den]
    def residual(k):
        f = basis(u,k)
        return (f @ ls(f,v2)-v2).ravel()
    k = least_squares(residual, [.25,.04], bounds=(0,3), gtol=1e-12).x
    r = basis(z,k) @ ls(basis(u,k),v2)
    return z,p,r


def ridge(f, target, query, penalty):
    fm, ym = f.mean(axis=0), target.mean(axis=0)
    a = f-fm
    return (query-fm) @ np.linalg.solve(a.T@a+penalty*np.eye(f.shape[1]),a.T@(target-ym))+ym


def augment(base, pool, y, tolerance=.009, limit=12):
    n=len(y)
    y=np.asarray(y).reshape(n,-1)
    norm=max(np.linalg.norm(y-y.mean(0)),1e-12)
    avail=np.ones(pool.shape[1],dtype=bool)
    for _ in range(limit):
        a=affine(base[:n])
        e=y-a@ls(a,y)
        if np.linalg.norm(e) <= tolerance*norm:
            break
        f=pool[:n]
        f=f-a@ls(a,f)
        energy=np.sum(f*f,axis=0)
        score=np.sum((f.T@e)**2,axis=1)/np.maximum(energy,1e-18)
        score[~avail | (energy < 1e-14)]=-np.inf
        j=int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j]<.012*np.sum(e*e):
            break
        avail[j]=False
        base=np.c_[base,pool[:,j]]
    return base


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']
    t=s['train']; n=len(t['x']); x=np.concatenate((t['x'],s['test']['x']))
    z,p,r=spatial(s,x)
    zp,pr=outer(z,p),outer(p,r)
    core=np.c_[p,r,np.sum(p*r,axis=1),zp[:,[3,4,9,14]],pr[:,1]]
    pool=np.c_[zp,pr,z,outer(z,r),quadratic(z)]
    f=augment(core,pool,t['y'])
    y=ridge(f[:n],t['y'],f[n:],1e-7)
    common=augment(core,pool,t['traj'])
    offdiag=np.column_stack([pr[:,i] for i in range(16) if i//4!=i%4 and i!=1])
    return {'common':common,'offdiag':offdiag,'y_hat':y}
