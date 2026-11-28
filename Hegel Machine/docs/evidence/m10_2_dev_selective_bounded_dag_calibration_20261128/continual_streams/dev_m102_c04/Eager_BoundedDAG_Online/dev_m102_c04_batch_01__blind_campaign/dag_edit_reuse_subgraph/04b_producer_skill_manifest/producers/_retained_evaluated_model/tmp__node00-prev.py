import numpy as np
from scipy.optimize import minimize_scalar
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
        return affine(2*q/(1+k*q)-q)
    def objective(k):
        f = basis(u,k)
        return np.linalg.norm(f @ ls(f,v2)-v2)
    grid = np.geomspace(.03,4,50)
    i = np.argmin([objective(k) for k in grid])
    k = minimize_scalar(objective,bounds=(grid[max(0,i-1)], grid[min(len(grid)-1,i+1)]),method='bounded').x
    r = basis(z,k) @ ls(basis(u,k),v2)
    return z,p,r


def ridge(f, target, query, penalty):
    fm, ym = f.mean(axis=0), target.mean(axis=0)
    a = f-fm
    return (query-fm) @ np.linalg.solve(a.T@a+penalty*np.eye(f.shape[1]),a.T@(target-ym))+ym


def candidates(x,z,p,r):
    pr = outer(p,r)
    core = np.c_[z,quadratic(z),r]
    both = np.c_[core,pr]
    yield np.c_[p,r,(p*r).sum(1)]
    yield np.c_[core,(p*r).sum(1)]
    yield both
    yield np.c_[both,outer(z,p),outer(z,r)]
    yield np.c_[both,quadratic(p),quadratic(r)]
    yield np.c_[both,outer(z,p),outer(z,r),quadratic(p),quadratic(r)]
    yield core
    yield np.c_[core,outer(z,r)]


def select(ff, y):
    folds = np.array_split(np.random.default_rng(138).permutation(len(y)), 4)
    best = None
    for f in ff:
        for alpha in (.00001,.003,.08,1.):
            err = []
            for te in folds:
                tr = np.setdiff1d(np.arange(len(y)),te)
                sigma = np.maximum(f[tr].std(0),1e-8)
                pred = ridge(f[tr]/sigma,y[tr],f[te]/sigma,alpha)
                err.append(np.mean(abs(pred-y[te])))
            loss = np.mean(err)
            if best is None or loss<best[0]:
                best = loss,f,alpha
    return best[1:]


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    train = s['train']
    n = len(train['x'])
    x = np.concatenate((train['x'],s['test']['x']))
    z,p,r = spatial(s,x)
    fs = tuple(candidates(x,z,p,r))
    fy, ay = select(tuple(f[:n] for f in fs),train['y'])
    j = next(i for i,f in enumerate(fs) if np.shares_memory(f,fy))
    f = fs[j]/np.maximum(fy.std(0),1e-8)
    y = ridge(f[:n],train['y'],f[n:],ay)
    ft, at = select(tuple(f[:n] for f in fs),train['traj'])
    j = next(i for i,f in enumerate(fs) if np.shares_memory(f,ft))
    f = fs[j]/np.maximum(ft.std(0),1e-8)
    # Scale features so the fixed sink penalty equals the selected ridge penalty.
    f = f*np.sqrt(1e-6/at)
    return {'common':f,'offdiag':np.zeros((len(x),1)),'y_hat':y}
