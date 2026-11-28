import itertools
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _powers(d):
    return np.array([np.bincount(c, minlength=d) for k in range(4)
                     for c in itertools.combinations_with_replacement(range(d), k)], dtype=int)


def _design(x, powers):
    return np.prod(x[:, None, :] ** powers[None, :, :], axis=2)


def _calibration(u, v):
    powers = _powers(u.shape[1]); a = _design(u, powers)
    b = a[:, :u.shape[1]+1]
    mask = np.ones(len(u), dtype=bool)
    for _ in range(6):
        e = np.linalg.norm(b @ _ls(b[mask], v[mask]) - v, axis=1)
        keep = e <= max(4 * np.median(e[mask]), 1e-8)
        if keep.sum() < len(powers) or np.array_equal(mask, keep):
            break
        mask = keep
    w = _ls(a[mask], v[mask]); sparse = np.zeros_like(w)
    for j in range(v.shape[1]):
        sel = abs(w[:, j]) > max(0.002, .003 * np.max(abs(w[:, j])))
        sparse[sel, j] = _ls(a[mask][:, sel], v[mask, j])
    if np.linalg.norm(a[mask] @ sparse-v[mask]) <= max(.003, 2*np.linalg.norm(a[mask] @ w-v[mask])):
        w = sparse
    return powers, w


def _inverse(x, powers, w):
    d = x.shape[1]; q = _ls(w[1:d+1].T, (x-w[0]).T).T
    active = np.any(w != 0, axis=1); powers, w = powers[active], w[active]
    def jac(z):
        return np.stack([_design(z, np.maximum(powers-np.eye(d,dtype=int)[j],0)) @ (w*powers[:,j,None]) for j in range(d)], axis=2)
    for i in range(len(x)):
        obs = x[i]; guesses = (q[i], np.clip(q[i],-1,1), np.zeros(d))
        best, err = q[i], np.inf
        for start in guesses:
            fit = least_squares(lambda z: _design(z[None,:],powers)[0] @ w - obs, start,
                                jac=lambda z:jac(z[None,:])[0], max_nfev=75, gtol=1e-11)
            e = np.linalg.norm(fit.fun)
            if e < err:
                best, err = fit.x, e
            if err < 1e-9:
                break
        q[i] = best
    return q


def _operator(u, v, x, j):
    def des(z,k):
        h = z/(1+k*abs(z)) if j==1 else z*z/(1+k*z*z)
        return np.column_stack((np.ones(len(z)),z,h))
    def err(k):
        a=des(u,k[0]); return (a@_ls(a,v)-v).ravel()
    k=least_squares(err,[.8 if j==1 else .16],bounds=(.01,5.)).x[0]
    return des(x,k) @ _ls(des(u,k),v)


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']; a=study['panel_a']; c=study['panel_c']
    x=np.concatenate((train['x'],study['test']['x']))
    q=_inverse(x,*_calibration(a['u'],a['v']))
    p,r=(_operator(c['u'],c[f'v{j}'],q,j) for j in (1,2))
    products=(p[:,:,None]*r[:,None,:]).reshape(len(x),-1)
    design=np.column_stack((np.ones(len(x)),p,r,products))
    n=len(train['x']); yt=train['y']; y=design[n:] @ _ls(design[:n],yt)
    return {'train_features':design[:n], 'test_features':design[n:], 'regression':y}
