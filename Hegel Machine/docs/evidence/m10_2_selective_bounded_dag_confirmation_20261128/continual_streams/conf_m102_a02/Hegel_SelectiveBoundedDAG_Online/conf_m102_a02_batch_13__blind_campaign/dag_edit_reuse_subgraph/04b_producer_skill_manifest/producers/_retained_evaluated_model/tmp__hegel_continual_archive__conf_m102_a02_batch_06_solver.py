"""Invert the reference-panel calibration and fit separate scalar and state readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import root


_TERMS = tuple(term for degree in range(4)
               for term in combinations_with_replacement(range(4), degree))


def _design(x, size):
    x = np.atleast_2d(x)
    return np.column_stack([np.prod(x[:, term], axis=1) for term in _TERMS[:size]])


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _fit_calibration(panel):
    u, v = panel['u'], panel['v']
    delta = np.linalg.norm(v - u, axis=1)
    med = np.median(delta)
    good = delta <= med + max(1., 8 * np.median(np.abs(delta - med)))
    f = _design(u[good], len(_TERMS))
    coef = _ls(f, v[good])
    # Refit the resolved monomials without the weakly identified dense terms.
    leverage = np.linalg.pinv(f)
    sigma = np.sqrt(np.sum((v[good] - f @ coef)**2, axis=0) / max(1, len(f) - f.shape[1]))
    cutoff = np.maximum(0.002, 6 * np.linalg.norm(leverage, axis=1)[:, None] * sigma)
    for j in range(v.shape[1]):
        keep = np.abs(coef[:, j]) >= cutoff[:, j]
        coef[~keep, j] = 0
        coef[keep, j] = _ls(f[:, keep], v[good, j])
    return coef


def _invert(x, coef):
    initial = np.linalg.solve(coef[1:5].T, (x - coef[0]).T).T
    def transform(row, start):
        residual = lambda q: (_design(q, len(_TERMS)) @ coef)[0] - row
        fit = root(residual, start, tol=1e-10)
        return fit.x if np.linalg.norm(residual(fit.x)) < 1e-7 else start
    return np.array([transform(row, start) for row, start in zip(x, initial)]).reshape(-1, 4)


def _panel_b(panel, q):
    from scipy.optimize import least_squares
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    def design(a, p):
        logabs = np.log(np.maximum(abs(a)*p[0], 1e-50))
        h = a*np.expm1(p[2]*np.logaddexp(0, p[1]*logabs))
        return np.c_[np.ones(len(a)),a,h]
    def residual(p):
        f=design(u,p)
        return (f@_ls(f,v)-v).ravel()
    p=least_squares(residual,[2.5,12,.036],bounds=([.1,1,.001],[10,40,2]),max_nfev=350).x
    f=design(u,p)
    return design(q,p)@_ls(f,v)


def _quad(a):
    return np.column_stack([a]+[a[:,i]*a[:,j] for i in range(a.shape[1]) for j in range(i,a.shape[1])])


def _ridge(a,b,new,alpha):
    mu=a.mean(0); scale=np.maximum(a.std(0),.1)
    f=(a-mu)/scale; v=b.mean(0)
    w=np.linalg.solve(f.T@f+alpha*np.eye(f.shape[1]),f.T@(b-v))
    return (new-mu)/scale@w+v


def _readouts(q,b,y,z):
    from sklearn.linear_model import Lasso
    n=len(y)
    full=_quad(np.c_[q,b])
    cyclic=b*np.roll(q,-1,axis=1)
    structural=np.c_[b,cyclic]
    fs=np.maximum(full[:n].std(0),.03)
    ys=Lasso(alpha=.001,max_iter=12000,tol=1e-7).fit(full[:n]/fs,y).predict(full[n:]/fs)
    yc=_ridge(structural[:n],y,structural[n:],1e-6)
    zc=_ridge(structural[:n],z,structural[n:],1e-6)
    return ys,yc,zc


def solve_batch(study: dict) -> dict:
    tr=study['train']
    q=_invert(np.vstack([tr['x'],study['test']['x']]),_fit_calibration(study['panel_a']))
    b=_panel_b(study['panel_b'],q)
    ys,yc,z=_readouts(q,b,tr['y'],tr['traj'])
    y=SCALAR_COMPACT_WEIGHT*yc+(1-SCALAR_COMPACT_WEIGHT)*ys
    return {'y_hat':y,'state_pred':z,'diagnostics_vec':np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}


SCALAR_COMPACT_WEIGHT = 0.25
