"""Predict the dual-branch readout after inverting cyclic sensor calibration."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    rows = np.arange(4)
    nxt = (rows + 1) % 4

    def design(w):
        q = w[:, nxt]
        return np.stack((np.ones_like(w), w, q + w*q, w**3), axis=-1)

    a = design(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(12):
        co = np.column_stack([_fit(a[keep, j], v[keep, j]) for j in rows])
        err = np.linalg.norm(np.einsum('njk,kj->nj', a, co) - v, axis=1)
        new = err <= max(0.001, 3 * np.median(err))
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    # Pool the two shared nonlinear terms after the robust row screening.
    ident = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    g = np.concatenate((ident, ident*u[:, :, None], a[:, :, 2:]), axis=2)
    off, gain, common = np.split(_fit(g[keep].reshape(-1, 10), v[keep].ravel()), [4, 8])
    coupling, cubic = common
    lin = np.diag(gain)
    lin[rows, nxt] = coupling
    z0 = np.linalg.solve(lin, (x - off).T).T

    def forward(z):
        return off + gain*z + coupling*z[nxt]*(1+z) + cubic*z**3

    def jac(z):
        result = np.diag(gain + coupling*z[nxt] + 3*cubic*z*z)
        result[rows, nxt] = coupling*(1+z)
        return result

    return np.array([least_squares(lambda z: forward(z)-target, p, jac=jac,
                                   gtol=1e-12).x for target, p in zip(x, z0)])


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    u, v1, v2 = (np.asarray(study['panel_c'][k], dtype=float) for k in ('u', 'v1', 'v2'))

    def cycle(q):
        return _aff(q * np.roll(q, -1, axis=1))

    c = cycle(z) @ _fit(cycle(u), v1)

    def cosine(q, rate):
        return _aff(np.cos(q*rate[0]))

    def residual(rate):
        a = cosine(u, rate)
        return (a @ _fit(a, v2) - v2).ravel()

    rate = least_squares(residual, [2.4], bounds=(1.5, 3.2), gtol=1e-12).x
    d = cosine(z, rate) @ _fit(cosine(u, rate), v2)
    cross = (c[:, :, None]*d[:, None, :]).reshape(len(z), 16)
    return z, c, d, cross


from itertools import combinations_with_replacement


def _prune(x, y, tolerance):
    keep = list(range(x.shape[1]))
    while len(keep) > 2:
        a = x[:, keep]
        b = _fit(a, y)
        residual = y-a@b
        rss = np.sum(residual**2)
        inv = np.diag(np.linalg.pinv(a.T@a, rcond=1e-12))
        increase = np.sum(b*b, axis=1)/np.maximum(inv, 1e-30)
        increase[0] = np.inf
        idx = np.argmin(increase)
        if rss+increase[idx] > y.size*tolerance**2:
            break
        del keep[idx]
    return keep


def _spatial(study, x):
    z,c,d,cross = _features(study,x)
    cube = np.column_stack([np.prod(z[:,p], axis=1) for p in combinations_with_replacement(range(4),3)])
    diag = np.arange(0,16,5)
    base = np.column_stack((_aff(z),c,d,cross[:,diag],cross[:,1],cube))
    return base,cross,diag


def solve_batch(study: dict) -> dict:
    train = study['train']; n=len(train['x']); t=study['eval_times']
    base,cross,diag = _spatial(study,np.vstack((train['x'],study['test']['x'])))
    y=np.asarray(train['y']); traj=np.asarray(train['traj'])
    keep=_prune(base[:n],y[:,None],0.0045)
    mat=base[:,keep]; y_hat=(mat[n:]@_fit(mat[:n],y))
    extra=np.setdiff1d(np.arange(16),np.r_[diag,1]); states=np.c_[base,cross[:,extra]]
    def modes(r):
        return -np.expm1(-t[:,None]*r)
    def tem(r):
        q=modes(r); return (q@_fit(q,traj.T)-traj.T).ravel()
    rates=least_squares(tem,[.4,1.8,4.7],bounds=([.15,.9,2.6],[.85,2.5,10]),max_nfev=150).x
    q=modes(rates); target=traj@np.linalg.qr(q)[0]
    options=[]
    for M in (base,states):
        for tol in (.003,.0036,.0042,.005,.006,.008,.01,.012,.016,.024):
            k=_prune(M[:n],target,tol)
            L=M[:n,k]; u=np.linalg.qr(L)[0]; resid=target-u@(u.T@target)
            loo=np.mean((resid/np.maximum(.05,1-(u*u).sum(axis=1))[:,None])**2)
            options.append((loo,M,k))
    _,M,k=min(options,key=lambda p:p[0]); a=M[:,k]
    state_pred=a[n:]@_fit(a[:n],target)@np.linalg.qr(q)[0].T
    last=state_pred[:,-1]
    return dict(y_hat=y_hat,state_pred=state_pred,diagnostics_vec=np.array([y_hat.mean(),y_hat.std(),last.mean(),last.std()]))
