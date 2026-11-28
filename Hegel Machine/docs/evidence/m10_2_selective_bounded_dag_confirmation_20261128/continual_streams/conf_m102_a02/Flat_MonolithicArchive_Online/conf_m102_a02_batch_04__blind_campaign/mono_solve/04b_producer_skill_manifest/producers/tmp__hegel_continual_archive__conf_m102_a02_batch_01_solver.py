"""Robust coordinate calibration and a two-mode relaxation model."""
import numpy as np
from scipy.optimize import least_squares


def _calibration(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    # Median pairwise slopes initialize the fit without giving outliers leverage.
    a, b = np.triu_indices(len(u), 1)
    du = u[a] - u[b]
    dv = v[a] - v[b]
    slope = np.nanmedian(np.divide(dv, du, out=np.full_like(du, np.nan), where=abs(du)>1e-10), axis=0)
    shift = np.median(v - u*slope, axis=0)
    for j in range(u.shape[1]):
        r = v[:,j] - u[:,j]*slope[j] - shift[j]
        keep = abs(r) <= max(1e-7, 6*np.median(abs(r)))
        if keep.sum() >= 3:
            shift[j], slope[j] = np.linalg.lstsq(np.c_[np.ones(keep.sum()), u[keep,j]], v[keep,j], rcond=None)[0]
    return shift, slope


def _time_basis(t, z):
    def basis(r):
        return -np.expm1(-t[:,None]*r)
    def resid(r):
        B = basis(r)
        return (np.linalg.lstsq(B,z.T,rcond=None)[0].T @ B.T - z).ravel()
    r = least_squares(resid, [1.8,.4], bounds=([.8,.03],[8.,.79])).x
    return basis(r)


def _design(x, kind):
    if kind == 'gaussian':
        sat = x * np.exp(-x*x)
    else:
        sat = x / (1 + .75*x*x)
    i, j = np.triu_indices(x.shape[1], 1)
    a, b = x[:, i]*sat[:, j], sat[:, i]*x[:, j]
    core = np.c_[np.ones(len(x)), x, sat]
    extra = np.c_[x*x, x[:, i]*x[:, j], (a+b)/2, (a-b)/2,
                  sat[:, i]*sat[:, j], x*sat]
    return core, extra


def _ridge(train, test, response, penalty):
    core, extra = train
    a = np.c_[core, extra]
    q = np.c_[test[0], test[1]]
    reg = np.diag(np.r_[np.zeros(core.shape[1]),
                        np.full(extra.shape[1], np.sqrt(penalty))])
    coefs = np.linalg.lstsq(np.r_[a, reg],
                            np.r_[response, np.zeros((len(reg), response.shape[1]))],
                            rcond=None)[0]
    return q @ coefs


def solve_batch(study: dict) -> dict:
    tr = study['train']
    shift, slope = _calibration(study['panel_a'])
    x = (tr['x']-shift)/slope
    q = (study['test']['x']-shift)/slope
    B = _time_basis(np.asarray(study['eval_times']), np.asarray(tr['traj']))
    amp = np.linalg.lstsq(B, tr['traj'].T, rcond=None)[0].T
    y = np.zeros(len(q))
    z = np.zeros((len(q), len(B)))
    # Average two nearby response shapes because sparse panel measurements
    # distinguish the odd nonlinear response more reliably than its tails.
    for kind, weight in [('gaussian', .65), ('rational', .35)]:
        A, Q = _design(x, kind), _design(q, kind)
        y += weight * _ridge(A, Q, tr['y'][:, None], .1)[:, 0]
        z += weight * (_ridge(A, Q, amp, 1.) @ B.T)
    return {'y_hat':y, 'state_pred':z,
            'diagnostics_vec':np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}
