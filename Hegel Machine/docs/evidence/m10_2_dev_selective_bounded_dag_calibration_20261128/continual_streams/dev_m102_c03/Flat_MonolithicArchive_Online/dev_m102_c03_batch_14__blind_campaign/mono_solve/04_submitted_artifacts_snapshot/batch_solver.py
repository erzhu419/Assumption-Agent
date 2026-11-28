"""Calibrate the four inputs and fit separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _pairs(x):
    i, j = np.triu_indices(x.shape[1], 1)
    return x[:, i] * x[:, j]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    a = _aff(u)
    err = np.linalg.norm(v - a @ _ls(a, v), axis=1)
    keep = err < 3 * np.median(err)

    def design(z, rates):
        return np.stack((np.ones_like(z), z, np.tanh(rates[0]*z),
                         np.sin(rates[1]*np.roll(z, 2, axis=1))), axis=-1)

    def fit(rates):
        d = design(u[keep], rates)
        b = np.column_stack([_ls(d[:, j, :], v[keep, j]) for j in range(u.shape[1])])
        return (np.einsum('nki,ik->nk', d, b)-v[keep]).ravel(), b

    rates = least_squares(lambda r: fit(r)[0], [1.9, 2.3],
                          bounds=([.5, 1.4], [4., 3.5]), gtol=1e-12).x
    coef = fit(rates)[1]
    return np.array([least_squares(lambda z: np.einsum('ki,ik->k', design(z[None, :], rates)[0], coef)-row,
                                   row, gtol=1e-11).x for row in x])


def _project(u, v, x, design, bounds):
    def loss(rate):
        a = design(u, rate)
        q = np.linalg.qr(a, mode='reduced')[0]
        residual = v-q@(q.T@v)
        return np.mean((residual/np.maximum(1e-8, 1-np.sum(q*q, axis=1))[:, None])**2)

    rate = minimize_scalar(loss, method='bounded', bounds=bounds).x
    return design(x, rate) @ _ls(design(u, rate), v)


def _first(x, r):
    return np.column_stack((_aff(np.tanh(r*x)), x, x**3))


def _second(x, r):
    w = r*x
    return np.column_stack((_aff(w/np.sqrt(1+w*w)), _pairs(x)))


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    p = study['panel_c']
    a = _project(p['u'], p['v1'], z, _first, (.2, 3.))
    b = _project(p['u'], p['v2'], z, _second, (.2, 3.))
    return a, b


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x = np.asarray(study['test']['x'])
    n = len(tr['x'])
    if not len(x):
        return dict(y_hat=np.empty(0), state_pred=np.empty((0, len(study['eval_times']))), diagnostics_vec=np.zeros(4))
    a, b = _features(study, np.vstack((tr['x'], x)))
    scalar = _aff(np.column_stack((a, b, np.sum(a*b, axis=1))))
    cross = (a[:, :, None]*b[:, None, :]).reshape(len(a), -1)
    states = _aff(np.column_stack((a, b, cross)))
    modes = np.linalg.svd(tr['traj'], full_matrices=False)[2][:2]
    y_hat = scalar[n:] @ _ls(scalar[:n], tr['y'])
    state_pred = (states[n:] @ _ls(states[:n], tr['traj']@modes.T)) @ modes
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics)
