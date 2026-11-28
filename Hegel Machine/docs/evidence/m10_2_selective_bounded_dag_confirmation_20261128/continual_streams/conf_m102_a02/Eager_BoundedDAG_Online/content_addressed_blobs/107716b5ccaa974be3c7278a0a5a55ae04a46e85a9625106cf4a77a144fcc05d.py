"""Calibrate the measured coordinates and fit separate endpoint and trajectory readouts."""
import numpy as np
from scipy.optimize import minimize_scalar


def _calibration(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    delta = u[j] - u[i]
    slopes = np.divide(v[j] - v[i], delta, out=np.full_like(delta, np.nan), where=np.abs(delta) > 1e-8)
    scale = np.nanmedian(slopes, axis=0)
    offset = np.median(v - scale * u, axis=0)
    for k in range(u.shape[1]):
        e = v[:, k] - offset[k] - scale[k] * u[:, k]
        keep = np.abs(e) < max(.025, 6 * np.median(np.abs(e)))
        offset[k], scale[k] = np.linalg.lstsq(np.c_[np.ones(keep.sum()), u[keep, k]], v[keep, k], rcond=None)[0]
    return offset, scale


def _panel(panel, key, op, bounds):
    u, v = panel['u'], panel[key]
    def design(x, k):
        return np.c_[np.ones(len(x)), x, op(k*x)]
    def residual(k):
        a = design(u, k)
        return np.square(a @ np.linalg.lstsq(a, v, rcond=None)[0] - v).sum()
    k = minimize_scalar(residual, bounds=bounds, method='bounded').x
    a = design(u, k)
    c = np.linalg.lstsq(a, v, rcond=None)[0]
    return lambda x: design(x, k) @ c, lambda x: op(k*x)


def _quad(x):
    i, j = np.triu_indices(x.shape[1])
    return x[:, i] * x[:, j]


def _ridge(x, w, y, lam):
    m, s = x.mean(0), np.maximum(x.std(0), 1e-8)
    a = (x-m)/s
    ym = y.mean(0)
    c = np.linalg.solve(a.T @ a + lam*np.eye(a.shape[1]), a.T @ (y-ym))
    return ym + ((w-m)/s) @ c


def _features(study, raw):
    offset, scale = _calibration(study['panel_a'])
    z = np.concatenate((raw['a'][:, 1:], raw['a_test'][:, 1:]))
    z = (z-offset)/scale
    bp, h = _panel(study['panel_b'], 'v', np.sin, (2., 3.8))
    cp, _ = _panel(study['panel_c'], 'v1', np.arctan, (1., 4.))
    dp, g = _panel(study['panel_c'], 'v2', lambda x: np.exp(-x*x), (.2, .9))
    b, c, d, h, g = bp(z), cp(z), dp(z), h(z), g(z)-1
    return z, b, c, d, h, g


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    z, b, c, d, h, g = _features(study, raw)
    y, traj = study['train']['y'], study['train']['traj']
    n = len(y)
    pred = []
    for f, lam in [(np.c_[b, _quad(d), _quad(h), b*c], .1),
                   (np.c_[b, _quad(d), _quad(h), _quad(z)], .01),
                   (np.c_[b, _quad(b), d], .025)]:
        pred.append(_ridge(f[:n], f[n:], y, lam))
    endpoint = np.average(pred, axis=0, weights=[.60, .25, .15])
    f = np.c_[z, b, c, np.sin(3*z)]
    state = _ridge(f[:n], f[n:], traj, .05)
    values = np.column_stack((endpoint, state))
    return {'test_features': values, 'coefficients': np.eye(values.shape[1])}
