"""Pair-calibrated panel-B features with separate scalar and trajectory readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, root


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




def _paired_design(q, frequency):
    z = np.roll(q, -2, axis=1)
    return np.stack([np.ones_like(q), q, np.sin(frequency*z), z,
                     q*q, q**3, z*z, q*z], axis=-1)


def _paired_coordinates(x, panel, frequency):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    initial = _invert(x, _fit_calibration(panel))
    delta = np.linalg.norm(v-u, axis=1)
    keep = delta < 2.
    if np.count_nonzero(keep) < 16:
        return initial
    design = _paired_design(u[keep], frequency)
    w = np.array([_ls(design[:, j], v[keep, j]) for j in range(u.shape[1])])

    def invert(row, start):
        def residual(q):
            return (_paired_design(q[None, :], frequency)[0] * w).sum(axis=-1) - row
        fit = least_squares(residual, start, max_nfev=35)
        return fit.x if np.linalg.norm(fit.fun) < 1e-6 else start

    return np.array([invert(row, start) for row, start in zip(x, initial)]).reshape(initial.shape)


def _panel_features(q, upstream):
    scale, power, exponent = upstream['p']
    logabs = np.log(np.maximum(np.abs(q), np.finfo(float).tiny))
    h = q * np.exp(exponent * np.logaddexp(0, power * (np.log(scale) + logabs)))
    return np.c_[np.ones(len(q)), q, h] @ upstream['w']


def _ridge(x, y, new, alpha):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .02)
    a = (x - mean) / scale
    y_mean = y.mean(axis=0)
    coef = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y - y_mean))
    return ((new - mean) / scale) @ coef + y_mean


def run_node(inputs: dict) -> dict:
    study, upstream = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    q = _paired_coordinates(np.vstack([train['x'], study['test']['x']]),
                            study['panel_a'], float(study['telemetry_vec'][0]))
    b = _panel_features(q, upstream)
    scalar = np.c_[b, q, _design(b, 15)[:, 5:]]
    cross_stage = np.sum(b * np.roll(q, -1, axis=1), axis=1)
    state = np.c_[b, cross_stage]
    yh = _ridge(scalar[:n], train['y'], scalar[n:], .0025)
    sp = _ridge(state[:n], train['traj'], state[n:], 1e-6)
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])}
