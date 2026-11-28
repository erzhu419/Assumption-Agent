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


def _b_features(q, p):
    scale, power, exponent = p
    logabs = np.log(np.maximum(np.abs(q), np.finfo(float).tiny))
    h = q * np.exp(exponent * np.logaddexp(0, power * (np.log(scale) + logabs)))
    return np.c_[np.ones(len(q)), q, h]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    operator = inputs['node_00']
    train = study['train']
    coef = _fit_calibration(study['panel_a'])
    p, w = operator['p'], operator['w']

    def designs(x):
        b = _b_features(_invert(x, coef), p) @ w
        linear = np.c_[np.ones(len(b)), b]
        return linear, np.c_[linear, b[:, 0] * b[:, 2], b[:, 1] * b[:, 3]]

    a, f = designs(train['x'])
    at, ft = designs(study['test']['x'])
    y = ft @ _ls(f, train['y'])
    state = at @ _ls(a, train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
