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


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    q = _invert(np.vstack([train['x'], study['test']['x']]),
                _fit_calibration(study['panel_a']))
    b = _panel_b(study['panel_b'], q)
    c = b * np.roll(q, -1, axis=1)
    h = q[:, 2]
    state_design = np.column_stack([np.ones(len(q)), b, c.sum(axis=1)])
    # Telemetry supplies the strength of the coordinate-2 scalar gate.
    gate = 1.0 / (1.0 + float(study['telemetry_vec'][2]) * h**2)
    scalar_design = np.column_stack([np.ones(len(q)), b * gate[:, None], c,
                                     b[:, 3] * h**3 * gate])
    y = scalar_design[n:] @ _ls(scalar_design[:n], train['y'])
    state = state_design[n:] @ _ls(state_design[:n], train['traj'])
    diag = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diag}
