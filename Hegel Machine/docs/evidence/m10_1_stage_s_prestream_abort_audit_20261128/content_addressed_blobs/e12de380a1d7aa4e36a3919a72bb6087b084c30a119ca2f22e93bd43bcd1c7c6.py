from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _poly(x, degree):
    return np.column_stack([
        np.prod(x[:, cols], axis=1)
        for d in range(degree + 1)
        for cols in combinations_with_replacement(range(x.shape[1]), d)
    ])


def _kernel(x, w, kind, degree, scale):
    if kind == 'dot':
        smooth = np.exp((x @ w.T) / (4 * scale))
    else:
        d2 = cdist(x, w, 'sqeuclidean') / scale**2
        if kind == 'gauss':
            smooth = np.exp(-d2 / 2)
        elif kind == 'rq':
            smooth = (1 + d2 / 2)**-2
        else:
            r = np.sqrt(5 * d2)
            smooth = (1 + r + r*r/3) * np.exp(-r)
    return smooth + 100 * (_poly(x, degree) @ _poly(w, degree).T)


def _predict(x, w, target, spec):
    kind, degree, scale, ridge = spec
    cov = _kernel(x, x, kind, degree, scale)
    cov.flat[::len(x) + 1] += ridge
    return _kernel(w, x, kind, degree, scale) @ solve(cov, target, assume_a='pos')


def _trajectory_ridges(x, w, scores):
    # Profile the quadratic trend and linear ridge amplitudes out of the fit.
    trend = _poly(x, 2)
    q = np.linalg.qr(trend, mode='reduced')[0]
    residual = scores - q @ (q.T @ scores)
    penalty = 1.e-4

    def features(a, params):
        p = params.reshape(3, 5)
        return np.tanh(a @ p[:, :4].T + p[:, 4])

    def profile(params):
        h = features(x, params)
        g = h - q @ (q.T @ h)
        coef = np.linalg.solve(g.T @ g + penalty * np.eye(3), g.T @ residual)
        return h, g, coef

    def objective(params):
        _, g, coef = profile(params)
        return np.r_[(g @ coef - residual).ravel(),
                     np.sqrt(penalty) * coef.ravel(), .002 * params]

    starts = np.random.default_rng(21).normal(0, .55, (8, 15))
    best = min((least_squares(objective, start, max_nfev=120) for start in starts),
               key=lambda result: result.cost)
    h, _, coef = profile(best.x)
    beta = np.linalg.lstsq(trend, scores - h @ coef, rcond=None)[0]
    return _poly(w, 2) @ beta + features(w, best.x) @ coef


def solve_batch(study):
    data = study['train']
    x = np.asarray(data['x'], dtype=float)
    w = np.asarray(study['test']['x'], dtype=float)
    y = np.asarray(data['y'], dtype=float)
    z = np.asarray(data['traj'], dtype=float)
    modes = np.linalg.svd(z, full_matrices=False)[2][:2]
    y_hat = .65 * _predict(x, w, y, ('gauss', 2, 2., .0001)) + .35 * _predict(x, w, y, ('rq', 2, 4., .0001))
    scores = z @ modes.T
    smooth = (.6 * _predict(x, w, scores, ('m52', 1, 8., 1.e-6))
              + .4 * _predict(x, w, scores, ('dot', 1, 1., .003)))
    state_pred = (.5 * smooth + .5 * _trajectory_ridges(x, w, scores)) @ modes
    last = state_pred[:, -1]
    return dict(y_hat=y_hat, state_pred=state_pred,
                diagnostics_vec=np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]))
