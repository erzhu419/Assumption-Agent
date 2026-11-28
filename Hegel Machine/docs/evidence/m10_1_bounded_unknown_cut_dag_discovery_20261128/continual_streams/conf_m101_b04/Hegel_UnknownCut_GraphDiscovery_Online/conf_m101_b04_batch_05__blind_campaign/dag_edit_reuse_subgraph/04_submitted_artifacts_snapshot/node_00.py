import numpy as np
from scipy.optimize import least_squares, root


def _fit(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    neighbor = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, (u - 1.) * neighbor, u**3), axis=-1)
    weights = np.ones(len(u))

    def fit(rows, w):
        return np.array([
            np.linalg.lstsq(design[rows, j] * w[:, None], v[rows, j] * w, rcond=None)[0]
            for j in range(u.shape[1])
        ]).T

    rows = np.arange(len(u))
    for _ in range(20):
        coef = fit(rows, weights)
        error = np.linalg.norm(np.einsum('njk,kj->nj', design, coef) - v, axis=1)
        scale = max(np.median(error), 1.e-4)
        weights = 1. / np.hypot(1., error / scale)
    # The calibration protocol introduces exactly three contaminated rows.
    keep = np.argsort(error)[:-3]
    return fit(keep, np.ones(len(keep)))


def _invert(v, coef):
    bias, gain, cross, cubic = coef
    j = np.arange(len(bias))
    nxt = (j + 1) % len(j)
    linear = np.diag(gain)
    linear[j, nxt] = -cross

    def forward(x):
        return bias + gain * x + cross * (x - 1.) * x[nxt] + cubic * x**3

    def jac(x):
        jacobian = np.diag(gain + cross * x[nxt] + 3. * cubic * x**2)
        jacobian[j, nxt] = cross * (x - 1.)
        return jacobian

    seed = np.linalg.solve(linear, (v - bias).T).T
    result = np.empty_like(seed)
    for i, target in enumerate(v):
        def residual(x):
            return forward(x) - target

        start = np.clip(seed[i], -1.5, 1.5)
        sol = root(residual, start, jac=jac, tol=1.e-10)
        if np.linalg.norm(sol.fun, ord=np.inf) > 1.e-9:
            sol = least_squares(residual, start, jac=jac, gtol=1.e-12)
        result[i] = sol.x
    return result


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    coef = _fit(study['panel_a'])
    return {'x_tr': _invert(study['train']['x'], coef),
            'x_te': _invert(study['test']['x'], coef)}
