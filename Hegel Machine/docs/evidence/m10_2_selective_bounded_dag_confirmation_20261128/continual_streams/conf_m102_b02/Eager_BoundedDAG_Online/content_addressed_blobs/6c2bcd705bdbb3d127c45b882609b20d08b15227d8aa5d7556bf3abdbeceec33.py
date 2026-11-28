"""Invert panel A and estimate the folding map measured by panel B."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a = study['panel_a']
    design, observed = _design(a['u']), np.asarray(a['v'], dtype=float)
    keep = np.ones(len(design), dtype=bool)
    for _ in range(6):
        coef = _fit(design[keep], observed[keep])
        error = np.linalg.norm(design @ coef - observed, axis=1)
        proposed = error < max(0.001, 3 * np.median(error))
        if proposed.sum() < design.shape[1] or np.array_equal(keep, proposed):
            break
        keep = proposed
    coef = _fit(design[keep], observed[keep])
    train_x = study['train']['x']
    x = np.vstack((train_x, study['test']['x']))
    z = np.linalg.solve(coef[1:].T, (x - coef[0]).T).T

    panel = study['panel_b']
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def basis(q, points):
        return _design(erf(q[0] * points) + q[1] * points)

    def residual(q):
        a = basis(q, u)
        return (a @ _fit(a, v) - v).ravel()

    fits = [least_squares(residual, [s, -0.5*s], bounds=([0.1, -8], [4, 8]),
                         max_nfev=100) for s in (0.4, 0.8, 1.4, 2.2)]
    q = min(fits, key=lambda sol: sol.cost).x
    b = basis(q, z) @ _fit(basis(q, u), v)
    n = len(train_x)
    return {'z_train': z[:n], 'z_test': z[n:], 'b_train': b[:n], 'b_test': b[n:]}
