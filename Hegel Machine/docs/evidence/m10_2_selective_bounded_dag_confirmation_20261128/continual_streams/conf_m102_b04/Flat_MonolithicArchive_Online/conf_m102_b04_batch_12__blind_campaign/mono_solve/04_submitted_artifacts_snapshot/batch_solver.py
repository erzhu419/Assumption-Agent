"""Cyclic cubic calibration, cosine operator, and two-mode readout."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_aff(x), x[:, i] * x[:, j]))


def _cal_design(u, v):
    return np.column_stack((np.ones_like(u), u, v, u*v, u**3))


def _fit_calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    gap = np.linalg.norm(v-u, axis=1)
    keep = gap < 4.0 * np.median(gap)
    p = np.stack([_ls(_cal_design(u[keep, j], u[keep,(j+1)%4]), v[keep,j]) for j in range(4)])
    return p


def _inverse(x, p):
    a, b, c, d, e = p.T
    j = np.arange(4)
    def residual(z, target):
        nxt = np.roll(z, -1)
        return a + b*z + c*nxt + d*z*nxt + e*z**3 - target
    def jacobian(z, target):
        jac = np.diag(b + d*np.roll(z, -1) + 3*e*z*z)
        jac[j, (j+1)%4] = c + d*z
        return jac
    result = np.empty_like(x)
    for i, target in enumerate(x):
        result[i] = least_squares(residual, target, jac=jacobian, args=(target,),
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return result


def _fit_operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u','v'))
    def design(f):
        return _aff(np.cos(f*u))
    def residual(f):
        a = design(f[0])
        return (a @ _ls(a, v) - v).ravel()
    f = least_squares(residual, [2.6], bounds=(1.5, 4.)).x[0]
    return f, _ls(design(f), v)


def _trajectory(x, q, t, observed):
    # Fit shared rates after profiling the spatial readout coefficients.
    basis = lambda rates: -np.expm1(-np.outer(t, rates))
    fitted = x @ _ls(x, observed)
    def residual(rates):
        h = basis(rates)
        return (h @ _ls(h, fitted.T) - fitted.T).ravel()
    rates = least_squares(residual, [.42, 1.8], bounds=([.03, .9], [1.0, 5.0])).x
    h = basis(rates)
    coef = _ls(x, _ls(h, observed.T).T)
    return q @ coef @ h.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    p = _fit_calibration(study['panel_a'])
    f, w = _fit_operator(study['panel_b'])
    def features(x):
        return _aff(np.cos(f*_inverse(np.asarray(x, dtype=float), p))) @ w
    x, q = features(train['x']), features(study['test']['x'])
    y = _quadratic(q) @ _ls(_quadratic(x), np.asarray(train['y'], dtype=float))
    state = _trajectory(_aff(x), _aff(q), np.asarray(study['eval_times'], dtype=float),
                        np.asarray(train['traj'], dtype=float))
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
