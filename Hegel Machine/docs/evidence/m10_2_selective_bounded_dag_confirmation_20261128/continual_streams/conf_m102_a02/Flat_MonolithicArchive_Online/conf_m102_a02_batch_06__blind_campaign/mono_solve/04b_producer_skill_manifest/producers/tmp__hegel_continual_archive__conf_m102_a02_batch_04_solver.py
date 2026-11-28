"""Recover calibrated coordinates, then fit separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    coeff = []
    for j in range(u.shape[1]):
        a, b = u[:, j], u[:, (j+1) % u.shape[1]]
        design = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
        target = v[:, j]
        fit = least_squares(lambda c: design @ c - target, [0, 1, .3, .3, .15], loss='soft_l1', f_scale=.001).x
        err = target - design @ fit
        keep = abs(err - np.median(err)) <= max(6*np.median(abs(err - np.median(err))), .0007)
        coeff.append(_ls(design[keep], target[keep]) if keep.sum() >= 8 else fit)
    return np.array(coeff).T


def _inverse(x, coeff):
    x = np.asarray(x, dtype=float)
    bias, slope, cross, bilinear, cubic = coeff
    z = (x - bias) / slope
    j = np.arange(x.shape[1]); k = (j+1) % len(j)
    for _ in range(35):
        nxt = z[:, k]
        err = bias + slope*z + (cross + bilinear*z)*nxt + cubic*z**3 - x
        if not err.size or abs(err).max() < 1e-11:
            break
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = slope + bilinear*nxt + 3*cubic*z*z
        jac[:, j, k] = cross + bilinear*z
        step = np.linalg.solve(jac, err[..., None])[..., 0]
        z -= np.clip(step, -.8, .8)
    return z


def _operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(x, p):
        return np.column_stack((_affine(x), -x*np.expm1(-p[0]*abs(x)**p[1])))
    def fit(p):
        f = design(u, p)
        return f, _ls(f, v)
    def residual(p):
        f, c = fit(p)
        return (f @ c - v).ravel()
    p = least_squares(residual, [1., 2.], bounds=([.1, 1.], [4., 4.])).x
    _, c = fit(p)
    return lambda x: design(x, p) @ c


def _scalar_basis(h):
    i, j = np.triu_indices(h.shape[1], 1)
    return np.column_stack((_affine(h), h[:, i]*h[:, j]))


def _predict(study):
    tr = study['train']
    calibration = _calibration(study['panel_a'])
    op = _operator(study['panel_b'])
    h = op(_inverse(tr['x'], calibration))
    q = op(_inverse(study['test']['x'], calibration))
    f = _scalar_basis(h)
    y = _scalar_basis(q) @ _ls(f, tr['y'])
    state = _affine(q) @ _ls(_affine(h), tr['traj'])
    return y, state


def solve_batch(study: dict) -> dict:
    y, state = _predict(study)
    last = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), last.mean(), last.std()]) if y.size else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
