"""Calibrated cosine operator with separate scalar and temporal readouts."""
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
    from scipy.optimize import minimize_scalar
    u, v = [np.asarray(panel[k], dtype=float) for k in ('u', 'v')]
    def design(z, p):
        return np.column_stack((_affine(z), np.cos(p*z)))
    def loss(p):
        a = design(u, p)
        return np.mean((a @ _ls(a, v) - v)**2)
    p0 = min(np.linspace(.15, 6., 120), key=loss)
    p = minimize_scalar(loss, bounds=(max(.03, p0-.1), p0+.1), method='bounded').x
    w = _ls(design(u, p), v)
    return lambda z: design(z, p) @ w


def _scalar(h, q, y, mode='full'):
    a, aq = _affine(h), _affine(q)
    j, k = np.triu_indices(h.shape[1]) if mode != 'sparse' else (np.array([0, 1]), np.array([2, 3]))
    d, dq = np.c_[a, h[:, j]*h[:, k]], np.c_[aq, q[:, j]*q[:, k]]
    rng = np.random.default_rng(42)
    starts = np.r_[np.eye(5)*.45, np.c_[np.zeros(4), -.45*np.eye(4)], rng.normal(0,.45,(6,5))]
    def design(b, x, p):
        w = x @ p
        if mode == 'rational':
            g = -(w*w)/(1+w*w)
            return np.c_[b, g[:, None]*x]
        return np.c_[b, np.sin(w)]
    def residual(p):
        z = design(d, a, p)
        return z @ _ls(z, y) - y
    fit = min((least_squares(residual, p, bounds=(-4,4), max_nfev=130) for p in starts), key=lambda x:x.cost)
    return design(dq, aq, fit.x) @ _ls(design(d, a, fit.x), y)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    op = _operator(study['panel_b'])
    calibration = _calibration(study['panel_a'])
    z = op(_inverse(np.vstack((tr['x'], study['test']['x'])), calibration))
    h, q = np.split(z, [len(tr['x'])])
    y = _scalar(h, q, np.asarray(tr['y']))
    state = _affine(q) @ _ls(_affine(h), tr['traj'])
    last = state[:, -1]
    diag = np.array([y.mean(), y.std(), last.mean(), last.std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diag}
