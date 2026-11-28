"""Calibrate the raw coordinates and fit the two panel-C readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _with_bias(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_basis(x):
    return np.stack((np.ones_like(x), x, x**3,
                     (x - 1) * np.roll(x, -1, axis=1)), axis=2)


def _calibrate(panel, raw):
    u, v = panel['u'], panel['v']
    affine = _with_bias(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        residual = np.linalg.norm(affine @ _fit(affine[keep], v[keep]) - v, axis=1)
        updated = residual < 5 * np.median(residual)
        if np.array_equal(keep, updated):
            break
        keep = updated
    basis = _calibration_basis(u)

    def fit_mask(mask):
        return np.stack([_fit(basis[mask, j], v[mask, j]) for j in range(4)])

    for _ in range(5):
        weights = fit_mask(keep)
        residual = np.linalg.norm((basis * weights).sum(axis=2) - v, axis=1)
        updated = residual < max(.0006, 5 * np.median(residual))
        if np.array_equal(keep, updated):
            break
        keep = updated
    offset, slope, cubic, coupling = fit_mask(keep).T
    x = (raw - offset) / slope
    cols = np.arange(4)
    jacobian = np.zeros((len(x), 4, 4))
    for _ in range(35):
        nxt = np.roll(x, -1, axis=1)
        residual = offset + slope*x + cubic*x**3 + coupling*(x-1)*nxt - raw
        if np.max(np.abs(residual)) < 1e-12:
            break
        jacobian[:, cols, cols] = slope + 3*cubic*x**2 + coupling*nxt
        jacobian[:, cols, (cols+1) % 4] = coupling*(x-1)
        step = np.linalg.solve(jacobian, residual[..., None])[..., 0]
        x -= np.clip(step, -.4, .4)
    return x


def _operator(panel, x, branch):
    u, v = panel['u'], panel[branch]
    if branch == 'v1':
        def basis(a, p):
            return np.column_stack((_with_bias(a), a/(1+p[0]*np.abs(a)), np.sin(p[1]*a)))
        start, bounds = (.8, 4.5), ((.05, .1), (5., 9.))
    else:
        def basis(a, p):
            return np.column_stack((_with_bias(a), np.cos(p[0]*a), np.cos(p[1]*a)))
        start, bounds = (1.1, 3.9), ((.3, 1.5), (2., 8.))

    def residual(p):
        a = basis(u, p)
        return (a @ _fit(a, v) - v).ravel()

    p = least_squares(residual, start, bounds=bounds, gtol=1e-12).x
    return basis(x, p) @ _fit(basis(u, p), v)


def _design(study):
    raw = np.concatenate((study['train']['x'], study['test']['x']))
    x = _calibrate(study['panel_a'], raw)
    c = _operator(study['panel_c'], x, 'v1')
    d = _operator(study['panel_c'], x, 'v2')
    matched = (c*d).sum(axis=1)
    scalar = np.column_stack((_with_bias(c), d, matched))
    fast = np.column_stack((c, matched))
    slow = np.column_stack((d, c[:, 0]*d[:, 1]))
    return scalar, fast, slow, x


def _trajectory(fast, slow, t, targets):
    """Fit an exponential fast response and a rational slow response jointly."""
    n, width = len(targets), len(t)
    values = targets.ravel()

    def design(f, g, rates):
        rise = -np.expm1(-rates[0]*t)
        tail = t/(1 + rates[1]*t)
        return np.column_stack(((f[:, None, :]*rise[None, :, None]).reshape(len(f)*width, f.shape[1]),
                                (g[:, None, :]*tail[None, :, None]).reshape(len(g)*width, g.shape[1])))

    def residual(rates):
        a = design(fast[:n], slow[:n], rates)
        return a @ _fit(a, values) - values

    rates = least_squares(residual, [1.8, .45], bounds=([.05, .001], [6., 2.]), gtol=1e-10).x
    coeff = _fit(design(fast[:n], slow[:n], rates), values)
    return (design(fast[n:], slow[n:], rates) @ coeff).reshape(len(fast)-n, width)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    scalar, fast, slow, _ = _design(study)
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    state = _trajectory(fast, slow, study['eval_times'], train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
