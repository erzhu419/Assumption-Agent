"""Predict the two readouts of the calibrated panel-B / panel-C1 path."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    result = np.empty_like(x, dtype=float)
    for j in range(u.shape[1]):
        a = _affine(u[:, j:j+1])
        w = _fit(a, v[:, j])
        for _ in range(8):
            keep = np.argsort(abs(a @ w - v[:, j]))[:3*len(u)//4]
            w = _fit(a[keep], v[keep, j])
        result[:, j] = (x[:, j] - w[0]) / w[1]
    return result


def _coordinates(study, x):
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    z = _affine(z) @ _fit(_affine(b['u']), b['v'])

    def design(a, k):
        return _affine(a / (1 + k*abs(a))**2)

    def objective(k):
        d = design(c['u'], k)
        return np.mean((d @ _fit(d, c['v1']) - c['v1'])**2)

    k = minimize_scalar(objective, bounds=(0.01, 5), method='bounded').x
    return design(z, k) @ _fit(design(c['u'], k), c['v1'])


def _scalar(q, y, n):
    z = (q - q[:n].mean(axis=0)) / q[:n].std(axis=0)
    i, j = np.triu_indices(z.shape[1])
    design = np.column_stack((_affine(z), z[:, i]*z[:, j]))
    a = design[:n]
    penalty = 0.045*np.maximum(a.std(axis=0), 0.06)
    penalty[0] = 0
    # Project out the regularized quadratic block while fitting the tanh direction.
    basis = np.linalg.qr(np.vstack((a, np.diag(penalty))), mode='reduced')[0]
    target = np.r_[y, np.zeros(len(penalty))]
    residual = target - basis @ (basis.T @ target)

    def objective(p):
        col = np.r_[np.tanh(z[:n] @ p[:-1] + p[-1]), np.zeros(len(penalty))]
        col -= basis @ (basis.T @ col)
        return np.r_[residual - col*((col @ residual)/(col @ col + 0.03**2)), 0.005*p]

    d = z.shape[1]
    rng = np.random.default_rng(123)
    starts = [np.r_[1.2*e, 0.] for e in np.eye(d)]
    starts += [np.r_[rng.normal(0, 0.7, d), rng.normal(0, 0.2)] for _ in range(6)]
    fits = (least_squares(objective, p, bounds=(-5, 5), max_nfev=70) for p in starts)
    p = min(fits, key=lambda f: f.cost).x
    matrix = np.column_stack((design, np.tanh(z @ p[:-1] + p[-1])))
    coeff = _fit(np.vstack((matrix[:n], np.diag(np.r_[penalty, 0.03]))), np.r_[target, 0.])
    return matrix[n:] @ coeff


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    q = _coordinates(study, np.concatenate((train['x'], study['test']['x'])))
    y = _scalar(q, train['y'], n)
    d = _affine(q)
    state = d[n:] @ _fit(d[:n], train['traj'])
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
