"""Calibrate inputs, compose the measured operators, and fit two readouts."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    a = np.column_stack((np.ones(len(u)), u))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        c = _ls(a[keep], v[keep])
        r = np.linalg.norm(a @ c - v, axis=1)
        take = r < max(.001, 3 * np.median(r))
        if take.sum() < a.shape[1] or np.array_equal(take, keep):
            return c
        keep = take
    return _ls(a[keep], v[keep])


def _operator(u, v, fn):
    def design(w, q):
        return np.column_stack((np.ones(len(w)), w, fn(w * np.exp(q))))

    def objective(q):
        d = design(u, q)
        return (d @ _ls(d, v) - v).ravel()

    fits = [least_squares(objective, np.full(u.shape[1], np.log(a)),
                          bounds=(-2, 2), max_nfev=100)
            for a in (.65, 1.4, 2.5, 3.8)]
    q = min(fits, key=lambda f: f.cost).x
    c = _ls(design(u, q), v)
    return lambda w: design(w, q) @ c


def _ridge(d, target, lam, n):
    m = d[:n].mean(0)
    scale = np.maximum(d[:n].std(0), 1e-8)
    w = (d - m) / scale
    a = w[:n]
    ym = target.mean(0)
    gram = a.T @ a + lam * np.eye(a.shape[1])
    return w[n:] @ np.linalg.solve(gram, a.T @ (target - ym)) + ym


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    cal = _calibration(study['panel_a'])
    x = np.vstack((train['x'], test['x']))
    z = np.linalg.solve(cal[1:].T, (x - cal[0]).T).T
    pb, pc = study['panel_b'], study['panel_c']
    b = _operator(pb['u'], pb['v'], erf)
    c = _operator(pc['u'], pc['v1'], np.arcsinh)
    r, h = c(z), c(b(z))
    dy = np.column_stack((z, h, (z * h).sum(1)))
    ds = np.column_stack((z, r, h, z * r, z * h, r * h))
    t = np.asarray(train['traj'], dtype=float)
    modes = np.linalg.svd(t, full_matrices=False)[2][:2]
    y = _ridge(dy, np.asarray(train['y']), .0003, n)
    amplitudes = t @ modes.T
    cubic = np.column_stack((ds, (h**3).sum(1)))
    projected = _ridge(cubic, amplitudes, .003, n)
    state = projected @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
