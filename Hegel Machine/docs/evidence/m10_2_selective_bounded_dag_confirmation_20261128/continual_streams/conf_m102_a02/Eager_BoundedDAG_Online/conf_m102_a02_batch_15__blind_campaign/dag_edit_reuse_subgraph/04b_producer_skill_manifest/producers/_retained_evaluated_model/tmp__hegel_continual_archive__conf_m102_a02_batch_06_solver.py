"""Calibrated cyclic operator with separate scalar and temporal readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf


def _aff(a):
    return np.column_stack((np.ones(len(a)), a))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    # Each reference coordinate couples to its successor on the four-cycle.
    def block(a, i):
        p, q = a[:, i], a[:, (i + 1) % 4]
        return np.column_stack((np.ones(len(a)), p, q, p*q, p**3))
    blocks = [block(u, i) for i in range(4)]
    mask = np.ones(len(u), dtype=bool)
    for _ in range(8):
        coef = [_fit(a[mask], v[mask, i]) for i, a in enumerate(blocks)]
        err = np.linalg.norm(v - np.column_stack([a @ c for a, c in zip(blocks, coef)]), axis=1)
        med = np.median(err[mask]); mad = np.median(abs(err[mask] - med))
        updated = err <= max(0.002, med + 5 * 1.4826 * mad)
        if updated.sum() < 12 or np.array_equal(updated, mask):
            break
        mask = updated
    c = np.stack([_fit(a[mask], v[mask, i]) for i, a in enumerate(blocks)])
    def forward(a):
        b = np.roll(a, -1)
        return c[:, 0] + c[:, 1]*a + c[:, 2]*b + c[:, 3]*a*b + c[:, 4]*a**3
    def jac(a):
        i = np.arange(4)
        j = np.diag(c[:, 1] + c[:, 3]*np.roll(a, -1) + 3*c[:, 4]*a*a)
        j[i, (i+1)%4] = c[:, 2] + c[:, 3]*a
        return j
    return np.asarray([least_squares(lambda a: forward(a) - b, b, jac=jac, max_nfev=60).x for b in x]).reshape(-1, 4)


def _panel(panel, z):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    def design(a, w):
        return np.column_stack((_aff(a), erf(w*a)))
    def loss(w):
        a = design(u, w)
        return np.mean((a @ _fit(a, v) - v)**2)
    w = minimize_scalar(loss, bounds=(0.4, 3.0), method='bounded').x
    return design(z, w) @ _fit(design(u, w), v)


def _predict(z, g, train_idx, test_idx, y, traj):
    v = z * np.roll(g, 1, axis=1)
    a = _aff(np.column_stack((g, z, v)))
    q = g[:, :2] * g[:, 2:]
    b = np.column_stack((a, q))
    # Shrink supplemental terms using their scale after projecting out the core.
    extras = np.column_stack((z*g, z*np.roll(v, 2, axis=1), g*np.roll(g, 1, axis=1)))
    core = _fit(b[train_idx], np.column_stack((y, extras[train_idx])))
    r = y - b[train_idx] @ core[:, 0]
    e = extras - b @ core[:, 1:]
    scl = np.maximum(np.linalg.norm(e[train_idx], axis=0), 1e-8)
    e = e / scl
    w = np.linalg.solve(e[train_idx].T @ e[train_idx] + 2.0*np.eye(e.shape[1]), e[train_idx].T @ r)
    yp = b[test_idx] @ core[:, 0] + e[test_idx] @ w
    # State observations share smooth temporal modes; keep the dominant three.
    _, _, vt = np.linalg.svd(traj, full_matrices=False)
    modes = vt[:3]
    tp = (a[test_idx] @ _fit(a[train_idx], traj @ modes.T)) @ modes
    return yp, tp


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], float)
    test = np.asarray(study['test']['x'], float)
    z = _calibrate(study['panel_a'], np.concatenate((x, test)))
    g = _panel(study['panel_b'], z)
    y, t = _predict(z, g, np.arange(len(x)), np.arange(len(x), len(z)), np.asarray(train['y']), np.asarray(train['traj']))
    diagnostic = np.array([y.mean(), y.std(), t[:, -1].mean(), t[:, -1].std()])
    return {'y_hat': y, 'state_pred': t, 'diagnostics_vec': diagnostic}
