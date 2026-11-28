"""Calibrated bilinear readout and a two-rate exponential state model."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    nxt = np.roll(u, -1, axis=1)
    terms = [np.column_stack((np.ones(len(u)), u[:, j], nxt[:, j],
                              u[:, j] * nxt[:, j], u[:, j] ** 3))
             for j in range(u.shape[1])]
    w = np.ones(len(u))
    for _ in range(30):
        c = np.column_stack([_fit(d * w[:, None], v[:, j] * w)
                             for j, d in enumerate(terms)])
        r = np.linalg.norm(np.column_stack([d @ c[:, j] for j, d in enumerate(terms)]) - v, axis=1)
        w = np.sqrt(np.minimum(1., 1.5 * max(np.median(r), 1e-12) / np.maximum(r, 1e-12)))
    good = r <= 6 * max(np.median(r), 1e-12)
    return np.column_stack([_fit(d[good], v[good, j]) for j, d in enumerate(terms)])


def _inverse(x, c):
    idx = np.arange(c.shape[1])
    following = (idx + 1) % len(idx)
    linear = np.diag(c[1])
    linear[idx, following] = c[2]

    def model(q):
        return c[0] + c[1] * q + c[2] * q[following] + c[3] * q * q[following] + c[4] * q**3

    def jac(q):
        j = np.diag(c[1] + c[3] * q[following] + 3 * c[4] * q**2)
        j[idx, following] = c[2] + c[3] * q
        return j

    starts = np.linalg.solve(linear, (x - c[0]).T).T
    out = np.empty_like(starts)
    for i, (row, start) in enumerate(zip(x, starts)):
        result = least_squares(lambda q: model(q) - row, start, jac=jac, gtol=1e-12)
        out[i] = result.x
    return out


def _readout(z):
    return np.column_stack((z, z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]))


def _state_fit(z, observed, t):
    q, r = np.linalg.qr(z, mode='reduced')
    target = q.T @ observed

    def profile(h):
        basis = -np.expm1(-t[:, None] * np.exp(h))
        projected = _fit(basis, target.T).T @ basis.T
        return projected

    def residual(h):
        return (profile(h) - target).ravel()

    candidates = [least_squares(residual, np.log(start), bounds=(-5, 3), max_nfev=180)
                  for start in ((.3, 1.5), (.1, 3.), (.65, 2.5))]
    best = min(candidates, key=lambda f: f.cost)
    return _fit(r, profile(best.x))


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    x, x_test = (np.asarray(d['x'], dtype=float) for d in (train, test))
    c = _calibration(study['panel_a'])
    panel = study['panel_b']
    transform = _fit(_affine(np.asarray(panel['u'], dtype=float)), np.asarray(panel['v'], dtype=float))
    z = _affine(_inverse(x, c)) @ transform
    z_test = _affine(_inverse(x_test, c)) @ transform
    y_hat = _readout(z_test) @ _fit(_readout(z), np.asarray(train['y'], dtype=float))
    state_pred = z_test @ _state_fit(z, np.asarray(train['traj'], dtype=float), np.asarray(study['eval_times'], dtype=float))
    terminal = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
