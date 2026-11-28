"""Recover latent coordinates and predict the scalar and trajectory readouts."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = _affine(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(3):
        r = np.linalg.norm(d @ _fit(d[keep], v[keep]) - v, axis=1)
        keep = r < 3 * np.median(r)
    neighbor = np.roll(u, -1, axis=1)
    # The cyclic coupling and cubic term are shared across channels.
    d = np.column_stack((np.tile(np.eye(u.shape[1]), (len(u), 1)),
                         (u[:, :, None] * np.eye(u.shape[1])).reshape(u.size, -1),
                         neighbor.ravel(), (u * neighbor).ravel(), (u**3).ravel()))
    return _fit(d[keep.repeat(u.shape[1])], v[keep].ravel())


def _invert(x, coef):
    n = x.shape[1]
    offset, gain = coef[:n], coef[n:2*n]
    coupling, product, cubic = coef[2*n:]
    i = np.arange(n)
    j = (i + 1) % n
    u = (x - offset) / gain
    jac = np.zeros((len(x), n, n))
    for _ in range(25):
        neighbor = u[:, j]
        residual = offset + gain*u + coupling*neighbor + product*u*neighbor + cubic*u**3 - x
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, i, i] = gain + product*neighbor + 3*cubic*u**2
        jac[:, i, j] = coupling + product*u
        u -= np.linalg.solve(jac, residual[..., None])[..., 0]
    return u


def _label_design(z):
    a, b, c, d = z.T
    return np.column_stack((_affine(z), a*c + b*d))


def _state_readout(design, observed, test_design, times):
    coeff = _fit(design, observed)

    def projected(log_rates):
        basis = -np.expm1(-np.exp(log_rates)[:, None] * times)
        return _fit(basis.T, coeff.T).T @ basis

    def residual(log_rates):
        return (design @ (projected(log_rates) - coeff)).ravel()

    result = least_squares(residual, np.log([0.4, 1.8]), bounds=(-4, 3))
    return test_design @ projected(result.x)


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    panel = study['panel_b']
    mixer = _fit(_affine(panel['u']), panel['v'])
    z = _affine(_invert(train['x'], calibration)) @ mixer
    z_test = _affine(_invert(study['test']['x'], calibration)) @ mixer
    y_hat = _label_design(z_test) @ _fit(_label_design(z), train['y'])
    state_pred = _state_readout(_affine(z), train['traj'], _affine(z_test), study['eval_times'])
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
