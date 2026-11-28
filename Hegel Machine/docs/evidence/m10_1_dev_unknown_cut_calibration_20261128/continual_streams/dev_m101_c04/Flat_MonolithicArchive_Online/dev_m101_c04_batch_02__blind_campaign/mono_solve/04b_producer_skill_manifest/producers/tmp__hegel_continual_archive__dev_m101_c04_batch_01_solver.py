"""Calibrated shared-score scalar response and rank-two trajectory regression."""

import numpy as np
from scipy.optimize import least_squares


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(study):
    a = study['panel_a']
    u, v = a['u'], a['v']
    i, j = np.triu_indices(len(u), 1)
    coefficients = []
    for x, y in zip(u.T, v.T):
        dx = x[i] - x[j]
        valid = dx != 0
        gain = np.median((y[i] - y[j])[valid] / dx[valid])
        offset = np.median(y - gain * x)
        error = np.abs(y - (offset + gain * x))
        inlier = error <= max(0.001, 5 * np.median(error))
        coefficients.append(_ls(_affine(x[inlier]), y[inlier]))
    offset, gain = np.asarray(coefficients).T
    b = study['panel_b']
    return offset, gain, _ls(_affine(b['u']), b['v'])


def _coordinates(x, calibration):
    offset, gain, mixing = calibration
    return _affine((x - offset) / gain) @ mixing


def _scalar_design(z):
    cross = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    return np.column_stack((z, cross))


def _scalar_basis(design, direction):
    score = design @ direction
    return np.column_stack((score, np.tanh(score)))


def _predict_scalar(z, y, z_test):
    design = _scalar_design(z)
    seed = _ls(design, y)

    def residual(direction):
        basis = _scalar_basis(design, direction)
        return basis @ _ls(basis, y) - y

    # Profile out the linear and saturation amplitudes for each shared score.
    fits = (least_squares(residual, scale * seed, bounds=(-8., 8.), max_nfev=100)
            for scale in (0.65, 1., 1.5))
    direction = min(fits, key=lambda fit: fit.cost).x
    weights = _ls(_scalar_basis(design, direction), y)
    return _scalar_basis(_scalar_design(z_test), direction) @ weights


def _predict_state(z, trajectories, z_test):
    # Reduce rank after projecting out trajectory variation not predictable by z.
    q, r = np.linalg.qr(z, mode='reduced')
    u, s, vt = np.linalg.svd(q.T @ trajectories, full_matrices=False)
    rank = 2
    projected = (u[:, :rank] * s[:rank]) @ vt[:rank]
    return z_test @ _ls(r, projected)


def solve_batch(study: dict) -> dict:
    calibration = _calibration(study)
    train = study['train']
    z = _coordinates(train['x'], calibration)
    z_test = _coordinates(study['test']['x'], calibration)
    y_hat = _predict_scalar(z, train['y'], z_test)
    state_pred = _predict_state(z, train['traj'], z_test)
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
