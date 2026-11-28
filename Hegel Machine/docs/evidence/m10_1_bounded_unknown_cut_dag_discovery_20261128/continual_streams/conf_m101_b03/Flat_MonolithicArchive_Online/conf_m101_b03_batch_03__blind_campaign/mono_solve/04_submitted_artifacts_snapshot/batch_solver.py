"""Calibrated dual-operator readout with a shared relaxation basis."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _affine(values):
    return np.column_stack((np.ones(len(values)), values))


def _calibration_design(z):
    neighbor = np.roll(z, -1, axis=1)
    return np.stack((np.ones_like(z), z, neighbor, z * neighbor, z**3), axis=-1)


def _calibrated_coordinates(panel, x):
    u, v = panel['u'], panel['v']
    design = _calibration_design(u)
    # Start below the gross-outlier scale, then re-admit any clean large rows.
    clean = np.max(np.abs(v), axis=1) < 3
    for _ in range(2):
        coef = np.stack([_fit(design[clean, j], v[clean, j]) for j in range(u.shape[1])])
        clean = np.max(np.abs(np.sum(design * coef, axis=-1) - v), axis=1) < 0.01

    z = (x - coef[:, 0]) / coef[:, 1]
    j = np.arange(u.shape[1])
    for _ in range(16):
        residual = np.sum(_calibration_design(z) * coef, axis=-1) - x
        if np.max(np.abs(residual), initial=0.0) < 1e-12:
            break
        jacobian = np.zeros((len(z), len(j), len(j)))
        jacobian[:, j, j] = coef[:, 1] + coef[:, 3] * np.roll(z, -1, axis=1) + 3 * coef[:, 4] * z**2
        jacobian[:, j, (j + 1) % len(j)] = coef[:, 2] + coef[:, 3] * z
        z -= np.linalg.solve(jacobian, residual[..., None])[..., 0]
    return z


def _panel_operator(u, v, z, transform):
    def design(x, rate):
        return _affine(transform(x, rate))

    def loss(rate):
        a = design(u, rate)
        return np.linalg.norm(a @ _fit(a, v) - v)

    grid = np.linspace(0.25, 8, 180)
    i = np.argmin([loss(rate) for rate in grid])
    bounds = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    rate = minimize_scalar(loss, bounds=bounds, method='bounded').x
    return design(z, rate) @ _fit(design(u, rate), v)


def _readout_design(study, x):
    z = _calibrated_coordinates(study['panel_a'], x)
    c = study['panel_c']
    f = _panel_operator(c['u'], c['v1'], z, lambda a, k: np.column_stack(
        (a, np.sign(a) * -np.expm1(-k * np.abs(a)))))
    g = _panel_operator(c['u'], c['v2'], z, lambda a, k: np.cos(k * a))
    cyclic = np.sum(np.roll(z, -1, axis=1) * f, axis=1)
    diagonal = np.sum(f * g, axis=1)
    return _affine(np.column_stack((z, f, g, cyclic, diagonal, f[:, 0] * g[:, 1])))


def _time_basis(times, trajectories):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        a = basis(rates)
        return (a @ _fit(a, trajectories.T) - trajectories.T).ravel()

    rates = least_squares(residual, [0.38, 1.9, 5.3],
                          bounds=([0.05, 0.8, 3], [0.8, 3, 10])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    design = _readout_design(study, np.vstack((train['x'], study['test']['x'])))
    time_basis = _time_basis(study['eval_times'], train['traj'])
    amplitudes = _fit(time_basis, train['traj'].T).T
    targets = np.column_stack((train['y'], amplitudes))
    pred = design[n:] @ _fit(design[:n], targets)
    y = pred[:, 0]
    state = pred[:, 1:] @ time_basis.T
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
