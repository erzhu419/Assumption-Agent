"""Predict the scalar response and trajectory from three fitted panel operators."""

from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _calibration_basis(z):
    pairwise = np.column_stack([z[:, i] * z[:, j]
                                for i, j in combinations_with_replacement(range(z.shape[1]), 2)])
    return _design(z, pairwise, z**3)


def _calibrate(panel, measured):
    design = _calibration_basis(panel['u'])
    values = panel['v']
    rng = np.random.default_rng(672)
    keep_count = int(0.82 * len(values))
    best_loss = np.inf
    for _ in range(250):
        sample = rng.choice(len(values), design.shape[1] + 2, replace=False)
        coeff = _fit(design[sample], values[sample])
        for _ in range(9):
            residual2 = np.sum((design @ coeff - values)**2, axis=1)
            keep = np.argpartition(residual2, keep_count - 1)[:keep_count]
            coeff = _fit(design[keep], values[keep])
        residual2 = np.sum((design @ coeff - values)**2, axis=1)
        loss = np.partition(residual2, keep_count - 1)[:keep_count].sum()
        if loss < best_loss:
            best_loss, best_coeff = loss, coeff
    # The panel separates measurement noise from gross corruption by several orders of magnitude.
    inliers = np.linalg.norm(design @ best_coeff - values, axis=1) < 0.003
    coeff = _fit(design[inliers], values[inliers])
    for j in range(values.shape[1]):
        support = np.abs(coeff[:, j]) >= 0.001
        coeff[~support, j] = 0.0
        coeff[support, j] = _fit(design[inliers][:, support], values[inliers, j])

    initial = np.linalg.solve(coeff[1:5].T, (measured - coeff[0]).T).T
    return np.array([least_squares(
        lambda z: _calibration_basis(z[None, :])[0] @ coeff - row,
        guess, gtol=1e-12).x for row, guess in zip(measured, initial)])


def _threshold_power(z):
    return np.sign(z) * np.maximum(np.abs(z) - 0.35, 0.0)**1.5


def _odd_saturation(z):
    return -np.sign(z) * np.expm1(-1.8 * np.abs(z))


def _operator(panel, key, coordinates, transform):
    coeff = _fit(_design(transform(panel['u'])), panel[key])
    return _design(transform(coordinates)) @ coeff


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    measured = np.vstack((train['x'], study['test']['x']))
    a = _calibrate(study['panel_a'], measured)
    b = _operator(study['panel_b'], 'v', a, _threshold_power)
    c = _operator(study['panel_c'], 'v1', b, _odd_saturation)

    interaction = np.sum(a * c, axis=1)
    scalar_design = _design(a, c, interaction)
    y = scalar_design[n:] @ _fit(scalar_design[:n], train['y'])

    # Both trajectory modes share the scalar spatial basis, with one extra asymmetric interaction.
    state_design = np.column_stack((scalar_design, a[:, 1] * c[:, 0]))
    t = np.asarray(study['eval_times'])
    observed = train['traj'].T

    def modes(rates):
        return -np.expm1(-t[:, None] * rates)

    def temporal_residual(rates):
        basis = modes(rates)
        return (basis @ _fit(basis, observed) - observed).ravel()

    rates = least_squares(temporal_residual, [0.42, 1.8],
                          bounds=([0.05, 0.95], [0.95, 5.0])).x
    basis = modes(rates)
    amplitudes = _fit(basis, observed).T
    state = (state_design[n:] @ _fit(state_design[:n], amplitudes)) @ basis.T
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
