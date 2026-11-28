"""Predict the label and time course from the calibrated mixed coordinates."""

import numpy as np
from scipy.optimize import least_squares


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    """Fit each gain and offset after excluding reference-panel outliers."""
    coefficients = []
    for u, v in zip(panel['u'].T, panel['v'].T):
        design = _affine(u)
        keep = np.arange(len(u))
        for _ in range(3):
            c = _fit(design[keep], v[keep])
            residual = np.abs(v - design @ c)
            keep = np.argsort(residual)[:3 * len(u) // 4]
        good = residual <= 8 * np.median(residual)
        coefficients.append(_fit(design[good], v[good]))
    return np.asarray(coefficients).T


def _label_basis(z):
    paired_products = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    return np.column_stack((_affine(z), paired_products))


def _trajectory(design, observed, design_test, times):
    # Profile out the affine amplitudes, leaving only two shared rates to fit.
    affine_coeff = _fit(design, observed)

    def time_basis(log_rates):
        return -np.expm1(-np.exp(log_rates)[:, None] * times)

    def profile(log_rates):
        basis = time_basis(log_rates)
        amplitudes = _fit(basis.T, affine_coeff.T).T
        return amplitudes @ basis

    def residual(log_rates):
        return (design @ (affine_coeff - profile(log_rates))).ravel()

    rates = least_squares(residual, np.log([0.4, 1.8]), bounds=(-4, 3)).x
    return design_test @ profile(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    offset, gain = _calibration(study['panel_a'])
    panel_b = study['panel_b']
    mixer = _fit(_affine(panel_b['u']), panel_b['v'])

    def coordinates(x):
        return _affine((x - offset) / gain) @ mixer

    z_train = coordinates(train['x'])
    z_test = coordinates(study['test']['x'])
    y_hat = _label_basis(z_test) @ _fit(_label_basis(z_train), train['y'])
    state_pred = _trajectory(_affine(z_train), train['traj'], _affine(z_test),
                             study['eval_times'])
    final = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
