"""Calibrated scalar response and shared two-mode trajectory regression."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _coordinates(study, x):
    a = study['panel_a']
    # The calibration is close to identity; the three corrupted rows are far outside it.
    good = np.max(np.abs(a['v'] - a['u']), axis=1) < 1.0
    u, v = a['u'][good], a['v'][good]
    centered = u - u.mean(axis=0)
    gain = np.sum(centered * (v - v.mean(axis=0)), axis=0) / np.sum(centered**2, axis=0)
    offset = v.mean(axis=0) - gain * u.mean(axis=0)
    b = study['panel_b']
    operator = _fit(np.column_stack((np.ones(len(b['u'])), b['u'])), b['v'])
    return (x - offset) / gain @ operator[1:] + operator[0]


def _scalar_design(z):
    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    return np.column_stack((z, interaction))


def _trajectory(x, target, x_test, t):
    # For fixed rates, project onto the temporal basis and profile out the loadings.
    def basis(rates):
        return -np.expm1(-t[:, None] * rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _fit(b, target.T) - target.T).ravel()

    rates = least_squares(residual, [0.4, 1.8], bounds=([0.01, 0.8], [0.8, 8.0])).x
    b = basis(rates)
    return (x_test @ _fit(x, _fit(b, target.T).T)) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = _coordinates(study, train['x'])
    x_test = _coordinates(study, study['test']['x'])
    y_hat = _scalar_design(x_test) @ _fit(_scalar_design(x), train['y'])
    state_pred = _trajectory(x, train['traj'], x_test, study['eval_times'])
    end = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
