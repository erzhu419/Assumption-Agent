"""Calibrated polynomial readout and two-mode trajectory predictor."""
import numpy as np
from scipy.optimize import least_squares


def _calibrated(study, x):
    a, b = study['panel_a'], study['panel_b']
    u, v = a['u'], a['v']
    # Estimate diagonal calibration from pairwise slopes, then refit on clean rows.
    i, j = np.triu_indices(len(u), 1)
    slope = np.array([np.median((v[i, k] - v[j, k])[keep] / du[keep])
                      for k in range(u.shape[1])
                      for du in [u[i, k] - u[j, k]]
                      for keep in [np.abs(du) > 1e-8]])
    offset = np.median(v - u * slope, axis=0)
    err = np.linalg.norm(v - u * slope - offset, axis=1)
    clean = err <= 5 * np.median(err)
    uc = u[clean] - u[clean].mean(axis=0)
    vc = v[clean] - v[clean].mean(axis=0)
    slope = (uc * vc).sum(axis=0) / (uc * uc).sum(axis=0)
    offset = v[clean].mean(axis=0) - u[clean].mean(axis=0) * slope

    # The auxiliary operator is a scaled orthogonal map plus a translation.
    bu, bv = b['u'], b['v']
    linear = np.linalg.lstsq(bu - bu.mean(axis=0), bv - bv.mean(axis=0), rcond=None)[0]
    left, singular, right = np.linalg.svd(linear)
    linear = singular.mean() * (left @ right)
    return ((x - offset) / slope - bu.mean(axis=0)) @ linear + bv.mean(axis=0)


def _readout(z):
    return np.column_stack((z, z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]))


def _trajectory(z, traj, ztest, times):
    fitted = z @ np.linalg.lstsq(z, traj, rcond=None)[0]

    def basis(rates):
        return -np.expm1(-np.outer(rates, times))

    def residual(rates):
        t = basis(rates)
        return (fitted @ np.linalg.pinv(t) @ t - fitted).ravel()

    rates = least_squares(residual, (0.4, 1.8), bounds=((0.05, 0.9), (0.89, 5))).x
    t = basis(rates)
    amplitudes = np.linalg.lstsq(z, traj @ np.linalg.pinv(t), rcond=None)[0]
    return ztest @ amplitudes @ t


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = train['x'], study['test']['x']
    z, ztest = np.split(_calibrated(study, np.concatenate((x, xt))), [len(x)])
    w = np.linalg.lstsq(_readout(z), train['y'], rcond=None)[0]
    y_hat = _readout(ztest) @ w
    state_pred = _trajectory(z, train['traj'], ztest, study['eval_times'])
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
