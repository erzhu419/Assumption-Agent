from itertools import combinations_with_replacement

import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), k=1)
    du = u[j] - u[i]
    dv = v[j] - v[i]
    gain = np.array([np.median(dv[du[:, k] != 0, k] / du[du[:, k] != 0, k])
                     for k in range(u.shape[1])])
    bias = np.median(v - u * gain, axis=0)
    error = np.linalg.norm(v - u * gain - bias, axis=1)
    center = np.median(error)
    inlier = error <= center + 8 * np.median(np.abs(error - center))
    bias, gain = np.array([_fit(_affine(u[inlier, k:k+1]), v[inlier, k])
                           for k in range(u.shape[1])]).T
    return bias, gain


def solve_batch(study: dict) -> dict:
    """Predict the scalar readout and trajectory in the calibrated operator coordinates."""
    train = study['train']
    bias, gain = _calibration(study['panel_a'])
    panel_b = study['panel_b']
    operator = _fit(_affine(panel_b['u']), panel_b['v'])
    z = _affine((train['x'] - bias) / gain) @ operator
    z_test = _affine((study['test']['x'] - bias) / gain) @ operator
    a, a_test = _affine(z), _affine(z_test)

    # Screen in the complete quadratic basis, then pool the two dominant terms.
    i, j = np.array(list(combinations_with_replacement(range(z.shape[1]), 2))).T
    products = z[:, i] * z[:, j]
    quadratic = _fit(np.column_stack((a, products)), train['y'])[a.shape[1]:]
    keep = np.argsort(np.abs(quadratic))[-2:]
    pooled = products[:, keep].sum(axis=1)
    pooled_test = (z_test[:, i[keep]] * z_test[:, j[keep]]).sum(axis=1)
    y_hat = np.column_stack((a_test, pooled_test)) @ _fit(np.column_stack((a, pooled)), train['y'])

    # Fit the input dependence in the two dominant temporal modes.
    traj = train['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state_pred = (a_test @ _fit(a, traj @ modes.T)) @ modes
    last = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
