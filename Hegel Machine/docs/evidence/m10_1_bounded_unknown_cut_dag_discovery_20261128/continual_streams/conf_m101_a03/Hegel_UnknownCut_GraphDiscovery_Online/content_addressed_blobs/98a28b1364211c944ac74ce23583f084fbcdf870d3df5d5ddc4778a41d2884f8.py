import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), k=1)
    du, dv = u[j] - u[i], v[j] - v[i]
    gain = np.array([np.median(dv[du[:, k] != 0, k] / du[du[:, k] != 0, k])
                     for k in range(u.shape[1])])
    bias = np.median(v - u * gain, axis=0)
    error = np.linalg.norm(v - u * gain - bias, axis=1)
    center = np.median(error)
    inlier = error <= center + 8 * np.median(np.abs(error - center))
    return np.array([_fit(_affine(u[inlier, k:k+1]), v[inlier, k])
                     for k in range(u.shape[1])]).T


def _readout_design(z):
    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    return np.column_stack((_affine(z), interaction))


def _readout(p, design):
    q = design @ p[:-3]
    offset, amplitude, frequency = p[-3:]
    return q + amplitude * np.tanh(frequency * q + offset)


def solve_batch(study: dict) -> dict:
    """Predict the scalar readout and time-resolved state after calibration."""
    train = study['train']
    bias, gain = _calibration(study['panel_a'])
    panel = study['panel_b']
    operator = _fit(_affine(panel['u']), panel['v'])

    def operator_coordinates(x):
        return _affine((x - bias) / gain) @ operator

    z = operator_coordinates(train['x'])
    z_test = operator_coordinates(study['test']['x'])
    design = _readout_design(z)
    y = train['y']
    linear_start = _fit(design, y)
    lower = np.r_[np.full(len(linear_start) + 2, -np.inf), 0.12]
    upper = np.r_[np.full(len(linear_start) + 2, np.inf), 4.0]
    candidates = [least_squares(lambda p: _readout(p, design) - y,
                                np.r_[linear_start, 0., amplitude, frequency],
                                bounds=(lower, upper), max_nfev=180)
                  for frequency in (0.65, 1.3, 1.7, 2.5)
                  for amplitude in (0.2, 0.45)]
    p = min(candidates, key=lambda fit: fit.cost).x
    y_hat = _readout(p, _readout_design(z_test))

    traj = train['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state_pred = (_affine(z_test) @ _fit(_affine(z), traj @ modes.T)) @ modes
    last = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
