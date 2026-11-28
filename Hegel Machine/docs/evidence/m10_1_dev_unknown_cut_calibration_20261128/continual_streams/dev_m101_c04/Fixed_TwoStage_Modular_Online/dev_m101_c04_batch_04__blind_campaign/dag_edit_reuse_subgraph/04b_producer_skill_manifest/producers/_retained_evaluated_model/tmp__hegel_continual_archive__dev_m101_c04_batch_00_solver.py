import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(study):
    a = study['panel_a']
    u, v = a['u'], a['v']
    i, j = np.triu_indices(len(u), 1)
    coeff = []
    for x, y in zip(u.T, v.T):
        dx = x[i] - x[j]
        good = dx != 0
        gain = np.median((y[i] - y[j])[good] / dx[good])
        offset = np.median(y - gain * x)
        err = np.abs(y - offset - gain * x)
        keep = err <= max(0.001, 5 * np.median(err))
        coeff.append(_ls(_affine(x[keep]), y[keep]))
    offset, gain = np.asarray(coeff).T
    b = study['panel_b']
    mixing = _ls(_affine(b['u']), b['v'])
    return offset, gain, mixing


def _coordinates(x, calibration):
    offset, gain, mixing = calibration
    return _affine((x - offset) / gain) @ mixing


def _cross(z):
    return z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]


def _relaxation(design, trajectories, times):
    # Project onto the affine input space before fitting the shared rates.
    q, r = np.linalg.qr(design, mode='reduced')
    projected = q.T @ trajectories

    def basis(log_rates):
        return -np.expm1(-times[:, None] * np.exp(log_rates))

    def residual(log_rates):
        b = basis(log_rates)
        return (b @ _ls(b, projected.T) - projected.T).ravel()

    result = least_squares(residual, np.log([0.4, 1.8]),
                           bounds=(np.log(0.03), np.log(15.0)))
    b = basis(result.x)
    weights = _ls(r, _ls(b, projected.T).T)
    return weights @ b.T


def solve_batch(study: dict) -> dict:
    calibration = _calibration(study)
    train = study['train']
    z = _coordinates(train['x'], calibration)
    z_test = _coordinates(study['test']['x'], calibration)
    design, test_design = _affine(z), _affine(z_test)
    # The scalar law has no baseline at the calibrated origin.
    w = _ls(np.column_stack((z, _cross(z))), train['y'])
    y_hat = np.column_stack((z_test, _cross(z_test))) @ w
    state_pred = test_design @ _relaxation(design, train['traj'], study['eval_times'])
    final_state = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         final_state.mean(), final_state.std()])}
