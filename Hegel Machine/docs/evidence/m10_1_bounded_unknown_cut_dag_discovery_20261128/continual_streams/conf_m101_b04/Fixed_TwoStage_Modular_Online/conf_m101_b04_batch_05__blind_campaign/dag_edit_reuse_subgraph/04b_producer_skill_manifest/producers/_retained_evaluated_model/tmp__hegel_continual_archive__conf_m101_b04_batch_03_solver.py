"""Calibrate inputs, recover the panel-B warp, and predict sparse readouts."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    design, target = _linear(panel['u']), panel['v']
    initial = _fit(design, target)
    robust = least_squares(
        lambda b: (design @ b.reshape(initial.shape) - target).ravel(),
        initial.ravel(), loss='cauchy', f_scale=.02,
    ).x.reshape(initial.shape)
    residual = np.linalg.norm(design @ robust - target, axis=1)
    keep = residual <= 10 * np.median(residual)
    return _fit(design[keep], target[keep])


def _warp(x, parameters):
    strength, scale = parameters
    return _linear(x - strength * erf(scale * x))


def _panel_operator(panel):
    u, v = panel['u'], panel['v']

    def residual(parameters):
        design = _warp(u, parameters)
        return (design @ _fit(design, v) - v).ravel()

    parameters = least_squares(residual, [.68, 1.45],
                               bounds=([.02, .15], [4., 4.])).x
    return parameters, _fit(_warp(u, parameters), v)


def _readouts(b):
    b0, b1, b2, b3 = b.T
    scalar = np.column_stack((_linear(b), b0 * b2, b1 * b3))
    trajectory = np.column_stack((scalar, b0 * b3))
    return scalar, trajectory


def _time_basis(times, trajectory):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _fit(b, trajectory.T) - trajectory.T).ravel()

    rates = least_squares(residual, [1.6, .3],
                          bounds=([.58, .01], [5., 1.2])).x
    return np.linalg.qr(basis(rates), mode='reduced')[0]


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    parameters, mixing = _panel_operator(study['panel_b'])

    def design(raw):
        calibrated = np.linalg.solve(calibration[1:].T,
                                     (raw - calibration[0]).T).T
        return _readouts(_warp(calibrated, parameters) @ mixing)

    scalar, trajectory = design(train['x'])
    scalar_test, trajectory_test = design(study['test']['x'])
    time_basis = _time_basis(study['eval_times'], train['traj'])
    y_hat = scalar_test @ _fit(scalar, train['y'])
    state_pred = trajectory_test @ _fit(trajectory, train['traj'] @ time_basis) @ time_basis.T
    final = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics)
