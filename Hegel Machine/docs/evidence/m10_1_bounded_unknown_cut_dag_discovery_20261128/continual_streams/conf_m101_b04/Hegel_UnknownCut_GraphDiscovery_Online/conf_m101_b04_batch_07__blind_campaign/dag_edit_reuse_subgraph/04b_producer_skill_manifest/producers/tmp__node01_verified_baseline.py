import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _panel_coordinates(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])

    def design(a, scale):
        return np.column_stack((np.ones(len(a)), a, a**3 / (1.0 + scale * a**2)))

    def fit(scale):
        d = design(u, scale)
        return d, np.linalg.lstsq(d, v, rcond=None)[0]

    def residual(scale):
        d, coef = fit(scale)
        return (d @ coef - v).ravel()

    scale = least_squares(residual, [0.75], bounds=(0.015, 10.0)).x
    _, coef = fit(scale)
    return design(x, scale) @ coef


def _scalar(b, bt, y):
    scale = np.sqrt(b.shape[1]) * np.std(b, axis=0)
    z, zt = b / scale, bt / scale
    trend = np.column_stack((np.ones(len(b)), b))
    trend_test = np.column_stack((np.ones(len(bt)), bt))
    k = np.exp(-0.5 * cdist(z, z, 'sqeuclidean'))
    k.flat[::len(b) + 1] += 0.05
    p = solve(k, np.column_stack((y, trend)), assume_a='pos')
    beta = np.linalg.lstsq(trend.T @ p[:, 1:], trend.T @ p[:, 0], rcond=None)[0]
    k_test = np.exp(-0.5 * cdist(zt, z, 'sqeuclidean'))
    return trend_test @ beta + k_test @ (p[:, 0] - p[:, 1:] @ beta)


def run_node(inputs: dict) -> dict:
    study, calibrated = inputs['__input__'], inputs['node_00']
    x, xt = calibrated['x_tr'], calibrated['x_te']
    b, bt = np.split(_panel_coordinates(study['panel_b'], np.vstack((x, xt))), [len(x)])
    y_hat = _scalar(b, bt, np.asarray(study['train']['y']))
    d = np.column_stack((np.ones(len(x)), x, b))
    dt = np.column_stack((np.ones(len(xt)), xt, bt))
    state_pred = dt @ np.linalg.lstsq(d, study['train']['traj'], rcond=None)[0]
    end = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
