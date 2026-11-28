import numpy as np
from scipy.optimize import least_squares


def _features(u, b):
    cross = np.sum(u * np.roll(b, 1, axis=1), axis=1)
    contrast = (b[:, 0] - b[:, 1] - b[:, 2] + b[:, 3]) / 2
    curvature = np.sum(b * b, axis=1) - 2 * contrast**2
    return np.column_stack((np.ones(len(b)), b, cross)), curvature


def _scalar(x, curvature, p):
    return (p[0] * np.tanh(x[:, 1:5] @ p[1:5] + p[5])
            + p[6] * x[:, 5] + p[7] + p[8] * curvature)


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    coords = inputs['node_00']
    panel = inputs['node_01']
    u, ut = coords['u_tr'], coords['u_te']
    x, q = _features(u, panel['b_tr'])
    xt, qt = _features(ut, panel['b_te'])
    y = np.asarray(train['y'])
    w = np.linalg.lstsq(x, y, rcond=None)[0]
    initial = np.r_[2., w[1:5] / 2, 0., w[5], w[0], 0.]
    fit = least_squares(lambda p: _scalar(x, q, p) - y, initial, max_nfev=250)
    y_hat = _scalar(xt, qt, fit.x)
    weights = np.linalg.lstsq(np.column_stack((x, u)), train['traj'], rcond=None)[0]
    state = np.column_stack((xt, ut)) @ weights
    terminal = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
