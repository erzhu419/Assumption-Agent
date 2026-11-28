import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _label_features(z):
    cross = (z[:, :2] * z[:, 2:]).sum(axis=1)
    return np.column_stack((z, cross))


def _label(z, z_test, y):
    x, x_test = _label_features(z), _label_features(z_test)
    initial = np.linalg.lstsq(_affine(x), y, rcond=None)[0][1:]

    def design(features, p):
        index = features @ p[:-1] + p[-1]
        return np.column_stack((np.ones(len(index)), index, np.tanh(index)))

    def residual(p):
        basis = design(x, p)
        return basis @ np.linalg.lstsq(basis, y, rcond=None)[0] - y

    fits = (least_squares(residual, np.r_[scale * initial, 0.0], max_nfev=100)
            for scale in (0.6, 1.0, 1.6))
    p = min(fits, key=lambda fit: fit.cost).x
    coef = np.linalg.lstsq(design(x, p), y, rcond=None)[0]
    return design(x_test, p) @ coef


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    z, z_test = (inputs['node_00'][k] for k in ('z_tr', 'z_te'))
    y_hat = _label(z, z_test, train['y'])
    coef = np.linalg.lstsq(_affine(z), train['traj'], rcond=None)[0]
    state_pred = _affine(z_test) @ coef
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
