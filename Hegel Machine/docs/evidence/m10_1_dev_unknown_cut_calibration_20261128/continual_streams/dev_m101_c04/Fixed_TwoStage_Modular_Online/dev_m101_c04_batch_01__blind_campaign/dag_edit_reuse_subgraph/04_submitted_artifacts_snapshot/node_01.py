import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _scalar_basis(z, affine):
    return np.column_stack((affine, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))


def _scalar_predict(basis, params):
    score = basis @ params[:-2]
    return score + params[-2] * np.tanh(params[-1] * score)


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    z_tr, z_te = data['z_tr'], data['z_te']
    tr, te = _affine(z_tr), _affine(z_te)
    state_pred = te @ np.linalg.lstsq(tr, data['traj'], rcond=None)[0]

    basis = _scalar_basis(z_tr, tr)
    y = data['y']
    linear = np.linalg.lstsq(basis, y, rcond=None)[0]
    fit = least_squares(
        lambda p: _scalar_predict(basis, p) - y,
        np.r_[0.75 * linear, 0.45, 1.5],
        bounds=(np.r_[np.full(len(linear), -np.inf), 0., 0.1],
                np.r_[np.full(len(linear), np.inf), 2., 8.]),
    )
    y_hat = _scalar_predict(_scalar_basis(z_te, te), fit.x)
    endpoint = state_pred[:, -1]
    return {
        'y_hat': y_hat,
        'state_pred': state_pred,
        'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), endpoint.mean(), endpoint.std()]),
    }
