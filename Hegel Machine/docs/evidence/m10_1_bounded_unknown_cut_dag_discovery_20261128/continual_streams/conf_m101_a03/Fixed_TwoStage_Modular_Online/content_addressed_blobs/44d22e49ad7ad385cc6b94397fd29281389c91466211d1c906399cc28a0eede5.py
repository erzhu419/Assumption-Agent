import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _scalar_basis(z, affine):
    return np.column_stack((affine, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))


def _scalar_design(basis, direction):
    return np.column_stack((basis, np.tanh(basis @ direction)))


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    z, z_test = data['z'], data['z_test']
    y = np.asarray(data['y']).reshape(-1)
    train, test = _affine(z), _affine(z_test)
    basis = _scalar_basis(z, train)

    def fit_scalar(direction):
        design = _scalar_design(basis, direction)
        return design, np.linalg.lstsq(design, y, rcond=None)[0]

    def residual(direction):
        design, weights = fit_scalar(direction)
        return design @ weights - y

    start = [0., 1., 1., 1., -1., 0., 0.]
    direction = least_squares(residual, start, max_nfev=150).x
    _, weights = fit_scalar(direction)
    y_hat = _scalar_design(_scalar_basis(z_test, test), direction) @ weights
    state_pred = test @ np.linalg.lstsq(train, data['traj'], rcond=None)[0]
    last = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
