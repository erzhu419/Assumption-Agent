import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _interactions(x, z):
    products = [z[:, i] * z[:, j] for i in range(z.shape[1])
                for j in range(i, z.shape[1])]
    return np.column_stack((*products, x[:, 3] * np.cos(2.0 * x[:, 2])))


def _readout(affine, interactions, direction):
    score = affine @ direction
    return np.column_stack((affine[:, 0], score, np.tanh(score), interactions))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    calibrated = inputs['node_00']
    operator = inputs['node_01']
    z_train, z_test = operator['z_tr'], operator['z_te']
    a_train, a_test = _affine(z_train), _affine(z_test)
    q_train = _interactions(calibrated['x_tr_cal'], z_train)
    q_test = _interactions(calibrated['x_te_cal'], z_test)
    y = operator['y']

    def residual(direction):
        design = _readout(a_train, q_train, direction)
        return design @ _fit(design, y) - y

    initial = _fit(a_train, y)
    result = min((least_squares(residual, scale * initial, max_nfev=100)
                  for scale in (0.65, 1.3)), key=lambda fit: fit.cost)
    readout_weights = _fit(_readout(a_train, q_train, result.x), y)
    y_hat = _readout(a_test, q_test, result.x) @ readout_weights
    state_pred = a_test @ _fit(a_train, operator['traj'])
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
