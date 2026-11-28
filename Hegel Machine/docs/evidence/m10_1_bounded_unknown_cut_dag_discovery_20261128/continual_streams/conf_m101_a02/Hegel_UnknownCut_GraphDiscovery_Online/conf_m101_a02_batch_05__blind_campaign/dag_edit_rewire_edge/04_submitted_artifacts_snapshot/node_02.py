import numpy as np


def _linear(z):
    return np.column_stack((np.ones(len(z)), z))


def _label_basis(x, z):
    # The label readout couples calibrated and mixed coordinates.
    difference = x[:, 2] - z[:, 3]
    return np.column_stack((
        _linear(z),
        z[:, 0] * z[:, 2],
        z[:, 1] * z[:, 3],
        x[:, 2],
        difference ** 3,
    ))


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    calibrated = inputs['node_00']
    mixed = inputs['node_01']
    z_tr, z_te = mixed['z_tr'], mixed['z_te']
    label_coef = np.linalg.lstsq(_label_basis(calibrated['cal_tr'], z_tr),
                                train['y'], rcond=None)[0]
    state_coef = np.linalg.lstsq(_linear(z_tr), train['traj'], rcond=None)[0]
    y_hat = _label_basis(calibrated['cal_te'], z_te) @ label_coef
    state_pred = _linear(z_te) @ state_coef
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
