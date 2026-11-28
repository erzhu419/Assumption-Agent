import numpy as np


def _panel_basis(x):
    return np.column_stack((np.ones(len(x)), x * np.roll(x, -1, axis=1)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    calibrated = inputs['node_00']['calibrated_x']
    transformed = inputs['node_01']['x']
    panel = study['panel_c']
    panel_weights = np.linalg.lstsq(_panel_basis(panel['u']), panel['v1'], rcond=None)[0]
    panel_readout = _panel_basis(transformed) @ panel_weights

    # The four shifted cross-stage products share one readout weight.
    coupling = np.sum(np.roll(calibrated, -1, axis=1) * panel_readout, axis=1)
    shared = np.column_stack((np.ones(len(transformed)), panel_readout, coupling))
    scalar_basis = np.column_stack((shared, np.cos(transformed[:, 0]) - 1))
    state_basis = np.column_stack((shared, calibrated**2))
    train = study['train']
    n_train = len(train['x'])
    scalar_weights = np.linalg.lstsq(scalar_basis[:n_train], train['y'], rcond=None)[0]
    state_weights = np.linalg.lstsq(state_basis[:n_train], train['traj'], rcond=None)[0]
    y_hat = scalar_basis[n_train:] @ scalar_weights
    state_pred = state_basis[n_train:] @ state_weights
    terminal = state_pred[:, -1]
    return {
        'y_hat': y_hat,
        'state_pred': state_pred,
        'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()]),
    }
