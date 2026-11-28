import numpy as np
from scipy.optimize import least_squares


def _c_basis(u, scale):
    return np.column_stack((np.ones(len(u)), u, np.tanh(u * scale)))


def _scalar_basis(b):
    i, j = np.triu_indices(b.shape[1])
    return np.column_stack((np.ones(len(b)), b, b[:, i] * b[:, j]))


def _state_basis(b, c):
    return np.column_stack((np.ones(len(b)), b, c, np.tanh(b)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    mapped = inputs['node_01']
    panel = study['panel_c']
    u, v = panel['u'], panel['v1']

    def residual(scale):
        design = _c_basis(u, scale)
        weights = np.linalg.lstsq(design, v, rcond=None)[0]
        return (design @ weights - v).ravel()

    scale = least_squares(residual, np.full(u.shape[1], 2.2),
                          bounds=(0.35, 6.0), max_nfev=180).x
    weights = np.linalg.lstsq(_c_basis(u, scale), v, rcond=None)[0]
    c_train = _c_basis(mapped['u_tr'], scale) @ weights
    c_test = _c_basis(mapped['u_te'], scale) @ weights
    b_train, b_test = mapped['b_tr'], mapped['b_te']
    y_weights = np.linalg.lstsq(_scalar_basis(b_train), study['train']['y'], rcond=None)[0]
    y_hat = _scalar_basis(b_test) @ y_weights

    trajectory = study['train']['traj']
    modes = np.linalg.svd(trajectory, full_matrices=False)[2][:2]
    state_weights = np.linalg.lstsq(_state_basis(b_train, c_train),
                                   trajectory @ modes.T, rcond=None)[0]
    states = (_state_basis(b_test, c_test) @ state_weights) @ modes
    final = states[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return {'y_hat': y_hat, 'state_pred': states, 'diagnostics_vec': diagnostics}
