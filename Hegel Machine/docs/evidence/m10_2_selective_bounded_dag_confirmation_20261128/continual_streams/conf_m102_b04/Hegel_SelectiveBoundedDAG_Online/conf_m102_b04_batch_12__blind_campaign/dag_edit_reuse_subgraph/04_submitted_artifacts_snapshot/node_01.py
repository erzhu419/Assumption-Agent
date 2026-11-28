"""Calibrate the periodic panel operator and fit separate outcome readouts."""
import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_affine(x), x[:, i] * x[:, j]))


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=None)[0]


def _operator(panel, q):
    u = panel['u']
    v = panel['v']

    def design(x, omega):
        return _affine(np.cos(omega * x))

    def loss(omega):
        a = design(u, omega)
        return np.sum((a @ _fit(a, v) - v) ** 2)

    omega = minimize_scalar(loss, bounds=(2.3, 2.9), method='bounded',
                            options={'xatol': 1e-10}).x
    return design(q, omega) @ _fit(design(u, omega), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    latent = inputs['node_00']
    train = study['train']
    qt = np.cbrt(latent['cubic_train'])
    qe = np.cbrt(latent['cubic_test'])
    n = len(qt)
    b = _operator(study['panel_b'], np.concatenate((qt, qe), axis=0))

    d = _quadratic(b)
    y_hat = d[n:] @ _fit(d[:n], train['y'])

    a = _affine(b)
    traj = train['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state_pred = (a[n:] @ _fit(a[:n], traj @ modes.T)) @ modes
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
