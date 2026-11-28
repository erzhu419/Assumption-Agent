import numpy as np
from scipy.optimize import least_squares


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return x[:, i] * x[:, j]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _scalar(u, z, y):
    # Calibrated-by-transformed terms use both frozen upstream outputs.
    h = np.tanh(2.0 * z)
    D = _affine(np.column_stack((z, h, _quadratic(h), u * h)))
    n = len(y)
    scale = np.maximum(np.sqrt(np.mean(D[:n] ** 2, axis=0)), 1e-9)
    D = D / scale
    Q, s, V = np.linalg.svd(D[:n], full_matrices=False)
    keep = s > 1e-7
    Q = Q[:, keep]
    weights = V[keep].T / s[keep]
    transfer = D[n:] @ weights
    ridge = 0.16 * weights[1:]
    penalty_jac = np.column_stack((ridge, np.zeros((len(ridge), 2))))
    power = 0.55

    def link(x):
        return x / (1.0 + np.abs(x) ** (2 * power)) ** (0.5 / power)

    def residual(a):
        b, amplitude, offset = a[:-2], a[-2], a[-1]
        return np.r_[amplitude * link(Q @ b) + offset - y, ridge @ b]

    def jacobian(a):
        x = Q @ a[:-2]
        derivative = (1.0 + np.abs(x) ** (2 * power)) ** (-1.0 - 0.5 / power)
        data_jac = np.column_stack((a[-2] * derivative[:, None] * Q,
                                    link(x), np.ones(n)))
        return np.vstack((data_jac, penalty_jac))

    initial = Q.T @ y
    fits = [least_squares(residual, np.r_[initial / m, m, y.mean()],
                           jac=jacobian, max_nfev=60)
            for m in (0.5, 1.0, 2.0)]
    a = min(fits, key=lambda fit: fit.cost).x
    return a[-2] * link(transfer @ a[:-2]) + a[-1]


def _trajectory(z, panel, traj):
    q = panel['u']
    v = panel['v1']

    def residual(f):
        A = _affine(np.column_stack((q, np.tanh(f * q))))
        return (A @ np.linalg.lstsq(A, v, rcond=None)[0] - v).ravel()

    f = least_squares(residual, np.full(q.shape[1], 2.2), bounds=(.15, 8)).x
    h = np.tanh(z * f)
    D = np.column_stack((z, h, np.tanh(h), np.sin(h)))
    n = len(traj)
    center = D[:n].mean(axis=0)
    scale = np.maximum(D[:n].std(axis=0), 1e-10)
    D = (D - center) / scale
    U, s, V = np.linalg.svd(D[:n], full_matrices=False)
    avg = traj.mean(axis=0)
    coef = (V.T * (s / (s ** 2 + .0001))) @ (U.T @ (traj - avg))
    return avg + D[n:] @ coef


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    u = inputs['node_00']['calibrated_x']
    z = inputs['node_01']['x']
    train = study['train']
    y = _scalar(u, z, train['y'])
    state = _trajectory(z, study['panel_c'], train['traj'])
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
