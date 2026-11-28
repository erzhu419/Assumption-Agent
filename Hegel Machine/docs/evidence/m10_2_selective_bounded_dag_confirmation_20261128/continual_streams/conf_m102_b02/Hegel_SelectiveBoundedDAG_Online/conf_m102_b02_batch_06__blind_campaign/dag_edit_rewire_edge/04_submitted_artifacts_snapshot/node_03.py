"""Fit separate scalar and trajectory readouts from the frozen DAG outputs."""
import numpy as np
from scipy.optimize import least_squares


def _features(x, c, design):
    # The signed saturation acts on main effects and paired interactions.
    main = design[:, :1 + x.shape[1] + c.shape[1]]
    pairs = x * c
    reverse_pairs = x * np.roll(c, 1, axis=1)
    return np.column_stack((main, pairs)), reverse_pairs


def _scalar(a, b, y, at, bt):
    scale_a = np.maximum(a.std(axis=0), 0.04)
    scale_a[0] = 1.0
    scale_b = b.std(axis=0)
    scale_b[scale_b < 0.04] = 1.0
    a, b = a / scale_a, b / scale_b
    at, bt = at / scale_a, bt / scale_b
    p, q = a.shape[1], b.shape[1]
    penalty = np.sqrt(0.15) * np.r_[np.full(p, 0.5), 0., np.ones(q)]
    penalty[0] = 0.0
    init = np.r_[np.linalg.lstsq(a, y, rcond=None)[0] / 2.5, 2.5, np.zeros(q)]

    def residual(w):
        z = a @ w[:p]
        return np.r_[w[p] * (z / (1 + np.abs(z))) + b @ w[p+1:] - y, penalty * w]

    def jacobian(w):
        z = a @ w[:p]
        den = 1 + np.abs(z)
        return np.vstack((np.column_stack((w[p] * a / den[:, None]**2, z / den, b)), np.diag(penalty)))

    lower = np.full(p + 1 + q, -np.inf)
    upper = -lower
    lower[p], upper[p] = 0.15, 15.0
    w = least_squares(residual, init, jac=jacobian, bounds=(lower, upper), max_nfev=100).x
    zt = at @ w[:p]
    return w[p] * zt / (1 + np.abs(zt)) + bt @ w[p+1:]


def run_node(inputs: dict) -> dict:
    coords, basis, supervision = inputs['node_00'], inputs['node_01'], inputs['node_02']
    train, test = basis['d_train'], basis['d_test']
    target, modes = supervision['target'], supervision['modes']
    a, b = _features(coords['x_train'], coords['c_train'], train)
    at, bt = _features(coords['x_test'], coords['c_test'], test)
    y_hat = _scalar(a, b, target[:, 0], at, bt)
    coef = np.linalg.lstsq(train, target[:, 1:] @ modes.T, rcond=None)[0]
    state = (test @ coef) @ modes
    last = state[:, -1]
    return dict(y_hat=y_hat, state_pred=state,
                diagnostics_vec=np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]))
