from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    x = inputs['node_00']['x']
    kernel = inputs['node_01']['base'][2]
    targets = np.column_stack((train['y'], train['traj']))
    n = len(targets)
    trend = np.column_stack((np.ones(len(x)), x, *(
        x[:, i] * x[:, j]
        for i, j in combinations_with_replacement(range(x.shape[1]), 2)
    )))
    factor = cho_factor(kernel[:n, :n] + 0.001 * np.eye(n))
    weighted_trend = cho_solve(factor, trend[:n])
    weighted_targets = cho_solve(factor, targets)
    coef = np.linalg.lstsq(trend[:n].T @ weighted_trend,
                           trend[:n].T @ weighted_targets, rcond=1e-9)[0]
    pred = trend[n:] @ coef + kernel[n:, :n] @ (weighted_targets - weighted_trend @ coef)
    y, traj = pred[:, 0], pred[:, 1:]
    return {'y_hat': y, 'state_pred': traj,
            'diagnostics_vec': np.array([y.mean(), y.std(), traj[:, -1].mean(), traj[:, -1].std()])}
