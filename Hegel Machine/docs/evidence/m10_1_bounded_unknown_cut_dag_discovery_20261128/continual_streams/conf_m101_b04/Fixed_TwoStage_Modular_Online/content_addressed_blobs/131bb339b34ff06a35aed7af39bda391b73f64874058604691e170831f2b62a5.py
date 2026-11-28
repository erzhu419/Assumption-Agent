from itertools import combinations, combinations_with_replacement

import numpy as np


def _pair_basis(x, frequency):
    pair = np.column_stack([x[:, i] + sign * x[:, j]
                            for i, j in combinations(range(4), 2)
                            for sign in (-1, 1)])
    return np.column_stack((np.tanh(frequency * x),
                            np.tanh(frequency * pair), np.cos(frequency * pair)))


def _basis(x, w, frequency):
    quadratic = [x[:, i] * x[:, j]
                 for i, j in combinations_with_replacement(range(4), 2)]
    return np.column_stack((np.ones(len(x)), x, *quadratic,
                            _pair_basis(x, frequency), _pair_basis(w, frequency)))


def _predict(x, xt, w, wt, target, frequency, penalty):
    design = _basis(x, w, frequency)
    scale = np.maximum(design.std(axis=0), 0.1)
    design /= scale
    reg = np.full(design.shape[1], penalty)
    reg[:15] *= 0.05
    reg[0] = 0
    coef = np.linalg.solve(design.T @ design + np.diag(reg), design.T @ target)
    return (_basis(xt, wt, frequency) / scale) @ coef


def _kernel_predict(x, xt, target, *, gaussian=False):
    dist = np.sum((x[:, None] - x[None, :]) ** 2, axis=2)
    cross_dist = np.sum((xt[:, None] - x[None, :]) ** 2, axis=2)
    scale = 1.3 * np.median(dist[np.triu_indices(len(x), 1)])

    def kernel(d):
        q = d / scale
        return np.exp(-q / 2) if gaussian else (1 + q / 4) ** -2

    base = np.column_stack((np.ones(len(x)), x)) if gaussian else np.ones((len(x), 1))
    test_base = np.column_stack((np.ones(len(xt)), xt)) if gaussian else np.ones((len(xt), 1))
    system = kernel(dist) + 0.0003 * np.eye(len(x))
    solved = np.linalg.solve(system, np.column_stack((base, target)))
    p, r = np.split(solved, [base.shape[1]], axis=1)
    trend = np.linalg.solve(base.T @ p, base.T @ r)
    pred = test_base @ trend + kernel(cross_dist) @ (r - p @ trend)
    return pred[:, 0] if target.ndim == 1 else pred


def run_node(inputs: dict) -> dict:
    d = inputs['node_00']
    scalar_basis = _predict(d['xa'], d['xat'], d['xab'], d['xabt'], d['y'], 1.8, 0.1)
    scalar_kernel = _kernel_predict(d['xa'], d['xat'], d['y'])
    y = 0.2 * scalar_basis + 0.8 * scalar_kernel
    state = _kernel_predict(d['z'], d['zt'], d['traj'], gaussian=True)
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
