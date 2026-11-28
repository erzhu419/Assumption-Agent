import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


def _predict(kernel, labels, ridge):
    n = len(labels)
    gram = kernel[:n, :n]
    cross = kernel[n:, :n]
    column_mean = gram.mean(axis=0)
    grand_mean = column_mean.mean()
    gram = gram - column_mean - column_mean[:, None] + grand_mean
    cross = cross - column_mean - cross.mean(axis=1, keepdims=True) + grand_mean
    gram.flat[::n + 1] += ridge
    mean = labels.mean(axis=0)
    return mean + cross @ solve(gram, labels - mean, assume_a='pos')


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    x = inputs['node_00']['x']
    dist = cdist(x, x, 'sqeuclidean')
    y_hat = _predict(np.exp(-0.25 * dist), train['y'], 0.00045)
    kernel = inputs['node_01']['base']
    state_pred = _predict(kernel[1] + 0.1 * kernel[0], train['traj'], 0.0005)
    final = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])}
