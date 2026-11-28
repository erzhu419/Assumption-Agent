"""Panel-calibrated scalar prediction and low-rank trajectory regression."""
from itertools import combinations_with_replacement
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _basis(x, degree):
    return np.column_stack([np.ones(len(x))] + [
        np.prod(x[:, j], axis=1)
        for d in range(1, degree + 1)
        for j in combinations_with_replacement(range(x.shape[1]), d)
    ])


def _calibrated(x, panel):
    u, v = panel['u'], panel['v']
    a = _affine(u)
    fit = np.linalg.lstsq(a, v, rcond=None)[0]
    for _ in range(10):
        keep = np.argsort(np.linalg.norm(a @ fit - v, axis=1))[:max(5, int(.82 * len(a)))]
        fit = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
    return (x - fit[0]) / fit[1:].diagonal()


def _ridge(f, y, n, penalty):
    a = f[:n]
    return f[n:] @ np.linalg.solve(a.T @ a + penalty * np.eye(a.shape[1]), a.T @ y)


def run_node(inputs: dict) -> dict:
    study, features = inputs['__input__'], inputs['node_00']
    train = study['train']
    y, z = train['y'], train['traj']
    n = len(y)
    x = np.vstack((train['x'], study['test']['x']))
    panel = study['panel_b']
    transform = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    q = _affine(_calibrated(x, study['panel_a'])) @ transform
    y_hat = _ridge(_basis(q / np.sqrt(1 + .3 * q*q), 3), y, n, .01)
    time_basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    amps = z @ time_basis.T
    state = (.5 * _ridge(features['raw'], amps, n, .014) +
             .5 * _ridge(features['aligned'], amps, n, .014)) @ time_basis
    end = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
