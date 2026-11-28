"""Panel-informed scalar and state prediction for the public campaign."""
from itertools import combinations_with_replacement

import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _basis(x):
    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, j], axis=1) for d in (1, 2, 3) for j in combinations_with_replacement(range(x.shape[1]), d)])


def _panel_features(study, x):
    a, c = study['panel_a'], study['panel_c']
    u, v = a['u'], a['v']
    f = _affine(u)
    fit = np.linalg.lstsq(f, v, rcond=None)[0]
    for _ in range(10):
        err = np.linalg.norm(f @ fit - v, axis=1)
        keep = np.argsort(err)[:int(.82 * len(u))]
        fit = np.linalg.lstsq(f[keep], v[keep], rcond=None)[0]
    maps = np.linalg.lstsq(_affine(c['u']), c['v1'], rcond=None)[0]
    raw = _affine(x) @ maps
    aligned = _affine((x - fit[0]) / fit[1:].diagonal()) @ maps
    squash = lambda q: q / np.sqrt(1 + .09 * q*q)
    return _basis(squash(raw)), _basis(squash(aligned))


def _ridge(x, y, q, penalty):
    return q @ np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    raw, aligned = _panel_features(study, np.vstack((train['x'], study['test']['x'])))
    y = _ridge(raw[:n], train['y'], raw[n:], .012)
    z = train['traj']
    time_basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    amps = z @ time_basis.T
    state = (.5 * _ridge(raw[:n], amps, raw[n:], .014) + .5 * _ridge(aligned[:n], amps, aligned[n:], .014)) @ time_basis
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
