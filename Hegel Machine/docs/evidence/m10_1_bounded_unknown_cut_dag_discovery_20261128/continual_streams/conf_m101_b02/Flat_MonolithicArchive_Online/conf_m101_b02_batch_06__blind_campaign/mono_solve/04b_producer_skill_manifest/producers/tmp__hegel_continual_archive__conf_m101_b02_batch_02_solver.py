"""Predict responses using calibrated coordinates and panel-supported interactions."""
from itertools import combinations_with_replacement

import numpy as np


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _fit(d, v):
    return np.linalg.lstsq(d, v, rcond=None)[0]


def _design(u, terms):
    return np.column_stack([u[:, term].prod(axis=1) for term in terms])


def _calibration(study):
    a, b = study['panel_a'], study['panel_b']
    d, v = _affine(a['u']), a['v']
    keep = np.ones(len(d), dtype=bool)
    # Trim high-residual rows while retaining enough points for the initial fits.
    for _ in range(8):
        residual = np.linalg.norm(d @ _fit(d[keep], v[keep]) - v, axis=1)
        keep = residual < max(0.004, np.partition(residual, 34)[34])
    return _fit(d[keep], v[keep]), _fit(_affine(b['u']), b['v'])


def _latent(x, a, b):
    return _affine(np.linalg.solve(a[1:].T, (x - a[0]).T).T) @ b


def _terms(panel):
    dim = panel['u'].shape[1]
    linear = [()] + [(i,) for i in range(dim)]
    pairs = list(combinations_with_replacement(range(dim), 2))
    weights = _fit(_design(panel['u'], linear + pairs), panel['v1'])[len(linear):]
    support = [p for p, w in zip(pairs, weights) if np.linalg.norm(w) > 0.01]
    cubic = sorted({tuple(sorted((*p, i))) for p in support for i in range(dim)})
    return linear + support + cubic


def solve_batch(study: dict) -> dict:
    train = study['train']
    a, b = _calibration(study)
    terms = _terms(study['panel_c'])
    basis = np.linalg.svd(train['traj'], full_matrices=False)[2][:2]
    targets = np.column_stack((train['y'], train['traj'] @ basis.T))
    coef = _fit(_design(_latent(train['x'], a, b), terms), targets)
    pred = _design(_latent(study['test']['x'], a, b), terms) @ coef
    y_hat = pred[:, 0]
    state_pred = pred[:, 1:] @ basis
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
