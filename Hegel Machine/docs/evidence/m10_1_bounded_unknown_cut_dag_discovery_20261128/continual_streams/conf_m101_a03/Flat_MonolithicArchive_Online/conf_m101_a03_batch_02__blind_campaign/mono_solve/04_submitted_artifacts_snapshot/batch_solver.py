from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import root


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    affine = np.c_[np.ones(len(u)), u]
    keep = np.arange(len(u))
    for _ in range(3):
        residual = v - affine @ _fit(affine[keep], v[keep])
        keep = np.argsort(np.linalg.norm(residual, axis=1))[:45]

    r = np.roll(u, -1, axis=1)
    terms = np.stack((np.ones_like(u), u, r, u*r, r**3), axis=1)
    for _ in range(3):
        coef = np.column_stack([_fit(terms[keep, :, i], v[keep, i]) for i in range(4)])
        residual = v - np.einsum('nki,ki->ni', terms, coef)
        keep = np.argsort(np.linalg.norm(residual, axis=1))[:45]

    def forward(z):
        r = np.roll(z, -1, axis=-1)
        return coef[0] + coef[1]*z + coef[2]*r + coef[3]*z*r + coef[4]*r**3

    linear = np.diag(coef[1])
    linear[(np.arange(4)+1)%4, np.arange(4)] = coef[2]
    initial = np.linalg.solve(linear.T, (x-coef[0]).T).T
    return np.array([root(lambda z: forward(z)-target, guess).x for target, guess in zip(x, initial)])


def _basis(z):
    return np.column_stack([np.ones(len(z)), z] + [z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)] + [np.tanh(z)])


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    z = _calibrate(study['panel_a'], np.r_[tr['x'], study['test']['x']])
    design = _basis(z)
    pred = design[n:] @ _fit(design[:n], np.c_[tr['y'], tr['traj']])
    y, state = pred[:, 0], pred[:, 1:]
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
