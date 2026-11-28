"""Panel-calibrated scalar prediction and two-mode relaxation."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    designs = [_aff(u[:, j:j+1]) for j in range(u.shape[1])]
    w = np.ones(len(u))
    for _ in range(20):
        coef = np.stack([_fit(a*w[:, None], v[:, j]*w)
                         for j, a in enumerate(designs)], axis=1)
        err = np.linalg.norm(coef[0]+u*coef[1]-v, axis=1)
        w = np.sqrt(np.minimum(1, 0.03/np.maximum(err, 1e-12)))
    keep = err < 0.01
    if np.count_nonzero(keep) >= len(u)//2:
        coef = np.stack([_fit(a[keep], v[keep, j])
                         for j, a in enumerate(designs)], axis=1)
    return (x-coef[0])/coef[1]


def _edges(z):
    return z[:, [0, 0, 1, 2]]*z[:, [1, 3, 2, 3]]


def _operators(panel, z):
    u = panel['u']
    first = _aff(_edges(z)) @ _fit(_aff(_edges(u)), panel['v1'])
    v = panel['v2']

    def design(x, freq):
        return _aff(np.column_stack((x, np.cos(freq*x))))

    def loss(freq):
        a = design(u, freq)
        return np.linalg.norm(a @ _fit(a, v)-v)

    grid = np.linspace(0.25, 6, 45)
    i = np.argmin([loss(f) for f in grid])
    freq = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(i+1, len(grid)-1)]),
                           method='bounded').x
    second = design(z, freq) @ _fit(design(u, freq), v)
    return first, second


def _quadratic(z):
    return np.column_stack((z, *(z[:, i]*z[:, j] for i,j in
                                combinations_with_replacement(range(z.shape[1]), 2))))


def _basis(t, rates):
    return -np.expm1(-t[:, None]*rates)


def _rates(t, trajectory):
    def residual(rates):
        a = _basis(t, rates)
        return (a @ _fit(a, trajectory.T)-trajectory.T).ravel()

    return least_squares(residual, [0.35, 1.6], bounds=([0.04, 0.65], [0.65, 6])).x


def _slow_correction(design, coordinates, slow, n):
    """Fit the nonlinear slow-mode residual without altering the shared readout."""
    train_coord = coordinates[:n]
    q = (coordinates - train_coord.mean(axis=0)) / train_coord.std(axis=0)
    smooth = np.sqrt(0.25 + q*q)
    train_design = design[:n]
    residual_features = smooth - design @ _fit(train_design, smooth[:n])
    residual_features /= np.maximum(residual_features[:n].std(axis=0), 1e-8)
    f = residual_features[:n]
    residual = slow - train_design @ _fit(train_design, slow)
    coef = np.linalg.solve(f.T @ f + np.eye(f.shape[1]), f.T @ residual)
    return residual_features[n:] @ coef


def solve_batch(study):
    train = study['train']
    n = len(train['x'])
    z = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    c, w = _operators(study['panel_c'], z)
    design = _aff(np.column_stack((_quadratic(z), c*w)))
    t = study['eval_times']
    basis = _basis(t, _rates(t, train['traj']))
    amplitudes = _fit(basis, train['traj'].T).T
    pred = design[n:] @ _fit(design[:n], np.column_stack((train['y'], amplitudes)))
    pred[:, 1] += _slow_correction(design, z+c, amplitudes[:, 0], n)
    y = pred[:, 0]
    state = pred[:, 1:] @ basis.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
