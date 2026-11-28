"""Reconstruct the shared panel operators and fit the two observed readouts."""
import numpy as np
from scipy.optimize import root, least_squares


_EDGES = ((0, 1), (0, 3), (1, 2), (2, 3))
_SCALAR_EDGES = ((0, 2), (1, 3))


def _ls(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _aff(values):
    return np.column_stack((np.ones(len(values)), values))


def _pair(values, edges):
    return np.column_stack([values[:, i] * values[:, j] for i, j in edges])


def _cal_basis(z):
    return np.stack((np.ones_like(z), z, np.tanh(1.8*z),
                     np.sin(2.4*np.roll(z, 2, axis=-1))), axis=-1)


def _calibrate(panel, x):
    design = _cal_basis(np.asarray(panel['u']))
    coef = []
    for j in range(4):
        d, y = design[:, j], panel['v'][:, j]
        idx = np.arange(len(d))
        for _ in range(15):
            err = np.abs(d @ _ls(d[idx], y[idx]) - y)
            idx = np.argsort(err)[:int(.82*len(d))]
        good = err < max(.001, 8*np.median(err))
        coef.append(_ls(d[good], y[good]))
    coef = np.asarray(coef)

    def invert(row):
        def residual(z):
            return (_cal_basis(z)*coef).sum(axis=-1) - row
        est = root(residual, row, tol=1e-10)
        if np.max(np.abs(est.fun)) > 1e-8:
            est = least_squares(residual, est.x, gtol=1e-12)
        return est.x

    return np.asarray([invert(row) for row in x]).reshape(-1, 4)


def _coordinates(study, x):
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    b_design = lambda x: _aff(np.cos(2.6*x))
    w = b_design(z) @ _ls(b_design(b['u']), b['v'])
    p = _aff(_pair(w, _EDGES))
    q = p @ _ls(_aff(_pair(c['u'], _EDGES)), c['v1'])
    return w, p, q


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    _, p, q = _coordinates(study, np.concatenate((train['x'], study['test']['x'])))
    d = np.column_stack((_aff(q), _pair(q, _SCALAR_EDGES)))
    y = d[n:] @ _ls(d[:n], train['y'])

    # Project the shared spatial readout onto the two observed relaxation modes.
    t = np.asarray(study['eval_times'], dtype=float)
    time_basis = -np.expm1(-t[:, None] * np.array([.42, 1.8]))
    amplitudes = _ls(time_basis, np.asarray(train['traj']).T).T
    state = p[n:] @ _ls(p[:n], amplitudes) @ time_basis.T
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
