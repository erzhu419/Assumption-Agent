"""Invert the reference calibration and predict the response and relaxation modes."""
from itertools import combinations, combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    nxt = np.roll(u, -1, axis=1)
    basis = [np.column_stack((np.ones(len(u)), x, z, x*z, x**3))
             for x, z in zip(u.T, nxt.T)]
    ids = np.arange(len(u))
    for _ in range(8):
        coef = np.column_stack([np.linalg.lstsq(f[ids], v[ids, j], rcond=None)[0]
                                for j, f in enumerate(basis)])
        residual = np.column_stack([f @ coef[:, j] for j, f in enumerate(basis)]) - v
        ids = np.argsort(np.linalg.norm(residual, axis=1))[:len(u)-3]
    return coef


def _inverse(x, c):
    def residual(z, row):
        w = np.roll(z, -1)
        return c[0] + c[1]*z + c[2]*w + c[3]*z*w + c[4]*z**3 - row
    return np.array([least_squares(residual, (row-c[0])/c[1], args=(row,),
                                   gtol=1e-11).x for row in x])


def _modes(t, traj):
    def basis(r):
        return -np.expm1(-t[:, None]*r)
    def project(h):
        return np.linalg.lstsq(h, traj.T, rcond=None)[0]
    def residual(r):
        h = basis(r)
        return (h @ project(h) - traj.T).ravel()
    r = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x
    h = basis(np.sort(r))
    return h, project(h).T


def _dictionary(z):
    x = z.T
    columns = [np.prod(x[list(ids)], axis=0) for k in range(1, 5)
               for ids in combinations_with_replacement(range(4), k)]
    for i, j in combinations(range(4), 2):
        w = x[i]*x[j]
        for k in (.7, 1.5, 2.5, 3.5):
            columns.extend((np.sin(k*w), np.tanh(k*w), np.cos(k*w)))
    for i, a in enumerate(x):
        for k in (.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.):
            for f in (np.sin(k*a), np.cos(k*a), np.tanh(k*a), 1/(1+k*a*a),
                      a/(1+k*a*a), np.sqrt(1+k*a*a)):
                columns.append(f)
                columns.extend(b*f for j, b in enumerate(x) if i != j)
    return np.column_stack(columns)


def _ridge(features, y, reg):
    n = len(y)
    a = features[:n]
    f = (features - a.mean(axis=0)) / np.maximum(a.std(axis=0), 1e-12)
    a = f[:n]
    return y.mean(axis=0) + (f[n:] @ a.T) @ np.linalg.solve(a@a.T+reg*np.eye(n), y-y.mean(axis=0))


def solve_batch(study):
    train, test = study['train'], study['test']
    z = _inverse(np.vstack((train['x'], test['x'])), _calibrate(study['panel_a']))
    h, modes = _modes(study['eval_times'], train['traj'])
    p = _ridge(_dictionary(z), np.column_stack((train['y'], modes)), .15)
    y, state = p[:, 0], p[:, 1:] @ h.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
