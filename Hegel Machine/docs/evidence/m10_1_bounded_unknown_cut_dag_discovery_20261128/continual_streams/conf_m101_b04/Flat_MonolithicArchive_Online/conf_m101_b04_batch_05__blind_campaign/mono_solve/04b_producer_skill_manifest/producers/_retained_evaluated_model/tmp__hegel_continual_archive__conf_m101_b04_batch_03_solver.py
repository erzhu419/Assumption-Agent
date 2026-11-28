"""Predict scalar and trajectory readouts from the panel B operator."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, i]*x[:, j]
                            for i, j in combinations_with_replacement(range(x.shape[1]), 2))))


def _panel_b(panel, x):
    u, v = panel['u'], panel['v']

    def design(z, frequencies):
        return np.column_stack((_affine(z), np.tanh(z*frequencies)))

    def residual(frequencies):
        a = design(u, frequencies)
        return (a @ _fit(a, v) - v).ravel()

    frequencies = least_squares(residual, [1.4]*4, bounds=(.25, 5.5)).x
    return design(x, frequencies) @ _fit(design(u, frequencies), v)


def _relaxation(traj, t, initial):
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _fit(b, traj.T) - traj.T).ravel()

    rates = least_squares(residual, initial, bounds=([.7, .05], [4., .65])).x
    modes = np.linalg.qr(basis(rates), mode='reduced')[0]
    return traj @ modes, modes


def _forecast(d2, d2_test, p, pt, response, penalty, raw_terms):
    k = np.exp(-.035*d2)
    k.flat[::len(k)+1] += .015
    factor = cho_factor(k, lower=True)
    # Penalize the raw-coordinate additions on a common scale, leaving the
    # panel trend unpenalized. Solve in whitened coordinates.
    w = solve_triangular(factor[0], p, lower=True)
    y = solve_triangular(factor[0], response, lower=True)
    reg = np.zeros((raw_terms, p.shape[1]))
    reg[:, -raw_terms:] = np.sqrt(penalty)*np.eye(raw_terms)
    zeros = np.zeros((raw_terms,) + response.shape[1:])
    beta = _fit(np.vstack((w, reg)), np.concatenate((y, zeros), axis=0))
    return pt @ beta + np.exp(-.035*d2_test) @ cho_solve(factor, response - p @ beta)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = train['x'], study['test']['x']
    n = len(x)
    all_x = np.vstack((x, xt))
    aligned = _panel_b(study['panel_b'], all_x)
    raw = _quadratic(all_x)[:, 1:]
    trend = np.column_stack((_quadratic(aligned), raw / raw[:n].std(axis=0)))
    p, pt = trend[:n], trend[n:]
    d2 = cdist(x, x, 'sqeuclidean')
    d2_test = cdist(xt, x, 'sqeuclidean')
    modes, basis = _relaxation(train['traj'], study['eval_times'], [1.8, .41])
    raw_terms = raw.shape[1]
    y_hat = _forecast(d2, d2_test, p, pt, train['y'], 20., raw_terms)
    state_pred = _forecast(d2, d2_test, p, pt, modes, 60., raw_terms) @ basis.T
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
