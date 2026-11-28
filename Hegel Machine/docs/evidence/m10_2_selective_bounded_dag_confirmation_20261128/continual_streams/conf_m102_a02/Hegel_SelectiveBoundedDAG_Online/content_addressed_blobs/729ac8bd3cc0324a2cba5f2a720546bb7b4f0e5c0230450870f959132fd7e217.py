"""Invert paired reference responses and fit the two study readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import root


PAIRS = ((0, 2), (1, 3))
QUADRATIC = tuple(combinations_with_replacement(range(4), 2))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _odd_design(q):
    return np.column_stack((np.ones(len(q)), q, q**3, q**5, q**7))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    displacement = np.max(np.abs(v - u), axis=1)
    med = np.median(displacement)
    mad = np.median(np.abs(displacement - med))
    good = displacement <= med + max(1., 8 * mad)
    result = []
    for i, j in PAIRS:
        a, b = u[good][:, [i, j]], v[good][:, [i, j]]
        result.append((_ls(_odd_design(a), b), _ls(np.c_[np.ones(len(b)), b], a)))
    return result


def _invert(x, models):
    q = np.empty_like(x, dtype=float)
    for (i, j), (w, inverse_affine) in zip(PAIRS, models):
        values = x[:, [i, j]]
        starts = np.c_[np.ones(len(values)), values] @ inverse_affine

        def jac(z):
            return (w[1:3] + (3*z**2)[:, None]*w[3:5]
                    + (5*z**4)[:, None]*w[5:7] + (7*z**6)[:, None]*w[7:9]).T

        for k, (value, start) in enumerate(zip(values, starts)):
            def residual(z):
                return (_odd_design(z[None, :]) @ w)[0] - value

            solution = root(residual, start, jac=jac, tol=1e-10)
            q[k, [i, j]] = (solution.x if np.max(np.abs(residual(solution.x))) < 1e-8
                             else start)
    return q


def _panel_b(q, p, w):
    scale, power, exponent = p
    log_abs = np.log(np.maximum(np.abs(q), np.finfo(float).tiny))
    h = q * np.exp(exponent * np.logaddexp(0., power * (np.log(scale) + log_abs)))
    return np.c_[np.ones(len(q)), q, h] @ w


def run_node(inputs: dict) -> dict:
    study, operator = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    q = _invert(x, _calibration(study['panel_a']))
    b = _panel_b(q, operator['p'], operator['w'])
    cross = b * np.roll(q, -1, axis=1)
    squares = np.column_stack([q[:, i] * q[:, j] for i, j in QUADRATIC])
    gate_strength = float(study['telemetry_vec'][0])
    gate = 1. / (1. + gate_strength * q[:, 0]**2)
    scalar = np.c_[np.ones(len(q)), b, b * gate[:, None], squares, cross]
    trajectory = np.c_[np.ones(len(q)), q]
    y = scalar[n:] @ _ls(scalar[:n], train['y'])
    state = trajectory[n:] @ _ls(trajectory[:n], train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
