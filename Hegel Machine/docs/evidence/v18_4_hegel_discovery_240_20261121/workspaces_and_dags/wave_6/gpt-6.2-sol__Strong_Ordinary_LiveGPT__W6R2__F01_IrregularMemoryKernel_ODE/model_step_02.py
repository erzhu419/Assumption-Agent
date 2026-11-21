import math
import numpy as np
from scipy.optimize import least_squares


def _memory(row, tau):
    measured = row.get('mem_conv')
    if measured is not None:
        return float(measured)
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    samples = sorted(prefix, key=lambda sample: sample['t'])
    z = 0.52 * samples[0]['x']
    for left, right in zip(samples, samples[1:]):
        q = (right['t'] - left['t']) / tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * left['x'] + right['x'])
    return z


def _forecast(row, params):
    a, b, c, tau = params
    h = float(row['H'])
    x = float(row['x_now'])
    z = _memory(row, tau)
    n = max(1, math.ceil(h / 0.004))
    # RK4 update polynomial for the augmented linear system (x, z, u).
    m = (h / n) * np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    identity = np.eye(3)
    step = identity + m @ (identity + m @ (identity / 2 + m @ (identity / 6 + m / 24)))
    state = np.array([x, z, float(row['u_now'])])
    return float((np.linalg.matrix_power(step, n) @ state)[0])


def fit(train_rows):
    global _params
    rows = list(train_rows)
    target = np.array([row['y'] for row in rows], dtype=float)

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - target

    result = least_squares(residual, [2.0, 1.0, 1.0, 0.7], bounds=([0.0, 0.0, 0.0, 0.01], [20.0, 20.0, 20.0, 20.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params


def predict(row):
    return _forecast(row, _params)
