import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, params):
    a, b, c, tau = params
    x = row['x_now']
    horizon = row['H']
    if horizon == 0:
        return float(x)
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix, prefix[1:]):
            q = (right['t'] - left['t']) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * left['x'] + right['x'])
    n = math.ceil(horizon / 0.004)
    h = horizon / n
    # Augment by a constant so one RK4 step is a matrix map.
    M = h * np.array([[-a, b, c * row['u_now']], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    M2 = M @ M
    step = np.eye(3) + M + M2 / 2.0 + (M2 @ M) / 6.0 + (M2 @ M2) / 24.0
    return float(np.linalg.matrix_power(step, n)[0] @ np.array([x, z, 1.0]))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    target = np.array([r['y'] for r in rows], dtype=float)
    def residual(params):
        return np.array([_forecast(r, params) for r in rows]) - target
    result = least_squares(residual, [2.0, 1.0, 1.0, 0.3], bounds=([0.0, 0.0, 0.0, 0.01], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return _forecast(row, _params)
