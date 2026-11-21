import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    measured = row.get('mem_conv')
    if measured is not None:
        return float(measured)
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    samples = sorted(prefix, key=lambda sample: sample['t'])
    z = 0.52 * float(samples[0]['x'])
    for left, right in zip(samples, samples[1:]):
        q = (float(right['t']) - float(left['t'])) / tau
        decay = math.exp(-q)
        z = decay * z + 0.26 * q * (decay * float(left['x']) + float(right['x']))
    return z

def _forecast(row, params):
    a, b, c, tau = map(float, params)
    x = float(row['x_now'])
    h = float(row['H'])
    if h == 0.0:
        return x
    z = _memory(row, tau)
    n = max(1, math.ceil(h / 0.004))
    dt = h / n
    A, B, C = -dt * a, dt * b, dt * c * float(row['u_now'])
    D, E = dt * 0.52 / tau, -dt / tau

    def increment(x, z):
        return A*x + B*z + C, D*x + E*z

    for _ in range(n):
        k1, l1 = increment(x, z)
        k2, l2 = increment(x + k1/2, z + l1/2)
        k3, l3 = increment(x + k2/2, z + l2/2)
        k4, l4 = increment(x + k3, z + l3)
        x += (k1 + 2*k2 + 2*k3 + k4) / 6
        z += (l1 + 2*l2 + 2*l3 + l4) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([row['y'] for row in rows], dtype=float)

    def residual(params):
        return np.array([_forecast(row, params) for row in rows]) - targets

    result = least_squares(residual, [2.0, 1.0, 1.0, 0.7], bounds=([0.0, 0.0, 0.0, 0.01], [np.inf]*4), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row) -> float:
    return _forecast(row, _params)
