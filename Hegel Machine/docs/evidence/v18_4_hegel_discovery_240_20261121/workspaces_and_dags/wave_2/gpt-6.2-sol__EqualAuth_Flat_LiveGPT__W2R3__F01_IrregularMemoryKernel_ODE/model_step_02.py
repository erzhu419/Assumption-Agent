import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    prefix = row.get('raw_prefix') or []
    return (float(row['x_now']), float(row['u_now']), float(row['H']),
            row.get('mem_conv'), tuple((float(p['t']), float(p['x'])) for p in prefix))

def _forecast(params, data):
    a, b, c, tau = params
    x, u, H, mem, prefix = data
    if mem is not None:
        z = float(mem)
    elif prefix:
        z = 0.52 * prefix[0][1]
        for (t0, x0), (t1, x1) in zip(prefix, prefix[1:]):
            q = (t1 - t0) / tau
            decay = math.exp(-q)
            z = decay * z + 0.26 * q * (decay * x0 + x1)
    else:
        z = 0.52 * x
    n = max(1, math.ceil(H / 0.004))
    # The third state is constant u; the RK4 step is a matrix polynomial.
    B = (H / n) * np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    B2 = B @ B
    step = np.eye(3) + B + B2 / 2.0 + (B2 @ B) / 6.0 + (B2 @ B2) / 24.0
    return float(np.linalg.matrix_power(step, n)[0] @ np.array([x, z, u]))

def fit(train_rows):
    global _params
    rows = [(_prepare(row), float(row['y'])) for row in train_rows]
    def residual(params):
        return np.array([_forecast(params, data) - y for data, y in rows])
    _params = least_squares(residual, [1.5, 1.0, 1.2, 0.35],
                            bounds=([0.0, 0.0, 0.0, 0.01], [np.inf] * 4),
                            x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    return _forecast(_params, _prepare(row))
