import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    base_factor = u / (1.2 + u)
    sink_factor = s / (1.0 + s)
    branches = [
        ('sqrt_plus', lambda c: c * np.sqrt(u)),
        ('sqrt_minus', lambda c: -c * np.sqrt(u)),
        ('linear_leak', lambda c: -c * u),
    ]
    candidates = []
    for name, branch in branches:
        def residual(p):
            vmax, rho, c = p
            pred = vmax * (1.0 - rho * sink_factor) * base_factor + branch(c)
            return pred - y
        result = least_squares(residual, x0=[3.0, 0.7, 0.1], bounds=([0.0, 0.0, 0.0], [20.0, 2.0, 20.0]))
        candidates.append((float(np.mean(result.fun ** 2)), name, result.x))
    _, name, params = min(candidates, key=lambda item: item[0])
    return {'branch': name, 'params': params}

def predict(row):
    p = _MODEL['params']
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = p[0] * (1.0 - p[1] * s / (1.0 + s)) * (u / (1.2 + u))
    if _MODEL['branch'] == 'sqrt_plus':
        return float(base + p[2] * math.sqrt(u))
    if _MODEL['branch'] == 'sqrt_minus':
        return float(base - p[2] * math.sqrt(u))
    return float(base - p[2] * u)
