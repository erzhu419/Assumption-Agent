import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * float(prefix[0]['x'])
        for left, right in zip(prefix[:-1], prefix[1:]):
            dt = float(right['t']) - float(left['t'])
            decay = math.exp(-dt / tau)
            z = decay * z + 0.52 * dt / (2.0 * tau) * (decay * float(left['x']) + float(right['x']))
    state = np.array([float(row['x_now']), float(z)], dtype=float)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    h = H / n
    def deriv(v):
        x, memory = v
        return np.array([-a*x + b*memory + c*u, (0.52*x - memory)/tau])
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5*h*k1)
        k3 = deriv(state + 0.5*h*k2)
        k4 = deriv(state + h*k3)
        state += (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    best = None
    for start in ([1.0, 1.0, 1.0, 0.3], [1.0, 0.5, 1.0, 1.0], [2.0, 2.0, 1.0, 0.1], [0.5, 1.0, 0.5, 2.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.02], [20.0, 30.0, 20.0, 20.0]), max_nfev=1000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return _forecast(row, _params)
