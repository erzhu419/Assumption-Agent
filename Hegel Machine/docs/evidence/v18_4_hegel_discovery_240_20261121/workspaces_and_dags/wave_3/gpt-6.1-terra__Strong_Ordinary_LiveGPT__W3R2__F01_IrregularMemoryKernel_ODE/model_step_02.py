import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    z = row.get('mem_conv')
    if z is None:
        prefix = row['raw_prefix']
        z = 0.52 * prefix[0]['x']
        for left, right in zip(prefix[:-1], prefix[1:]):
            dt = right['t'] - left['t']
            decay = math.exp(-dt / tau)
            z = z * decay + 0.52 * dt / (2.0 * tau) * (left['x'] * decay + right['x'])
    state = np.array([float(row['x_now']), float(z)])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    h = H / n
    def deriv(v):
        x, z = v
        return np.array([-a*x + b*z + c*u, (0.52*x-z)/tau])
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5*h*k1)
        k3 = deriv(state + 0.5*h*k2)
        k4 = deriv(state + h*k3)
        state += h*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return float(state[0])

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - row['y'] for row in train_rows])
    result = least_squares(residual, [1.0, 0.5, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.02], [20.0, 20.0, 20.0, 20.0]), max_nfev=1000)
    _params = result.x

def predict(row):
    return _forecast(row, _params)
