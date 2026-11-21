import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    else:
        pref = row['raw_prefix']
        end = pref[-1]['t']
        z = 0.52 * pref[0]['x'] * math.exp(-(end - pref[0]['t']) / tau)
        conv = 0.0
        for left, right in zip(pref[:-1], pref[1:]):
            dt = right['t'] - left['t']
            conv += 0.5 * dt * (math.exp(-(end - left['t']) / tau) * left['x'] + math.exp(-(end - right['t']) / tau) * right['x']) / tau
        z += 0.52 * conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x+h*k1x/2, z+h*k1z/2)
        k3x, k3z = f(x+h*k2x/2, z+h*k2z/2)
        k4x, k4z = f(x+h*k3x, z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p)-float(r['y']) for r in train_rows])
    best = None
    for initial in ([1.0, 0.5, 0.8, 0.3], [2.0, 1.0, 1.0, 1.0], [0.5, 0.2, 0.5, 0.1]):
        result = least_squares(residual, initial, bounds=([0.0, 0.0, 0.0, 0.03], [20.0, 30.0, 20.0, 10.0]), max_nfev=1000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
