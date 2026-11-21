import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, c, tau = p
    z = row.get('mem_conv')
    if z is None:
        pref = row['raw_prefix']
        end = pref[-1]['t']
        z = 0.52 * pref[0]['x'] * math.exp(-(end - pref[0]['t']) / tau)
        conv = 0.0
        for q, r in zip(pref[:-1], pref[1:]):
            dt = r['t'] - q['t']
            conv += 0.5 * dt * (math.exp(-(end - q['t']) / tau) * q['x'] + math.exp(-(end - r['t']) / tau) * r['x']) / tau
        z += 0.52 * conv
    x = float(row['x_now'])
    z = float(z)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    h = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x - z)/tau
    for _ in range(n):
        k1, l1 = f(x, z)
        k2, l2 = f(x + h*k1/2, z + h*l1/2)
        k3, l3 = f(x + h*k2/2, z + h*l2/2)
        k4, l4 = f(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - row['y'] for row in train_rows])
    result = least_squares(residual, [1.0, 1.0, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.02], [20.0, 20.0, 20.0, 10.0]), max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return _forecast(row, _params)
