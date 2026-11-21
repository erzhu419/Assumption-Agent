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
        end = float(pref[-1]['t'])
        z = 0.52 * float(pref[0]['x']) * math.exp(-(end - float(pref[0]['t'])) / tau)
        conv = 0.0
        for left, right in zip(pref[:-1], pref[1:]):
            tl, tr = float(left['t']), float(right['t'])
            conv += 0.5 * (math.exp(-(end-tl)/tau)*float(left['x']) + math.exp(-(end-tr)/tau)*float(right['x'])) * (tr-tl) / tau
        z += 0.52 * conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    h = H / n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, [1.0, 0.5, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return float(_forecast(row, _params))