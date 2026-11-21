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
        duration = end - float(pref[0]['t'])
        conv = 0.0
        for left, right in zip(pref[:-1], pref[1:]):
            t0, t1 = float(left['t']), float(right['t'])
            conv += 0.5 * (math.exp(-(end-t0)/tau)*float(left['x']) + math.exp(-(end-t1)/tau)*float(right['x'])) * (t1-t0) / tau
        z = math.exp(-duration/tau) * 0.52 * float(pref[0]['x']) + 0.52 * conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H)/0.004)))
    dt = H/n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x+dt*k1x/2, z+dt*k1z/2)
        k3x, k3z = deriv(x+dt*k2x/2, z+dt*k2z/2)
        k4x, k4z = deriv(x+dt*k3x, z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(r, p)-float(r['y']) for r in train_rows])
    fits = [least_squares(residual, start, bounds=([0, 0, 0, 0.03], [10, 10, 10, 10]), max_nfev=1000, xtol=1e-10, ftol=1e-10, gtol=1e-10) for start in ([1, 1, 1, 0.3], [1, 1, 1, 1.0], [1, 1, 1, 2.0])]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x
    return _params

def predict(row):
    return _forecast(row, _params)
