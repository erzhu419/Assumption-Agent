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
        z = math.exp(-duration/tau)*0.52*float(pref[0]['x']) + 0.52*conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    horizon = float(row['H'])
    n = max(1, int(math.ceil(abs(horizon)/0.004)))
    dt = horizon/n
    def deriv(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x+0.5*dt*k1x, z+0.5*dt*k1z)
        k3x, k3z = deriv(x+0.5*dt*k2x, z+0.5*dt*k2z)
        k4x, k4z = deriv(x+dt*k3x, z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(row, p)-float(row['y']) for row in rows])
    best = None
    for start in ([1.0, 1.0, 1.0, 0.3], [0.5, 0.5, 1.0, 0.8], [2.0, 2.0, 2.0, 0.15]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.03], [20.0, 20.0, 20.0, 10.0]), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
