import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    p = row['raw_prefix']
    end = float(p[-1]['t'])
    total = 0.0
    for left, right in zip(p[:-1], p[1:]):
        dt = float(right['t']) - float(left['t'])
        total += 0.5 * dt * (math.exp(-(end-float(left['t']))/tau)*float(left['x']) + math.exp(-(end-float(right['t']))/tau)*float(right['x'])) / tau
    return 0.52 * (math.exp(-(end-float(p[0]['t']))/tau)*float(p[0]['x']) + total)

def _predict(row, params):
    a, b, c, tau = params
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    h = float(row['H'])
    n = max(1, int(math.ceil(abs(h)/0.004)))
    dt = h/n
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
        return np.array([_predict(r, p)-float(r['y']) for r in train_rows])
    result = least_squares(residual, [1.5, 1.0, 1.0, 0.5], bounds=([0.0, -10.0, -10.0, 0.05], [20.0, 20.0, 20.0, 10.0]), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return _predict(row, _params)
