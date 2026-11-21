import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row['raw_prefix']
    end = float(pref[-1]['t'])
    start = float(pref[0]['t'])
    conv = 0.0
    for left, right in zip(pref[:-1], pref[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        conv += 0.5 * (math.exp(-(end-t0)/tau)*float(left['x']) + math.exp(-(end-t1)/tau)*float(right['x'])) * (t1-t0) / tau
    return 0.52 * (math.exp(-(end-start)/tau)*float(pref[0]['x']) + conv)

def _predict(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H)/0.004)))
    dt = H/n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x+dt*k1x/2, z+dt*k1z/2)
        k3x, k3z = f(x+dt*k2x/2, z+dt*k2z/2)
        k4x, k4z = f(x+dt*k3x, z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict(r, p)-float(r['y']) for r in rows])
    best = None
    for tau in (0.15, 0.4, 1.0, 2.0):
        result = least_squares(residual, [1.5, 0.8, 1.0, tau], bounds=([0.0, -10.0, -10.0, 0.05], [10.0, 10.0, 10.0, 10.0]), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _predict(row, _params)
