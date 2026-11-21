import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row['raw_prefix']
    end = float(pref[-1]['t'])
    total = 0.0
    for left, right in zip(pref[:-1], pref[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        total += 0.5 * (math.exp(-(end-t0)/tau)*float(left['x']) + math.exp(-(end-t1)/tau)*float(right['x'])) * (t1-t0) / tau
    return 0.52 * (math.exp(-(end-float(pref[0]['t']))/tau)*float(pref[0]['x']) + total)

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H)/0.004)))
    h = H/n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = f(x, z)
        k2, l2 = f(x+h*k1/2, z+h*l1/2)
        k3, l3 = f(x+h*k2/2, z+h*l2/2)
        k4, l4 = f(x+h*k3, z+h*l3)
        x += h*(k1+2*k2+2*k3+k4)/6
        z += h*(l1+2*l2+2*l3+l4)/6
    return float(x)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_forecast(row, p)-float(row['y']) for row in train_rows])
    best = None
    for tau in (0.1, 0.4, 1.0, 2.0):
        result = least_squares(residual, [1.0, 0.5, 1.0, tau], bounds=([0.0, 0.0, 0.0, 0.025], [10.0, 10.0, 10.0, 10.0]), max_nfev=1000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
