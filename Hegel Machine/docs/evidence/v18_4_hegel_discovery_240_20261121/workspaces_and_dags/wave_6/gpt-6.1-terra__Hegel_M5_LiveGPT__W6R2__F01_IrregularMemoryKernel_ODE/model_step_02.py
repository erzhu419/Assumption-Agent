import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict(row, p):
    a, b, c, tau = p
    z = row.get('mem_conv')
    if z is None:
        pref = row['raw_prefix']
        end = pref[-1]['t']
        span = end - pref[0]['t']
        conv = 0.0
        for left, right in zip(pref[:-1], pref[1:]):
            dt = right['t'] - left['t']
            conv += 0.5 * dt * (math.exp(-(end-left['t'])/tau)*left['x'] + math.exp(-(end-right['t'])/tau)*right['x']) / tau
        z = math.exp(-span/tau)*0.52*pref[0]['x'] + 0.52*conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H)/0.004)))
    h = H/n
    def deriv(x, z):
        return -a*x+b*z+c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x+h*k1/2, z+h*l1/2)
        k3, l3 = deriv(x+h*k2/2, z+h*l2/2)
        k4, l4 = deriv(x+h*k3, z+h*l3)
        x += h*(k1+2*k2+2*k3+k4)/6
        z += h*(l1+2*l2+2*l3+l4)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict(r, p)-r['y'] for r in rows])
    best = None
    for tau0 in (0.1, 0.4, 1.0, 2.0):
        result = least_squares(residual, [1.0, 0.5, 1.0, tau0], bounds=([0.0, 0.0, 0.0, 0.02], [20.0, 20.0, 20.0, 20.0]), max_nfev=1500)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return _predict(row, _params)
