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
    state = np.array([float(row['x_now']), z], dtype=float)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
    def deriv(v):
        x, z = v
        return np.array([-a*x+b*z+c*u, (0.52*x-z)/tau])
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state+0.5*dt*k1)
        k3 = deriv(state+0.5*dt*k2)
        k4 = deriv(state+dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6.0
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p)-float(r['y']) for r in rows])
    starts = [[1.0, 1.0, 1.0, 0.4], [1.5, 1.0, 1.0, 1.0], [0.8, 0.5, 1.0, 0.15]]
    results = [least_squares(residual, p, bounds=([0.0, 0.0, 0.0, 0.03], [10.0, 10.0, 10.0, 10.0]), max_nfev=1500) for p in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return _forecast(row, _params)
