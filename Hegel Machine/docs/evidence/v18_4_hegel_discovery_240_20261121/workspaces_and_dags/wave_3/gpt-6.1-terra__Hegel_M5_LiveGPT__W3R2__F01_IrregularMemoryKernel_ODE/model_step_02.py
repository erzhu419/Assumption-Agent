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
        for left, right in zip(pref[:-1], pref[1:]):
            dt = right['t'] - left['t']
            conv += 0.5 * dt * (math.exp(-(end-left['t'])/tau)*left['x'] + math.exp(-(end-right['t'])/tau)*right['x']) / tau
        z += 0.52 * conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.004)))
    dt = H / n
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
    def residual(p):
        return np.array([_forecast(r, p)-float(r['y']) for r in train_rows])
    bounds = ([0.0001, -10.0, -10.0, 0.03], [10.0, 10.0, 10.0, 10.0])
    results = [least_squares(residual, start, bounds=bounds, max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11) for start in ([1.0, 0.5, 1.0, 0.3], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.5, 2.0])]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return _forecast(row, _params)
