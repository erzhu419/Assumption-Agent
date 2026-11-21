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
        conv = 0.0
        for left, right in zip(pref[:-1], pref[1:]):
            dt = right['t'] - left['t']
            conv += 0.5 * dt * (math.exp(-(end-left['t'])/tau)*left['x'] + math.exp(-(end-right['t'])/tau)*right['x']) / tau
        z = math.exp(-(end-pref[0]['t'])/tau)*0.52*pref[0]['x'] + 0.52*conv
    x = float(row['x_now'])
    u = float(row['u_now'])
    h = float(row['H'])
    n = max(1, int(math.ceil(abs(h)/0.004)))
    dt = h/n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = f(x, z)
        k2, l2 = f(x+dt*k1/2, z+dt*l1/2)
        k3, l3 = f(x+dt*k2/2, z+dt*l2/2)
        k4, l4 = f(x+dt*k3, z+dt*l3)
        x += dt*(k1+2*k2+2*k3+k4)/6
        z += dt*(l1+2*l2+2*l3+l4)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(row, p)-row['y'] for row in rows])
    result = least_squares(residual, [1.0, 1.0, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.05], [10.0, 10.0, 10.0, 10.0]), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    _params = result.x
    return _params

def predict(row):
    return float(_forecast(row, _params))