import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    p = row['raw_prefix']
    z = 0.52 * float(p[0]['x'])
    for left, right in zip(p[:-1], p[1:]):
        dt = float(right['t']) - float(left['t'])
        e = math.exp(-dt / tau)
        z = e*z + 0.52*dt/(2.0*tau)*(float(left['x'])*e + float(right['x']))
    return z

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = f(x, z)
        k2, l2 = f(x+dt*k1/2, z+dt*l1/2)
        k3, l3 = f(x+dt*k2/2, z+dt*l2/2)
        k4, l4 = f(x+dt*k3, z+dt*l3)
        x += dt*(k1+2*k2+2*k3+k4)/6
        z += dt*(l1+2*l2+2*l3+l4)/6
    return float(x)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p)-float(r['y']) for r in rows])
    best = None
    for tau in (0.1, 0.3, 0.8, 1.5):
        result = least_squares(residual, [1.5, 0.8, 1.2, tau], bounds=([0.0, 0.0, 0.0, 0.03], [10.0, 15.0, 10.0, 5.0]), max_nfev=1500)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return _forecast(row, _params)
