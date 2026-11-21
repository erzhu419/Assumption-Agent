import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _integrate(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv - zv)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + .5*dt*k1x, z + .5*dt*k1z)
        k3x, k3z = deriv(x + .5*dt*k2x, z + .5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_integrate(r, p) for r in rows]) - y
    lower = np.array([0.001, 0.0, 0.0, 0.02])
    upper = np.array([5.0, 5.0, 10.0, 5.0])
    starts = [[0.5, 0.5, 2.0, 0.5], [1.0, 1.0, 3.0, 0.2], [0.1, 0.1, 1.0, 1.0]]
    best = None
    for s in starts:
        result = least_squares(residual, np.asarray(s), bounds=(lower, upper), max_nfev=250)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_integrate(row, _PARAMS))