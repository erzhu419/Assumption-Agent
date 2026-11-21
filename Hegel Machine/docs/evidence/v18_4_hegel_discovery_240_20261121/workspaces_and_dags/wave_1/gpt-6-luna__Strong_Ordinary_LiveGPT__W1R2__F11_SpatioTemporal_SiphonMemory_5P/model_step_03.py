import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _integrate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def rhs(xv, zv):
        return (-a_slow * xv + b_couple * zv + c_drive * u,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_integrate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 0.2, 0.5, 0.2]),
                           bounds=([1e-6, -5.0, -5.0, 1e-4], [5.0, 5.0, 5.0, 5.0]),
                           max_nfev=5000)
    _PARAMS = result.x
    return result.x.copy()

def predict(row):
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) before predict(row).')
    return float(_integrate(row, _PARAMS))