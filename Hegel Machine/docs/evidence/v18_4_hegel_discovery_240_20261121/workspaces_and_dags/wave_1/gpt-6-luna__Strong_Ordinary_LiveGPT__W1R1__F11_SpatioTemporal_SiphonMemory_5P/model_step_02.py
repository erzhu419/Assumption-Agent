import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _integrate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + c_drive * u, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_integrate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.5, 0.2, 0.5, 0.15]), bounds=([0.0, -5.0, -5.0, 0.005], [5.0, 5.0, 5.0, 5.0]), max_nfev=20000)
    _params = result.x.copy()
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_integrate(row, _params))