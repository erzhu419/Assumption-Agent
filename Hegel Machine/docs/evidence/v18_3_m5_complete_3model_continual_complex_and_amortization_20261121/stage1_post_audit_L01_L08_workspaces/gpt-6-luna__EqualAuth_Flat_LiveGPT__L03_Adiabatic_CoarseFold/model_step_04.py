import math
import numpy as np
from scipy.optimize import least_squares

def _integrate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    dt0 = 0.004
    n = max(1, int(math.ceil(H / dt0)))
    dt = H / n if H > 0 else 0.0
    def deriv(xv, zv):
        return (-a_slow * xv + b_couple * zv + c_drive * u,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_integrate(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.3, 0.1, 2.0, 0.5]),
                           bounds=([0.0, -10.0, -10.0, 1e-4],
                                   [10.0, 10.0, 10.0, 100.0]),
                           max_nfev=3000)
    return {'params': result.x}

def predict(row):
    return float(_integrate(row, _MODEL_PARAMS))