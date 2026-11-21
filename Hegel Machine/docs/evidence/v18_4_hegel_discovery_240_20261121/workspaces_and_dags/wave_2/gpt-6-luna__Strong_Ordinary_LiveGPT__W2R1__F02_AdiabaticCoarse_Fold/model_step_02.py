import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = np.array([0.3, 0.1, 1.0, 0.2], dtype=float)

def _integrate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    tau_fast = max(float(tau_fast), 1e-8)
    h = float(row['H'])
    n = max(1, int(math.ceil(h / 0.004)))
    dt = h / n
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    def deriv(xv, zv):
        return (-a_slow * xv + b_couple * zv + c_drive * u,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_integrate(r, p) for r in train_rows]) - y
    starts = ([0.3,0.1,1.0,0.2], [0.5,0.2,1.5,0.5], [0.1,0.05,0.8,1.0])
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0,0.0,0.0,0.005],[5.0,5.0,10.0,5.0]), max_nfev=300)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    return float(_integrate(row, _PARAMS))