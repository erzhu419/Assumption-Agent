import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    a, b, tau, vmax, rho = p
    H = max(0.0, float(row['H']))
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    z = float(row['mem_conv'] or 0.0)
    un = u  # characteristic_tau_scale is 1.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    n = max(1, int(math.ceil(H / 0.12)))
    dt = H / n
    def deriv(xv, zv):
        return -a * xv + b * zv + J, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.15, 0.03, 1.0, 1.0, 0.70]),
        bounds=([0.001, 0.0, 0.05, 0.001, 0.0], [2.0, 1.0, 20.0, 10.0, 1.0]),
        max_nfev=180, x_scale='jac')
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(row, _PARAMS))