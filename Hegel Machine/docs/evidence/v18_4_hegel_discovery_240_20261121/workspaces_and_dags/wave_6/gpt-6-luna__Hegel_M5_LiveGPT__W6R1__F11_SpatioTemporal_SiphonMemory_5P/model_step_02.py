import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    H = float(row['H'])
    n = max(100, int(math.ceil(H / 0.002)))
    dt = H / n
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    def rhs(xv, zv):
        return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
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
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in rows]) - y
    starts = [np.array([0.5, 0.5, 5.0, 0.7, 0.05]), np.array([1.0, 1.0, 7.0, 0.5, 0.1])]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([0.001, -5.0, 0.01, 0.001, 0.002], [10.0, 5.0, 30.0, 0.999, 2.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit must be called before predict')
    return float(_predict_params(row, _PARAMS))