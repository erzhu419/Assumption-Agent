import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    starts = [np.array([2.0, 0.7, 0.5, 0.2, 0.05]), np.array([1.0, 0.5, 0.2, 0.5, 0.1]), np.array([3.0, 0.8, 1.0, 0.1, 0.02])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, 0.0, 1e-5], [100.0, 1.0, 20.0, 20.0, 10.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_predict_params(row, _FIT_PARAMS))

_FIT_PARAMS = None

def _set_fit_params(train_rows):
    global _FIT_PARAMS
    _FIT_PARAMS = fit(train_rows)
    return _FIT_PARAMS
