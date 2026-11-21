import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau_scale = 2.0

def _predict_params(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / _tau_scale
    sg = float(row['s_grad'])
    H = float(row['H'])
    j = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def f(xx, zz):
        return -a_slow * xx + b_couple * zz + j, (0.52 * xx - zz) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - ys
    lo = np.array([0.001, -2.0, 0.01, 0.001, 0.01])
    hi = np.array([5.0, 2.0, 20.0, 0.999, 10.0])
    starts = [np.array([0.5, 0.1, 6.0, 0.7, 0.5]), np.array([1.0, 0.5, 8.0, 0.5, 1.0]), np.array([0.2, -0.5, 4.0, 0.8, 0.2])]
    sols = [least_squares(residual, np.clip(s, lo+1e-8, hi-1e-8), bounds=(lo, hi), max_nfev=3000) for s in starts]
    _params = min(sols, key=lambda q: float(np.dot(q.fun, q.fun))).x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _params))