import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.05

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    s = float(row['s_grad'])
    u_n = float(row['u_now']) / TAU_SCALE
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = f(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6
    return x

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.5, 0.2, 6.0, 0.7, 0.05]),
        bounds=(np.array([0.001, -2.0, 0.01, 0.0, 0.002]),
                np.array([10.0, 2.0, 30.0, 0.999, 2.0])), max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_simulate(row, _PARAMS))