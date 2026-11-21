import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    siphon = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    dt = 0.004
    n = max(1, int(math.ceil(H / dt)))
    h = H / n
    def rhs(xv, zv):
        return (-a_slow * xv + b_couple * zv + siphon,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = rhs(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = rhs(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _P
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.8, 0.3, 6.0, 0.7, 0.2]),
        bounds=([0.001, 0.0, 0.01, 0.0, 0.005], [5.0, 5.0, 30.0, 0.99, 5.0]),
        max_nfev=3000)
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _P))