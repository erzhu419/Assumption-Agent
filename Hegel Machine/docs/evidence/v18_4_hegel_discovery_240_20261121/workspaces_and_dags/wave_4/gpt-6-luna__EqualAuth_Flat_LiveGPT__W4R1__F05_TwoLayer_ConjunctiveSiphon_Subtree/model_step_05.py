import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _rhs(x, z, row, p):
    a_slow, b_couple, c_drive, tau_fast, rho = p
    u_n = max(0.0, float(row['u_now'])) / 4.0
    s = max(0.0, float(row['s_grad']))
    j_siph = c_drive * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return -a_slow * x + b_couple * z + j_siph, (0.52 * x - z) / tau_fast

def _simulate(row, p):
    x = float(row['x_now'])
    z = float(row['mem_conv'] or 0.0)
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    for _ in range(n):
        k1x, k1z = _rhs(x, z, row, p)
        k2x, k2z = _rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z, row, p)
        k3x, k3z = _rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z, row, p)
        k4x, k4z = _rhs(x + dt*k3x, z + dt*k3z, row, p)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array(v, dtype=float) for v in (
        [0.2, 0.05, 8.0, 0.4, 0.7], [0.5, 0.15, 10.0, 1.0, 0.5],
        [0.1, 0.01, 6.0, 0.15, 0.3], [0.8, 0.3, 12.0, 2.0, 0.8])]
    lo = [1e-5, 0.0, 0.0, 0.005, 0.0]
    hi = [5.0, 5.0, 50.0, 20.0, 1.0]
    results = [least_squares(residual, p, bounds=(lo, hi), max_nfev=3000) for p in starts]
    _PARAMS = min(results, key=lambda q: float(np.dot(q.fun, q.fun))).x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))