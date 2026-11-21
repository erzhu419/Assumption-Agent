import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + c_drive * u,
                         (0.52 * xx - zz) / tau_fast])
    s = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(s[0])

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows])
    starts = [np.array([0.5, 0.1, 1.0, 0.2]), np.array([1.0, 0.2, 2.0, 0.5]), np.array([0.2, 0.05, 1.0, 1.0])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([1e-5, 0.0, 1e-5, 1e-4], [10.0, 10.0, 20.0, 10.0]), max_nfev=3000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _simulate(row, _PARAMS)