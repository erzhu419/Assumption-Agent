import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lo = np.array([0.0, 0.0, 0.0, 0.0, 0.0002])
    hi = np.array([20.0, 0.999, 20.0, 20.0, 2.0])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([2.0, 0.5, 0.5, 0.5, 0.05]), np.array([1.0, 0.3, 1.0, 1.0, 0.2]), np.array([4.0, 0.7, 0.2, 0.2, 0.01])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=5000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.array(best[1], dtype=float)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))