import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(p, row):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    s = float(row['s_grad'])
    u = float(row['u_now'])  # tau_scale = 1.0
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(32, int(math.ceil(160.0 * H)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive,
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
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_with(p, r) for r in train_rows]) - y
    lower = np.array([0.001, 0.001, 0.01, 0.001, 0.01])
    upper = np.array([5.0, 5.0, 20.0, 0.999, 10.0])
    starts = [np.array([0.5, 0.5, 6.5, 0.70, 0.5]),
              np.array([0.2, 1.0, 5.0, 0.65, 1.0]),
              np.array([1.0, 0.2, 8.0, 0.75, 0.2])]
    best = None
    for p0 in starts:
        result = least_squares(residual, np.clip(p0, lower + 1e-8, upper - 1e-8),
                               bounds=(lower, upper), max_nfev=3000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _predict_with(_PARAMS, row)
