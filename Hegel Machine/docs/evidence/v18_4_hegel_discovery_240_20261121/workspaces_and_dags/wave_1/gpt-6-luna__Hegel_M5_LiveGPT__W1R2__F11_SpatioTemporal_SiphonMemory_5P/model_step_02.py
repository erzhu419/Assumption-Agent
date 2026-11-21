import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _predict_params(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u = row['u_now']
    s = row['s_grad']
    H = row['H']
    x = row['x_now']
    z = row['mem_conv']
    u_n = u  # tau_scale = 1.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(40, int(math.ceil(H * 200)))
    dt = H / n
    def rhs(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _P
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    lo = np.array([0.001, 0.0, 1.0, 0.05, 0.01])
    hi = np.array([10.0, 10.0, 15.0, 0.95, 10.0])
    starts = [np.array([1.0, 0.4, 6.0, 0.7, 0.5]),
              np.array([0.5, 1.0, 5.0, 0.5, 1.0]),
              np.array([2.0, 0.2, 8.0, 0.8, 0.2])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=3000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _P = best[1]
    return _P.copy()

def predict(row):
    if _P is None:
        raise RuntimeError('fit must be called before predict')
    return _predict_params(row, _P)
