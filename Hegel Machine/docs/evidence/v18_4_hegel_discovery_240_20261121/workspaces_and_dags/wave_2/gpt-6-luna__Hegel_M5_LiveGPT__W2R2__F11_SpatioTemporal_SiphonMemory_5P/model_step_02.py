import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    n = max(20, int(math.ceil(H * 160)))
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
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in rows]) - y
    lower = np.array([0.001, -3.0, 0.01, 0.001, 0.03])
    upper = np.array([5.0, 3.0, 20.0, 0.999, 10.0])
    starts = [np.array([0.5, 0.2, 6.0, 0.7, 0.5]),
              np.array([1.0, 0.0, 4.0, 0.5, 1.0]),
              np.array([0.2, 0.8, 8.0, 0.75, 0.25])]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=3000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]

def predict(row):
    return _predict_params(row, fit_rows_params)
