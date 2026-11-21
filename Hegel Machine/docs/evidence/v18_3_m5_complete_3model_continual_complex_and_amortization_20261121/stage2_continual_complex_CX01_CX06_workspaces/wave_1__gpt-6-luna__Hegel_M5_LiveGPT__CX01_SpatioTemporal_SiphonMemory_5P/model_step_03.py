import math
import numpy as np
from scipy.optimize import least_squares

def _predict(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u = float(row['u_now'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    dt = 0.004
    n = max(1, int(math.ceil(H / dt)))
    h = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive, (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * h * k1)
        k3 = f(state + 0.5 * h * k2)
        k4 = f(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.8, 0.4, 6.0, 0.70, 0.25]), bounds=([0.01, 0.0, 0.1, 0.0, 0.01], [5.0, 5.0, 20.0, 0.99, 3.0]), max_nfev=3000)
    return result.x

def predict(row):
    return _predict(row, fit(_TRAIN_ROWS))