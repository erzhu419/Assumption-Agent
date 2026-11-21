import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive,
                         (0.52 * xx - zz) / tau_fast])
    n = max(20, int(np.ceil(H / 0.005)))
    dt = H / n
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [[0.5, 0.2, 6.0, 0.7, 0.8], [1.0, 0.5, 4.0, 0.5, 1.5], [0.2, 0.8, 8.0, 0.8, 0.4]]
    fits = [least_squares(residual, x0, bounds=([0.001, -3.0, 0.01, 0.001, 0.02], [5.0, 3.0, 30.0, 0.999, 10.0]), max_nfev=3000) for x0 in starts]
    return min(fits, key=lambda f: np.sum(f.fun**2)).x

def predict(row):
    return _simulate(row, fit(_TRAIN_ROWS))