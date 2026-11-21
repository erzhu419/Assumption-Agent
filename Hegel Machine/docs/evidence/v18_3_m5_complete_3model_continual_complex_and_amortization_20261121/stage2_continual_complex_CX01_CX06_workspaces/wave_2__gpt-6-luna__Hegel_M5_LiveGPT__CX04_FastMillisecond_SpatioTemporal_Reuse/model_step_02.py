import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.005

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    u_n = float(row['u_now']) / TAU_SCALE
    sg = float(row['s_grad'])
    j_siph = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    n = max(100, int(math.ceil(H * 8.0 / tau_fast)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + j_siph,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return float(state[0])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lower = np.array([0.0, -10.0, 0.0, 0.05, 0.0005])
    upper = np.array([10.0, 10.0, 20.0, 0.95, 0.1])
    x0 = np.array([0.5, 0.5, 6.5, 0.7, 0.01])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=300)
    return result.x

def predict(row):
    return _simulate(row, fit_parameters)
