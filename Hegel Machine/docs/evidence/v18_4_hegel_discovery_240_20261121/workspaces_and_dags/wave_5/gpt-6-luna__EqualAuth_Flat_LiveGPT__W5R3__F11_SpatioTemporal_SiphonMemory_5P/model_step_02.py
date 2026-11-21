import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 2.0

def _integrate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / TAU_SCALE
    grad = float(row['s_grad'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    flux = vmax * (1.0 - rho * grad / (1.0 + grad)) * (u_n / (1.2 + u_n))
    n = max(8, int(math.ceil(H / 0.005)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + flux,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    def residual(p):
        return np.array([_integrate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.5, 0.5, 1.0]),
                           bounds=(np.array([0.0, 0.0, 0.0, 0.0, 0.02]),
                                   np.array([20.0, 1.0, 10.0, 10.0, 20.0])),
                           max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return result.x

def predict(row):
    return _integrate(row, fit_params)
