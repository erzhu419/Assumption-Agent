import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    u_n = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    # Resolve the fast lag timescale with stable, sufficiently fine RK4 steps.
    n = max(1, int(math.ceil(H / max(tau_fast / 5.0, 1e-7))))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J,
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
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0002])
    upper = np.array([20.0, 0.999, 10.0, 10.0, 0.05])
    starts = [np.array([2.0, 0.3, 0.5, 0.5, 0.005]),
              np.array([4.0, 0.5, 1.0, 1.0, 0.01])]
    results = [least_squares(residual, np.minimum(np.maximum(p, lower+1e-9), upper-1e-9),
                             bounds=(lower, upper), max_nfev=3000) for p in starts]
    return min(results, key=lambda r: np.sum(r.fun**2)).x

def predict(row):
    return _simulate(row, fit([dict(row, y=0.0)]))