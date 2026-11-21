import math
import numpy as np
from scipy.optimize import least_squares

# Six fitted quantities: four ODE rates plus the siphon magnitude and sink coupling.
def _rhs(state, u, s, p):
    x, z = state
    a, b, c, tau, vmax, rho = p
    j = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return np.array([-a * x + b * z + c * u + j, (0.52 * x - z) / tau], dtype=float)

def _simulate(row, p):
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.01)))
    dt = H / n
    state = np.array([float(row['x_now']), float(row.get('mem_conv') or 0.0)], dtype=float)
    u, s = float(row['u_now']), float(row['s_grad'])
    for _ in range(n):
        k1 = _rhs(state, u, s, p)
        k2 = _rhs(state + 0.5 * dt * k1, u, s, p)
        k3 = _rhs(state + 0.5 * dt * k2, u, s, p)
        k4 = _rhs(state + dt * k3, u, s, p)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    def residual(p):
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows], dtype=float)
    starts = [np.array([0.3, 0.05, 0.1, 1.0, 5.0, 0.5]), np.array([0.1, 0.2, 0.2, 2.0, 8.0, 0.7])]
    fits = [least_squares(residual, q, bounds=([0.001, -3.0, -3.0, 0.03, 0.001, 0.0], [3.0, 3.0, 3.0, 20.0, 30.0, 1.5]), max_nfev=3000) for q in starts]
    return min(fits, key=lambda z: np.sum(z.fun*z.fun)).x

def predict(row):
    return _simulate(row, _fitted_params)
