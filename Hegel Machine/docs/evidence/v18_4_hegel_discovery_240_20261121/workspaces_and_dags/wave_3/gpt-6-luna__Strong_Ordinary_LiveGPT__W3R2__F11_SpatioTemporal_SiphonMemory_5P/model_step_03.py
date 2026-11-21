import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        z += (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5, 0.2, 0.08]),
        bounds=(np.array([0.0, 0.0, 0.0, 0.0, 0.005]), np.array([10.0, 1.0, 10.0, 10.0, 2.0])),
        max_nfev=3000)
    params = result.x.copy()
    def predict(row):
        return float(_simulate(row, params))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')