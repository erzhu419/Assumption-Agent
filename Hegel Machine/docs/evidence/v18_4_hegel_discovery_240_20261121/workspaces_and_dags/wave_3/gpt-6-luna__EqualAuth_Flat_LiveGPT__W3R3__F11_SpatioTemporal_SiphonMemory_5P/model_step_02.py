import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    tau_scale = 2.0
    u_n = float(row['u_now']) / tau_scale
    sg = float(row['s_grad'])
    j_siph = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def deriv(xv, zv):
        return (-a_slow * xv + b_couple * zv + j_siph,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    starts = [np.array([2.0, 0.5, 0.5, 0.5, 1.0]), np.array([1.0, 0.2, 1.0, 1.0, 2.0]), np.array([3.0, 0.8, 0.2, 0.2, 0.5])]
    fits = [least_squares(residual, s, bounds=([0.0, 0.0, 0.0, 0.0, 0.03], [20.0, 3.0, 10.0, 10.0, 20.0]), max_nfev=5000) for s in starts]
    return min(fits, key=lambda q: float(np.sum(q.fun*q.fun))).x

def predict(row):
    return float(_simulate(row, fit([row])))