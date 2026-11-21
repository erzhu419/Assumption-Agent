import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now'])  # characteristic tau_scale is 1.0
    sg = float(row['s_grad'])
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    J = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    n = max(8, int(math.ceil(H / 0.002)))
    dt = H / n
    def rhs(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - ys
    starts = [np.array([2.0, 0.5, 0.5, 0.5, 0.5]),
              np.array([3.0, 0.2, 1.0, 0.3, 1.0]),
              np.array([1.0, 0.8, 0.2, 1.0, 0.2])]
    sols = [least_squares(residual, s, bounds=([0.0, 0.0, 0.0, 0.0, 0.01], [20.0, 1.0, 10.0, 10.0, 10.0]), max_nfev=5000) for s in starts]
    return min(sols, key=lambda q: np.sum(q.fun*q.fun)).x

def predict(row):
    return float(_simulate(row, fit_params))