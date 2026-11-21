import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now']) / 2.0
    sg = float(row['s_grad'])
    drive = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.03])
    upper = np.array([20.0, 0.999, 10.0, 10.0, 20.0])
    def residual(p):
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows])
    starts = [np.array([2.0, 0.5, 1.0, 0.5, 1.0]), np.array([3.0, 0.2, 0.5, 1.0, 2.0]), np.array([1.0, 0.8, 2.0, 0.2, 0.5])]
    results = [least_squares(residual, s, bounds=(lower, upper), max_nfev=3000) for s in starts]
    return min(results, key=lambda q: float(np.dot(q.fun, q.fun))).x

def predict(row):
    return float(_simulate(row, _fitted_params))