import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    def rhs(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    n = max(40, int(math.ceil(H * 200)))
    dt = H / n
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + dt*k1x/2, z + dt*k1z/2)
        k3x, k3z = rhs(x + dt*k2x/2, z + dt*k2z/2)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_pred(r, p) for r in rows]) - y
    lo = np.array([0.001, -3.0, 0.01, 0.001, 0.03])
    hi = np.array([4.0, 3.0, 20.0, 0.999, 5.0])
    starts = [np.array([0.5, 0.3, 6.0, 0.7, 0.5]), np.array([0.2, 0.8, 4.0, 0.6, 1.0]), np.array([1.0, -0.2, 8.0, 0.75, 0.25])]
    best = None
    for p0 in starts:
        result = least_squares(residual, np.clip(p0, lo+1e-8, hi-1e-8), bounds=(lo, hi), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    return float(_pred(row, fit([row])))