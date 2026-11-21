import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def predict_params(p, row):
        vmax, rho, a_slow, b_couple, tau_fast = p
        u = float(row['u_now'])
        s = float(row['s_grad'])
        H = max(0.0, float(row['H']))
        x = float(row['x_now'])
        z = float(row['mem_conv'] or 0.0)
        J = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        n = max(1, int(math.ceil(H / 0.004)))
        dt = H / n
        def deriv(xx, zz):
            return -a_slow * xx + b_couple * zz + J, (0.52 * xx - zz) / tau_fast
        for _ in range(n):
            k1x, k1z = deriv(x, z)
            k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
            k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
            k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
            x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
            z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
        return x
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([predict_params(p, r) for r in rows]) - y
    starts = [[2.0, 0.5, 0.1, 0.1, 1.0], [5.0, 0.7, 0.2, 0.2, 0.5], [1.0, 0.3, 0.05, 0.05, 2.0]]
    fits = [least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, 0.0, 0.02], [100.0, 1.0, 5.0, 5.0, 100.0]), max_nfev=4000) for p0 in starts]
    best = min(fits, key=lambda q: float(np.dot(q.fun, q.fun)))
    return best.x

def predict(row):
    p = fit(_hegel_train_rows)
    return float(_predict_with_params(p, row))