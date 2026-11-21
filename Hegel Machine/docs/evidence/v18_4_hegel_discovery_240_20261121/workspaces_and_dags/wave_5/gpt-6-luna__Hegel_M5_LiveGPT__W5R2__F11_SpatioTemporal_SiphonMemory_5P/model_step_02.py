import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    tau_scale = 0.005
    def simulate(row, p):
        a_slow, b_couple, vmax, rho, tau_fast = p
        x = float(row['x_now'])
        z = float(row['mem_conv'])
        H = float(row['H'])
        u_n = float(row['u_now']) / tau_scale
        s = float(row['s_grad'])
        drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
        def deriv(xv, zv):
            return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
        n = max(100, int(math.ceil(H / max(tau_fast * 0.12, 1e-5))))
        dt = H / n
        for _ in range(n):
            k1x, k1z = deriv(x, z)
            k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
            k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
            k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
            x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
            z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
        return x
    X = list(train_rows)
    y = np.array([float(r['y']) for r in X])
    lo = np.array([0.0, -2.0, 1.0, 0.05, 0.0001])
    hi = np.array([10.0, 2.0, 15.0, 0.95, 0.5])
    p0 = np.array([0.5, 0.2, 6.0, 0.7, 0.01])
    def residual(p):
        return np.array([simulate(r, p) for r in X]) - y
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=3000)
    return result.x.tolist()

def predict(row):
    p = fit(_FIT_ROWS)
    a_slow, b_couple, vmax, rho, tau_fast = p
    tau_scale = 0.005
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    u_n = float(row['u_now']) / tau_scale
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + drive, (0.52 * xv - zv) / tau_fast
    n = max(100, int(math.ceil(H / max(tau_fast * 0.12, 1e-5))))
    dt = H / n
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return float(x)

_FIT_ROWS = []
